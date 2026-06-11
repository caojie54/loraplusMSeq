"""Single-GPU sequential trainer for LoRA then compensation-module replay."""

from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .module_selection import CompensationModuleManager


Batch = Dict[str, torch.Tensor]
UpdateBatch = List[Batch]


class SequentialLoraPlusMTrainer:
    """Train each n-batch with LoRA first, then replay it for selected base modules."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset,
        data_collator,
        module_manager: CompensationModuleManager,
        output_dir: str,
        batch_size: int,
        gradient_accumulation_steps: int,
        num_train_epochs: float,
        learning_rate: float,
        module_learning_rate: float,
        weight_decay: float,
        warmup_steps: int,
        lr_scheduler_type: str,
        selection_interval: int,
        logging_steps: int,
        seed: int,
        bf16: bool,
        max_grad_norm: float = 1.0,
        dataloader_num_workers: int = 0,
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.data_collator = data_collator
        self.module_manager = module_manager
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.num_train_epochs = num_train_epochs
        self.learning_rate = learning_rate
        self.module_learning_rate = module_learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.lr_scheduler_type = lr_scheduler_type
        self.selection_interval = max(1, selection_interval)
        self.logging_steps = max(1, logging_steps)
        self.seed = seed
        self.bf16 = bf16
        self.max_grad_norm = max_grad_norm
        self.dataloader_num_workers = dataloader_num_workers

        self.global_step = 0
        self.lora_step = 0
        self.module_step = 0
        self.total_updates = 0
        self.total_lora_updates = 0
        self._static_random_names: Optional[List[str]] = None
        self._module_optimizer_key: Optional[Tuple[str, ...]] = None
        self._module_optimizer: Optional[torch.optim.Optimizer] = None
        self._train_log_path = os.path.join(output_dir, "train_log.jsonl")
        self._trainable_params_path = os.path.join(output_dir, "trainable_params.json")
        self._lora_trainable_params = 0
        self._module_param_records: List[Dict[str, object]] = []
        self._static_module_params_recorded = False

    def train(self) -> Dict[str, object]:
        os.makedirs(self.output_dir, exist_ok=True)
        if os.path.exists(self._train_log_path):
            os.remove(self._train_log_path)
        if os.path.exists(self._trainable_params_path):
            os.remove(self._trainable_params_path)
        self._module_param_records = []
        self._static_module_params_recorded = False

        start = time.time()
        self.model.train()
        self.module_manager.set_lora_phase()
        self._record_lora_trainable_params()

        lora_optimizer = self._create_optimizer(self.module_manager.lora_named_parameters(), self.learning_rate)
        self.total_lora_updates = self._planned_lora_updates()
        replay_multiplier = 1 if self.module_manager.method == "lora" else 2
        self.total_updates = self.total_lora_updates * replay_multiplier

        losses: Dict[str, List[float]] = {"lora": [], "module": []}
        block: List[UpdateBatch] = []
        block_scores: Dict[str, float] = defaultdict(float)

        progress = tqdm(total=self.total_updates, desc="seq-train", dynamic_ncols=True)
        epoch_index = 0
        while self.lora_step < self.total_lora_updates:
            dataloader = self._build_dataloader(epoch_index)
            for update_batch in self._iter_update_batches(dataloader):
                if self.lora_step >= self.total_lora_updates:
                    break

                lora_loss = self._run_lora_update(update_batch, lora_optimizer, block_scores)
                losses["lora"].append(lora_loss)
                block.append(update_batch)
                self.lora_step += 1
                self.global_step += 1
                progress.update(1)
                self._maybe_log(losses, phase="lora")

                if len(block) >= self.selection_interval:
                    self._run_module_replay_block(block, block_scores, losses, progress)
                    block = []
                    block_scores = defaultdict(float)

            epoch_index += 1

        if block:
            self._run_module_replay_block(block, block_scores, losses, progress)

        progress.close()
        self.module_manager.set_lora_phase()
        runtime = time.time() - start
        trainable_param_stats = self._save_trainable_param_stats()
        metrics = {
            "train_runtime": runtime,
            "train_updates": float(self.global_step),
            "lora_updates": float(self.lora_step),
            "module_updates": float(self.module_step),
            "train_lora_loss": float(sum(losses["lora"]) / max(1, len(losses["lora"]))),
            "train_module_loss": float(sum(losses["module"]) / max(1, len(losses["module"]))),
            "lora_trainable_params": trainable_param_stats["lora_trainable_params"],
            "module_average_trainable_params": trainable_param_stats["module_average_trainable_params"],
            "phase_average_trainable_params": trainable_param_stats["phase_average_trainable_params"],
            "overall_average_trainable_params": trainable_param_stats["overall_average_trainable_params"],
        }
        with open(os.path.join(self.output_dir, "train_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"[loraplusMSeq] train metrics: {metrics}", flush=True)
        return metrics

    def _planned_lora_updates(self) -> int:
        micro_batches_per_epoch = math.ceil(len(self.train_dataset) / self.batch_size)
        updates_per_epoch = math.ceil(micro_batches_per_epoch / self.gradient_accumulation_steps)
        return max(1, math.ceil(updates_per_epoch * self.num_train_epochs))

    def _build_dataloader(self, epoch_index: int) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(self.seed + epoch_index)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=self.dataloader_num_workers,
            pin_memory=torch.cuda.is_available(),
            generator=generator,
        )

    def _iter_update_batches(self, dataloader: DataLoader) -> Iterable[UpdateBatch]:
        update_batch: UpdateBatch = []
        for micro_batch in dataloader:
            update_batch.append(micro_batch)
            if len(update_batch) == self.gradient_accumulation_steps:
                yield update_batch
                update_batch = []
        if update_batch:
            yield update_batch

    def _run_lora_update(
        self,
        update_batch: UpdateBatch,
        optimizer: torch.optim.Optimizer,
        block_scores: Dict[str, float],
    ) -> float:
        self.module_manager.set_lora_phase()
        self._set_optimizer_lr(optimizer, self.learning_rate)
        optimizer.zero_grad(set_to_none=True)

        loss_value = self._backward_update_batch(update_batch)
        # The alpha strategy uses the LoRA gradient pressure accumulated over the whole n-batch.
        for name, score in self.module_manager.score_from_lora_grads().items():
            block_scores[name] += score

        self._clip_gradients([param for _, param in self.module_manager.lora_named_parameters()])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return loss_value

    def _run_module_replay_block(
        self,
        block: List[UpdateBatch],
        block_scores: Dict[str, float],
        losses: Dict[str, List[float]],
        progress: tqdm,
    ) -> None:
        if self.module_manager.method == "lora":
            return

        self._select_modules_for_block(block_scores)
        self.module_manager.set_module_phase()
        self._record_module_trainable_params(block)
        module_optimizer = self._get_module_optimizer()
        if module_optimizer is None:
            return

        for update_batch in block:
            self._set_optimizer_lr(module_optimizer, self.module_learning_rate)
            module_optimizer.zero_grad(set_to_none=True)
            module_loss = self._backward_update_batch(update_batch)
            module_params = [param for _, param in self.module_manager.selected_module_named_parameters()]
            self._clip_gradients(module_params)
            module_optimizer.step()
            module_optimizer.zero_grad(set_to_none=True)

            losses["module"].append(module_loss)
            self.module_step += 1
            self.global_step += 1
            progress.update(1)
            self._maybe_log(losses, phase="module")

        self.module_manager.freeze_all_compensation()

    def _record_lora_trainable_params(self) -> None:
        self._lora_trainable_params = self._count_unique_params(self.module_manager.lora_named_parameters())
        current_trainable = self._count_current_trainable_params()
        print(
            f"[loraplusMSeq] lora trainable params={self._lora_trainable_params} "
            f"current_trainable={current_trainable}",
            flush=True,
        )

    def _record_module_trainable_params(self, block: List[UpdateBatch]) -> None:
        method = self.module_manager.method
        if method == "static_random" and self._static_module_params_recorded:
            return

        module_trainable_params = self._count_unique_params(self.module_manager.selected_module_named_parameters())
        current_trainable = self._count_current_trainable_params()
        selected_record = self.module_manager.selection_records[-1] if self.module_manager.selection_records else {}
        record = {
            "block_index": len(self._module_param_records) + 1,
            "lora_step": int(self.lora_step),
            "module_step_start": int(self.module_step),
            "block_update_count": int(len(block)),
            "method": method,
            "selection_reason": selected_record.get("reason"),
            "applies_to_all_module_blocks": method == "static_random",
            "selected_module_count": int(len(self.module_manager.selected_names)),
            "selected_names": list(self.module_manager.selected_names),
            "module_trainable_params": int(module_trainable_params),
            "current_trainable_params": int(current_trainable),
            "selected_param_ratio": float(module_trainable_params / max(1, self.module_manager.total_candidate_params)),
        }
        self._module_param_records.append(record)
        if method == "static_random":
            self._static_module_params_recorded = True
        print(
            f"[loraplusMSeq] module trainable params block={record['block_index']} "
            f"params={module_trainable_params} current_trainable={current_trainable}",
            flush=True,
        )

    def _save_trainable_param_stats(self) -> Dict[str, object]:
        module_average = self._module_average_trainable_params()
        module_weighted_total = self._module_weighted_trainable_total(module_average)
        phase_average = self._phase_average_trainable_params(module_average)
        overall_updates = self.lora_step + self.module_step
        if overall_updates > 0:
            overall_average = (
                self._lora_trainable_params * self.lora_step + module_weighted_total
            ) / overall_updates
        else:
            overall_average = float(self._lora_trainable_params)

        stats: Dict[str, object] = {
            "method": self.module_manager.method,
            "lora_trainable_params": int(self._lora_trainable_params),
            "module_trainable_params_by_block": list(self._module_param_records),
            "module_average_trainable_params": module_average,
            "phase_average_trainable_params": phase_average,
            "overall_average_trainable_params": float(overall_average),
            "lora_updates": int(self.lora_step),
            "module_updates": int(self.module_step),
            "total_model_params": int(self._count_total_model_params()),
            "total_candidate_module_params": int(self.module_manager.total_candidate_params),
        }
        with open(self._trainable_params_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"[loraplusMSeq] trainable param stats saved to {self._trainable_params_path}", flush=True)
        return stats

    def _module_average_trainable_params(self) -> Optional[float]:
        if self.module_manager.method == "lora" or not self._module_param_records:
            return None
        if self.module_manager.method == "static_random":
            return float(self._module_param_records[0]["module_trainable_params"])

        total_weight = sum(int(record["block_update_count"]) for record in self._module_param_records)
        if total_weight <= 0:
            return 0.0
        weighted_sum = sum(
            int(record["module_trainable_params"]) * int(record["block_update_count"])
            for record in self._module_param_records
        )
        return float(weighted_sum / total_weight)

    def _phase_average_trainable_params(self, module_average: Optional[float]) -> float:
        if self.module_manager.method == "lora" or module_average is None:
            return float(self._lora_trainable_params)
        return float((self._lora_trainable_params + module_average) / 2.0)

    def _module_weighted_trainable_total(self, module_average: Optional[float]) -> float:
        if self.module_manager.method == "lora" or self.module_step <= 0 or module_average is None:
            return 0.0
        if self.module_manager.method == "static_random":
            return float(module_average * self.module_step)
        return float(
            sum(
                int(record["module_trainable_params"]) * int(record["block_update_count"])
                for record in self._module_param_records
            )
        )

    def _count_unique_params(self, named_params: Iterable[Tuple[str, torch.nn.Parameter]]) -> int:
        seen = set()
        total = 0
        for _, param in named_params:
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            total += param.numel()
        return int(total)

    def _count_current_trainable_params(self) -> int:
        return int(sum(param.numel() for param in self.model.parameters() if param.requires_grad))

    def _count_total_model_params(self) -> int:
        return int(sum(param.numel() for param in self.model.parameters()))

    def _select_modules_for_block(self, block_scores: Dict[str, float]) -> None:
        method = self.module_manager.method
        if method == "alpha":
            self.module_manager.select_by_alpha_scores(dict(block_scores), step=self.lora_step)
        elif method == "dynamic_random":
            self.module_manager.select_random(step=self.lora_step, reason="seq_dynamic_random")
        elif method == "static_random":
            if self._static_random_names is None:
                self._static_random_names = self.module_manager.select_random(
                    step=self.lora_step,
                    reason="seq_static_random",
                )
            else:
                self.module_manager.record_selection(
                    self._static_random_names,
                    step=self.lora_step,
                    reason="seq_static_random_reuse",
                )
        else:
            raise ValueError(f"Unsupported method: {method}")

    def _backward_update_batch(self, update_batch: UpdateBatch) -> float:
        total_loss = 0.0
        scale = 1.0 / max(1, len(update_batch))
        for micro_batch in update_batch:
            inputs = self._move_to_device(micro_batch)
            with self._autocast_context():
                outputs = self.model(**inputs)
                loss = outputs.loss
            total_loss += float(loss.detach().float().item())
            (loss * scale).backward()
        return total_loss / max(1, len(update_batch))

    def _autocast_context(self):
        if not torch.cuda.is_available():
            return torch.autocast(device_type="cpu", enabled=False)
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16)

    def _move_to_device(self, batch: Batch) -> Batch:
        device = self._model_input_device()
        return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}

    def _model_input_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _create_optimizer(
        self,
        named_params: List[Tuple[str, torch.nn.Parameter]],
        learning_rate: float,
    ) -> torch.optim.Optimizer:
        decay_params: List[torch.nn.Parameter] = []
        no_decay_params: List[torch.nn.Parameter] = []
        for name, param in named_params:
            if not param.requires_grad:
                continue
            if self._use_weight_decay(name):
                decay_params.append(param)
            else:
                no_decay_params.append(param)
        param_groups = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(param_groups, lr=learning_rate)

    def _get_module_optimizer(self) -> Optional[torch.optim.Optimizer]:
        named_params = self.module_manager.selected_module_named_parameters()
        if not named_params:
            return None
        key = tuple(name for name, _ in named_params)
        if self._module_optimizer is None or key != self._module_optimizer_key:
            self._module_optimizer = self._create_optimizer(named_params, self.module_learning_rate)
            self._module_optimizer_key = key
        return self._module_optimizer

    def _use_weight_decay(self, name: str) -> bool:
        no_decay_terms = ("bias", "layer_norm.weight", "LayerNorm.weight", "norm.weight")
        return not any(term in name for term in no_decay_terms)

    def _set_optimizer_lr(self, optimizer: torch.optim.Optimizer, base_lr: float) -> None:
        lr = self._lr_for_step(self.global_step, base_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

    def _lr_for_step(self, completed_steps: int, base_lr: float) -> float:
        if self.lr_scheduler_type == "constant":
            return base_lr
        if self.lr_scheduler_type == "constant_with_warmup":
            if self.warmup_steps <= 0:
                return base_lr
            return base_lr * min(1.0, completed_steps / self.warmup_steps)
        if self.lr_scheduler_type == "linear":
            if self.warmup_steps > 0 and completed_steps < self.warmup_steps:
                return base_lr * completed_steps / self.warmup_steps
            remaining = max(0, self.total_updates - completed_steps)
            decay_steps = max(1, self.total_updates - self.warmup_steps)
            return base_lr * remaining / decay_steps
        raise ValueError(f"Unsupported lr_scheduler_type: {self.lr_scheduler_type}")

    def _clip_gradients(self, params: List[torch.nn.Parameter]) -> None:
        if self.max_grad_norm and self.max_grad_norm > 0:
            params_with_grad = [p for p in params if p.grad is not None]
            if params_with_grad:
                torch.nn.utils.clip_grad_norm_(params_with_grad, self.max_grad_norm)

    def _maybe_log(self, losses: Dict[str, List[float]], phase: str) -> None:
        if self.global_step % self.logging_steps != 0:
            return
        record = {
            "step": self.global_step,
            "lora_step": self.lora_step,
            "module_step": self.module_step,
            "phase": phase,
            "lora_lr": self._lr_for_step(self.global_step, self.learning_rate),
            "module_lr": self._lr_for_step(self.global_step, self.module_learning_rate),
            "lora_loss": sum(losses["lora"][-self.logging_steps :]) / max(1, len(losses["lora"][-self.logging_steps :])),
            "module_loss": sum(losses["module"][-self.logging_steps :])
            / max(1, len(losses["module"][-self.logging_steps :])),
        }
        with open(self._train_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[loraplusMSeq] log {record}", flush=True)
