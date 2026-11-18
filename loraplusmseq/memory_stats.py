"""Per-update CUDA peak allocated-memory statistics (not reserved or device-wide memory)."""

import json
from pathlib import Path

import torch


class CudaMemoryTracker:
    def __init__(self, device, log_path=None):
        self.device = torch.device(device)
        self.enabled = self.device.type == "cuda" and torch.cuda.is_available()
        self.log_path = Path(log_path) if log_path is not None else None
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text("", encoding="utf-8")
        self._stats = {}
        self._active = None
        self.latest = {}

    def begin(self, phase, step):
        if self._active is not None:
            raise RuntimeError("A memory measurement is already active")
        if self.enabled:
            # Reset the high-water mark, not the allocator or live tensors.
            torch.cuda.reset_peak_memory_stats(self.device)
        self._active = (phase, step)

    def end(self):
        if self._active is None:
            raise RuntimeError("No memory measurement is active")
        phase, step = self._active
        peak = torch.cuda.max_memory_allocated(self.device) / 1024**3 if self.enabled else 0.0
        self._active = None
        self.latest = {"phase": phase, "step": step, "peak_memory_gib": peak}
        for key in ("overall", phase):
            stats = self._stats.setdefault(key, {"count": 0, "peak_sum": 0.0, "max": 0.0})
            stats["count"] += 1
            stats["peak_sum"] += peak
            stats["max"] = max(stats["max"], peak)
            stats["peak"] = peak
        if self.log_path is not None:
            # Every update is recorded, independently of the loss logging interval.
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({**self.latest, "phase_memory": self.summary(phase)}) + "\n")
        return dict(self.latest)

    def summary(self, phase="overall"):
        stats = self._stats.get(phase, {})
        count = stats.get("count", 0)
        return {
            "memory_samples": count,
            "peak_memory_gib": stats.get("peak", 0.0),
            "max_memory_gib": stats.get("max", 0.0),
            "avg_peak_memory_gib": stats.get("peak_sum", 0.0) / max(1, count),
        }

    def cancel(self):
        """Discard a window when no optimizer update took place."""
        self._active = None

    def metrics(self):
        return {
            **self.summary(),
            "memory_device": str(self.device),
            "memory_sampling": "optimizer_update_end_after_zero_grad",
            "memory_by_phase": {phase: self.summary(phase) for phase in self._stats if phase != "overall"},
        }
