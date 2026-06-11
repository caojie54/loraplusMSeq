# loraplusMSeq

Sequential LoRA plus compensation-module training for `commonsense170k`.

For each n-batch, the trainer first updates LoRA only and accumulates module-wise LoRA gradient pressure. It then selects compensation modules and replays the exact same cached n-batch with LoRA frozen and only the selected original module weights trainable.

Methods:

- `alpha`: sequential LoRA phase, select modules by accumulated LoRA gradient pressure, then module replay.
- `static_random`: sequential baseline; after the first LoRA n-batch, sample one fixed random module set and reuse it for all module replay phases.
- `dynamic_random`: sequential baseline; sample a fresh random module set after every LoRA n-batch.
- `lora`: LoRA-only baseline. The provided script runs `2` epochs for fair compute comparison with one sequential epoch.

Run one single-GPU experiment:

```bash
cd /mnt/petrelfs/caojie1/projects/loraplusMSeq
srun -p sciverse_agent --job-name=test_df --ntasks-per-node=1 --cpus-per-task=24 --gres=gpu:1 bash task.sh
```

Run all four requested jobs sequentially:

```bash
cd /mnt/petrelfs/caojie1/projects/loraplusMSeq
srun -p sciverse_agent --job-name=test_df --ntasks-per-node=1 --cpus-per-task=24 --gres=gpu:1 bash exps/commonsense170k/run_all.sh
```

Outputs are written under `/mnt/dhwfile/raise/user/caojie/loraplusMSeq/outputs/commonsense170k` by default.

After training, each run also writes `trainable_params.json` in the run output directory. It records the LoRA trainable parameter count once, module trainable parameter counts by replay block for `alpha` and `dynamic_random`, a single reused module count for `static_random`, plus module-phase, phase-average, and whole-training average trainable parameter counts. The key averages are also copied into `train_metrics.json`.
