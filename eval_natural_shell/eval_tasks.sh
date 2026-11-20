
CUDA_VISIBLE_DEVICES=0 \
  MODEL_DIR=/data/user/cj/data2/experiments/loraplusMSeq/outputs/natural-reasoning-20k/gemma-4-12b/gemma-4-12b-seq-alpha-effectivepressure-candidate0.06579-moduleoptdtype-fp32-qkvogateupdown-rank64-natural-reasoning-20k-epoch1-seed0-ratio0.02193-block50-loralr1e-4-modulelr1e-5 \
  bash eval_natural_shell/natural_reasoning_eval-sglang.sh

CUDA_VISIBLE_DEVICES=1 \
  MODEL_DIR=/data/user/cj/data2/experiments/loraplusMSeq/outputs/natural-reasoning-20k/gemma-4-12b/gemma-4-12b-seq-alpha-gradnorm-candidate0.0-moduleoptdtype-fp32-qkvogateupdown-rank64-natural-reasoning-20k-epoch1-seed0-ratio0.02193-block50-loralr1e-4-modulelr1e-5 \
  bash eval_natural_shell/natural_reasoning_eval-sglang.sh

CUDA_VISIBLE_DEVICES=2 \
  MODEL_DIR=/data/user/cj/data2/experiments/loraplusMSeq/outputs/natural-reasoning-20k/gemma-4-12b/gemma-4-12b-seq-dynamic_random-moduleoptdtype-fp32-qkvogateupdown-rank64-natural-reasoning-20k-epoch1-seed0-ratio0.02193-block50-loralr1e-4-modulelr1e-5 \
  bash eval_natural_shell/natural_reasoning_eval-sglang.sh

CUDA_VISIBLE_DEVICES=3 \
  MODEL_DIR=/data/user/cj/data2/experiments/loraplusMSeq/outputs/natural-reasoning-20k/gemma-4-12b/gemma-4-12b-seq-lora-qkvogateupdown-rank64-natural-reasoning-20k-epoch2-seed0-ratio0.02193-block50-loralr1e-4-modulelr1e-5 \
  bash eval_natural_shell/natural_reasoning_eval-sglang.sh

CUDA_VISIBLE_DEVICES=4 \
  MODEL_DIR=/data/user/cj/data2/experiments/loraplusMSeq/outputs/natural-reasoning-20k/gemma-4-12b/gemma-4-12b-seq-static_random-moduleoptdtype-fp32-qkvogateupdown-rank64-natural-reasoning-20k-epoch1-seed0-ratio0.02193-block50-loralr1e-4-modulelr1e-5 \
  bash eval_natural_shell/natural_reasoning_eval-sglang.sh

CUDA_VISIBLE_DEVICES=5 \
  MODEL_DIR=/data/user/cj/data2/experiments/misa/outputs/natural-reasoning-20k/gemma-4-12b/misa-gemma-4-12b-natural-reasoning-20k-ratio0.02193-interval50-eta1.0-samplelast2-epoch2-seed0-lr1e-5 \
  bash eval_natural_shell/natural_reasoning_eval-sglang.sh

CUDA_VISIBLE_DEVICES=0 \
  MODEL_DIR=/data/user/cj/data2/experiments/loraplusMSeq/outputs/natural-reasoning-20k/gemma-4-12b/gemma-4-12b-seq-alpha-effectivepressure-candidate0.06579-moduleoptdtype-fp32-qkvogateupdown-rank64-natural-reasoning-20k-epoch1-seed0-ratio0.02193-block50-loralr1e-4-modulelr1e-5 \
  bash eval_natural_shell/natural_reasoning_eval-sglang.sh