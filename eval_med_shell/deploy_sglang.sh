# deploy on 8 GPUs
log_num=0
model_name=Qwen/Qwen3.5-4B-Base
port=28${log_num}35
CUDA_VISIBLE_DEVICES=0,4,5,6 python -m sglang.launch_server --model-path $model_name --port $port --mem-fraction-static 0.8 --dp 4 --tp 1  > sglang${log_num}.log 2>&1 &


# qwen3.5-4B model
log_num=0
model_name=Qwen/Qwen3.5-4B
port=28${log_num}35
CUDA_VISIBLE_DEVICES=0,4,5,6 python -m sglang.launch_server --model-path $model_name --port $port --mem-fraction-static 0.8 --dp 4 --tp 1  > sglang${log_num}.log 2>&1 &


log_num=0
model_name=google/gemma-3-4b-pt
port=28${log_num}35
CUDA_VISIBLE_DEVICES=1,2,4,5 python -m sglang.launch_server --model-path $model_name --port $port --mem-fraction-static 0.8 --dp 4 --tp 1  > sglang${log_num}.log 2>&1 &


log_num=0
model_name=google/gemma-3-4b-it
port=28${log_num}35
CUDA_VISIBLE_DEVICES=1,2,4,5 python -m sglang.launch_server --model-path $model_name --port $port --mem-fraction-static 0.8 --dp 4 --tp 1  > sglang${log_num}.log 2>&1 &

log_num=0
model_name=meta-llama/Llama-3.1-8B
port=28${log_num}35
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server --model-path $model_name --port $port --mem-fraction-static 0.8 --dp 1 --tp 1  > sglang${log_num}.log 2>&1 &

log_num=0
model_name=meta-llama/Llama-3.1-8B-Instruct
port=28${log_num}35
CUDA_VISIBLE_DEVICES=0,1,6,7 python -m sglang.launch_server --model-path $model_name --port $port --mem-fraction-static 0.8 --dp 4 --tp 1  > sglang${log_num}.log 2>&1 &

log_num=0
model_name=google/gemma-4-12B-it
port=28${log_num}35
CUDA_VISIBLE_DEVICES=2 python -m sglang.launch_server --model-path $model_name --port $port --mem-fraction-static 0.8 --skip-server-warmup --dp 1 --tp 1  > sglang${log_num}.log 2>&1 &

log_num=1
model_name=google/gemma-4-12B
port=28${log_num}35
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server --model-path $model_name --port $port --mem-fraction-static 0.8  --skip-server-warmup --dp 1 --tp 1  > sglang${log_num}.log 2>&1 &

log_num=0
model_name=/data/user/cj/data2/experiments/loraplusMSeq/outputs/medreason/gemma-3-4b-pt/gemma-3-4b-pt-lora-qkvogateupdown-rank32-medreason-epoch2-seed2-loraoptbf16-loralr1e-4/merged
port=28${log_num}35
CUDA_VISIBLE_DEVICES=2 python -m sglang.launch_server --model-path $model_name --port $port --mem-fraction-static 0.8 --dp 1 --tp 1  > sglang${log_num}.log 2>&1 &

log_num=1
model_name=/data/user/cj/data2/experiments/loraplusMSeq/outputs/medreason/gemma-3-4b-pt/gemma-3-4b-pt-lora-qkvogateupdown-rank128-medreason-epoch2-seed2-loraoptbf16-loralr1e-4/merged
port=28${log_num}35
CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server --model-path $model_name --port $port --mem-fraction-static 0.8 --dp 1 --tp 1  > sglang${log_num}.log 2>&1 &

log_num=2
model_name=/data/user/cj/data2/experiments/misa/outputs/medreason/gemma-3-4b-pt/misa-gemma-3-4b-pt-medreason-qkvogateupdown-ratio0.05544-interval50-eta1.0-samplelast2-epoch2-seed2-lr1e-5/merged
port=28${log_num}35
CUDA_VISIBLE_DEVICES=5 python -m sglang.launch_server --model-path $model_name --port $port --mem-fraction-static 0.8 --dp 1 --tp 1  > sglang${log_num}.log 2>&1 &

# stop sglang server and release GPU memory
pkill -f sglang
pkill -f multiprocessing.spawn

# eval
MODEL_DIR=google/gemma-4-12B-it PORT=28035 bash eval_med_shell/eval_medreason.sh

MODEL_DIR=/data/user/cj/data2/experiments/loraplusMSeq/outputs/medreason/gemma-3-4b-pt/gemma-3-4b-pt-lora-qkvogateupdown-rank128-medreason-epoch2-seed2-loraoptbf16-loralr1e-4/merged PORT=28135 bash eval_med_shell/eval_medreason.sh

MODEL_DIR=/data/user/cj/data2/experiments/misa/outputs/medreason/gemma-3-4b-pt/misa-gemma-3-4b-pt-medreason-qkvogateupdown-ratio0.05544-interval50-eta1.0-samplelast2-epoch2-seed2-lr1e-5/merged PORT=28235 bash eval_med_shell/eval_medreason.sh


CUDA_VISIBLE_DEVICES=6 \
  MODEL_DIR=/data/user/cj/data2/experiments/loraplusMSeq/outputs/medreason/gemma-3-4b-pt/gemma-3-4b-pt-lora-qkvogateupdown-rank32-medreason-epoch2-seed2-loraoptbf16-loralr1e-4/merged \
  TEST_BATCH_SIZE=1000 \
  bash eval_med_shell/eval_medreason_offline.sh

CUDA_VISIBLE_DEVICES=0 \
  MODEL_DIR=/data/user/cj/data2/experiments/loraplusMSeq/outputs/medreason/gemma-4-12b/gemma-4-12b-lora-qkvogateupdown-rank64-medreason-epoch2-seed0-loraoptbf16-loralr1e-4/merged \
  TEST_BATCH_SIZE=400 \
  bash eval_med_shell/eval_medreason_offline.sh

CUDA_VISIBLE_DEVICES=1 \
  MODEL_DIR=/data/user/cj/data2/experiments/loraplusMSeq/outputs/medreason/gemma-4-12b/gemma-4-12b-seq-alpha-lora_effective_update_pressure-candidate0.06579-ratio0.02193-block50-qkvogateupdown-rank64-medreason-epoch1-seed0-loraoptbf16-loralr1e-4-moduleoptfp32-reset_offload-modulelr1e-5/merged \
  TEST_BATCH_SIZE=400 \
  bash eval_med_shell/eval_medreason_offline.sh

CUDA_VISIBLE_DEVICES=2 \
  MODEL_DIR=/data/user/cj/data2/experiments/loraplusMSeq/outputs/medreason/gemma-4-12b/gemma-4-12b-seq-alpha-lora_grad_norm-candidate0.0-ratio0.02193-block50-qkvogateupdown-rank64-medreason-epoch1-seed0-loraoptbf16-loralr1e-4-moduleoptfp32-reset_offload-modulelr1e-5/merged \
  TEST_BATCH_SIZE=400 \
  bash eval_med_shell/eval_medreason_offline.sh

CUDA_VISIBLE_DEVICES=3 \
  MODEL_DIR=/data/user/cj/data2/experiments/loraplusMSeq/outputs/medreason/gemma-4-12b/gemma-4-12b-seq-dynamic_random-ratio0.02193-block50-qkvogateupdown-rank64-medreason-epoch1-seed0-loraoptbf16-loralr1e-4-moduleoptfp32-reset_offload-modulelr1e-5/merged \
  TEST_BATCH_SIZE=400 \
  bash eval_med_shell/eval_medreason_offline.sh

CUDA_VISIBLE_DEVICES=4 \
  MODEL_DIR=/data/user/cj/data2/experiments/loraplusMSeq/outputs/medreason/gemma-4-12b/gemma-4-12b-seq-static_random-ratio0.02193-block50-qkvogateupdown-rank64-medreason-epoch1-seed0-loraoptbf16-loralr1e-4-moduleoptfp32-reset_offload-modulelr1e-5/merged \
  TEST_BATCH_SIZE=400 \
  bash eval_med_shell/eval_medreason_offline.sh

CUDA_VISIBLE_DEVICES=5 \
  MODEL_DIR=/data/user/cj/data2/experiments/misa/outputs/medreason/gemma-4-12b/misa-gemma-4-12b-medreason-qkvogateupdown-ratio0.02193-interval50-eta1.0-samplelast2-epoch2-seed0-lr1e-5/merged \
  TEST_BATCH_SIZE=400 \
  bash eval_med_shell/eval_medreason_offline.sh

CUDA_VISIBLE_DEVICES=2 \
  MODEL_DIR=/data/user/cj/data2/experiments/loraplusMSeq/outputs/medreason/gemma-4-12b/gemma-4-12b-seq-alpha-lora_grad_norm-candidate0.0-ratio0.02193-block50-qkvogateupdown-rank64-medreason-epoch1-seed0-loraoptbf16-loralr1e-4-moduleoptfp32-reset_offload-modulelr1e-5/merged \
  TEST_BATCH_SIZE=400 \
  bash eval_med_shell/eval_medreason_offline.sh