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
model_name=/data/user/cj/data2/experiments/loraplusMSeq/outputs/medreason/Qwen3-5-4B-Base/qwen3.5-4b-base-seq-alpha-lora_effective_update_pressure-candidate0.06-ratio0.02-block50-qkvogateupdown-rank64-medreason-epoch1-seed0-loraoptbf16-loralr1e-4-moduleoptfp32-reset_offload-modulelr1e-5/merged_sglang
port=28${log_num}35
CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server --model-path $model_name --port $port --mem-fraction-static 0.8 --dp 1 --tp 1  > sglang${log_num}.log 2>&1 &

log_num=0
model_name=/data/user/cj/data2/experiments/loraplusMSeq/outputs/medreason/llama-3-1-8b/llama-3-1-8b-seq-alpha-lora_effective_update_pressure-candidate0.06-qkvogateupdown-rank64-medreason-epoch1-ratio0.02-block50-seed0-loraoptbf16-moduleoptfp32-reset_offload-loralr1e-4-modulelr1e-5/merged
port=28${log_num}35
CUDA_VISIBLE_DEVICES=0,1,6,7 python -m sglang.launch_server --model-path $model_name --port $port --mem-fraction-static 0.8 --dp 4 --tp 1  > sglang${log_num}.log 2>&1 &

# stop sglang server and release GPU memory
pkill -f sglang
pkill -f multiprocessing.spawn

# eval
MODEL_DIR=/data/user/cj/data2/experiments/loraplusMSeq/outputs/medreason/Qwen3-5-4B-Base/qwen3.5-4b-base-seq-alpha-lora_effective_update_pressure-candidate0.06-ratio0.02-block50-qkvogateupdown-rank64-medreason-epoch1-seed0-loraoptbf16-loralr1e-4-moduleoptfp32-reset_offload-modulelr1e-5/merged_sglang bash eval_med_shell/eval_medreason.sh