#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

bash exps/commonsense170k/finetune_llama_seq_alpha_qkvogateupdown_rank32.sh
bash exps/commonsense170k/finetune_llama_seq_static_random_qkvogateupdown_rank32.sh
bash exps/commonsense170k/finetune_llama_seq_dynamic_random_qkvogateupdown_rank32.sh
bash exps/commonsense170k/finetune_llama_lora_2epoch_qkvogateupdown_rank32.sh

