#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
METHOD=dynamic_random RUN_NAME=llama-3-1-8b-seq-dynamic-random-qkvogateupdown-rank32-commonsense170k-epoch1 bash task.sh

