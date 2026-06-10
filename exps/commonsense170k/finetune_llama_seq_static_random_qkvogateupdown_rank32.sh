#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
METHOD=static_random RUN_NAME=llama-3-1-8b-seq-static-random-qkvogateupdown-rank32-commonsense170k-epoch1 bash task.sh

