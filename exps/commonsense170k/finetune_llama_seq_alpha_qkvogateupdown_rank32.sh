#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
METHOD=alpha RUN_NAME=llama-3-1-8b-seq-alpha-qkvogateupdown-rank32-commonsense170k-epoch1 bash task.sh

