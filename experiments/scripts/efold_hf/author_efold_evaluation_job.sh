#!/usr/bin/env bash
#$ -cwd
#$ -j y
set -euo pipefail
: "${AUTHOR_EFOLD_SHARD:?}"
project=/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro
cd "$project"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "$project/.venv/bin/python" "$project/scripts/efold_hf/author_efold_evaluation.py" predict --shard "$AUTHOR_EFOLD_SHARD"
