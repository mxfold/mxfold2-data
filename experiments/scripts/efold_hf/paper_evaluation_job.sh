#!/usr/bin/env bash
#$ -cwd
#$ -j y
set -euo pipefail
: "${PAPER_EVAL_ARM:?}"
project=/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro
cd "$project"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "$project/.venv/bin/python" "$project/scripts/efold_hf/paper_evaluation.py" predict --arm "$PAPER_EVAL_ARM"
