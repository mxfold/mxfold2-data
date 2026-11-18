#!/usr/bin/env bash
#$ -cwd
#$ -j y
set -euo pipefail
project=/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro
cd "$project"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$project/.venv/bin/python" "$project/scripts/efold_hf/author_efold_evaluation.py" summarize
exec "$project/.venv/bin/python" "$project/scripts/efold_hf/assemble_paper_comparison.py"
