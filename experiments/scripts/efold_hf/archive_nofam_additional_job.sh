#!/usr/bin/env bash
#$ -cwd
#$ -j y
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 TZ=Asia/Tokyo
cd /gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro
exec .venv/bin/python "$NOFAM_EXTRA_ROOT/controller.py" "$NOFAM_EXTRA_ACTION" --model "${NOFAM_EXTRA_MODEL:-RNAformer_bprna}"
