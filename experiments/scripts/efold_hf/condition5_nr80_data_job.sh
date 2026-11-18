#!/usr/bin/env bash
#$ -cwd
#$ -j y
#$ -l h_rt=23:50:00
set -euo pipefail
: "${CONDITION5_DATA_ROOT:?}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 TZ=Asia/Tokyo
cd /gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro
exec .venv/bin/python "$CONDITION5_DATA_ROOT/controller.py" --output "$CONDITION5_DATA_ROOT"
