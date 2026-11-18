#!/usr/bin/env bash
#$ -cwd
#$ -j y
#$ -l h_rt=23:50:00
set -euo pipefail
: "${VARIANTS_ROOT:?}" "${VARIANT_ARM:?}" "${VARIANT_ACTION:?}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 TZ=Asia/Tokyo
cd /gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro
exec .venv/bin/python "$VARIANTS_ROOT/controller.py" "$VARIANT_ACTION" --arm "$VARIANT_ARM" --epoch "${EVALUATION_EPOCH:-0}"
