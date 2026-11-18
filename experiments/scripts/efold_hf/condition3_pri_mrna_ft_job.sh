#!/usr/bin/env bash
#$ -cwd
#$ -j y
#$ -l h_rt=23:50:00
set -euo pipefail
: "${CURRICULUM_ROOT:?}" "${FT_ACTION:?}" "${FT_ARM:?}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 TZ=Asia/Tokyo
cd /gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro
exec .venv/bin/python "$CURRICULUM_ROOT/controller.py" "$FT_ACTION" \
  --arm "$FT_ARM" --epoch "${FT_EPOCH:-0}" --waited-job "${WAITED_STAGE1_JOB:-}"
