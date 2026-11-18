#!/usr/bin/env bash
#$ -cwd
#$ -j y
#$ -l h_rt=00:30:00
set -euo pipefail
: "${MILESTONE_ROOT:?}" "${MILESTONE_ACTION:?}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 TZ=Asia/Tokyo
cd /gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro
exec .venv/bin/python "$MILESTONE_ROOT/controller.py" "$MILESTONE_ACTION" --case "${MILESTONE_CASE:-}"
