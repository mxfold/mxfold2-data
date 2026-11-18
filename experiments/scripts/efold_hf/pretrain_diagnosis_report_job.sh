#!/usr/bin/env bash
#$ -cwd
#$ -j y
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 TZ=Asia/Tokyo
export MPLCONFIGDIR="$PWD/qsub_logs/efold_hf/pretrain_diagnosis_20261031/mpl-cache"
exec .venv/bin/python scripts/efold_hf/pretrain_diagnosis_report.py
