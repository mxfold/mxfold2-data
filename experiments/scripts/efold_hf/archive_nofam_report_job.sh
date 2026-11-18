#!/usr/bin/env bash
#$ -cwd
#$ -j y
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 TZ=Asia/Tokyo
exec .venv/bin/python scripts/efold_hf/archive_nofam_report.py "$NOFAM_REPORT_ACTION"
