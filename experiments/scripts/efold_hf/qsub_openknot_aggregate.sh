#!/usr/bin/env bash
#$ -cwd
#$ -j y
set -euo pipefail
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TZ=Asia/Tokyo
.venv/bin/python scripts/efold_hf/openknot_analysis.py
exec .venv/bin/python scripts/efold_hf/openknot_report.py
