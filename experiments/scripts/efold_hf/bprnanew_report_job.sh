#!/usr/bin/env bash
#$ -cwd
#$ -j y
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 TZ=Asia/Tokyo BENCH_REQUIRE_COMPLETE=1
exec .venv/bin/python scripts/efold_hf/bprnanew_report.py
