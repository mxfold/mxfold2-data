#!/usr/bin/env bash
#$ -cwd
#$ -j y
set -euo pipefail
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TZ=Asia/Tokyo
export MW_MAP_REQUIRE_COMPLETE=1
exec .venv/bin/python scripts/chemical_consistency/map_mw_report.py
