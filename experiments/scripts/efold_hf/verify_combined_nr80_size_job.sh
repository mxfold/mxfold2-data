#!/usr/bin/env bash
#$ -cwd
#$ -j y
#$ -l h_rt=02:00:00
set -euo pipefail
: "${NR80_SIZE_OUTPUT:?}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 TZ=Asia/Tokyo
cd /gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro
exec .venv/bin/python scripts/efold_hf/verify_combined_nr80_size.py --output "$NR80_SIZE_OUTPUT"
