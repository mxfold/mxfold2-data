#!/usr/bin/env bash
#$ -cwd
#$ -j y
set -euo pipefail
: "${EF_PROJECT_ROOT:?}" "${EF_FULL_ROOT:?}" "${EF_ACTION:?}"
cd "$EF_PROJECT_ROOT"
export TZ=Asia/Tokyo PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH="/apps/t4/rhel9/free/openmpi/5.0.7-gcc/bin:$PATH"
export LD_LIBRARY_PATH="/apps/t4/rhel9/free/openmpi/5.0.7-gcc/lib:/apps/t4/rhel9/free/ucx/1.18.0-gcc/lib:${LD_LIBRARY_PATH:-}"
exec "$EF_PROJECT_ROOT/.venv/bin/python" "$EF_FULL_ROOT/source/scripts/eternafold/fulltest_new_metrics.py" \
  "$EF_ACTION" --root "$EF_FULL_ROOT" --ranks "${EF_RANKS:-128}" --workers "${EF_WORKERS:-32}"
