#!/usr/bin/env bash
set -euo pipefail
root=${1:?run directory}
action=${2:?action}
python=${3:?python executable}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 TZ=Asia/Tokyo
export PYTHONPATH="$root/source"
cd "$root/source"
exec "$python" -m scripts.default_model.run "$action" "$root"
