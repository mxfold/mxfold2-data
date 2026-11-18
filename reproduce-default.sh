#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD/source"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1

case "${1:-train}" in
    prepare|-h|--help)
        exec "${MXFOLD2_PYTHON:-python3}" reproduce_default.py "$@"
        ;;
    train)
        if [[ "${2:-}" == --help || "${2:-}" == -h ]]; then
            exec "${MXFOLD2_PYTHON:-python3}" reproduce_default.py "$@"
        fi
        ;;
    *)
        echo 'Usage: ./reproduce-default.sh [train|prepare] [--work-dir DIR] [--hours HOURS]' >&2
        exit 2
        ;;
esac

if [[ -n "${MXFOLD2_PYTHON:-}" ]]; then
    python="$MXFOLD2_PYTHON"
else
    command -v uv >/dev/null || { echo 'Install uv or set MXFOLD2_PYTHON to a prepared Python environment.' >&2; exit 1; }
    if [[ ! -x .venv-default/bin/python ]]; then
        uv venv --python 3.13.8 .venv-default
    fi
    python="$PWD/.venv-default/bin/python"
    uv pip install --python "$python" --index-url https://download.pytorch.org/whl/cu126 'torch==2.7.1+cu126'
    uv pip install --python "$python" -r default-model/requirements.txt
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
"$python" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else "A CUDA GPU is required")'

cmake -S source/mxfold2/src -B .build/default-model -Dpybind11_DIR="$("$python" -m pybind11 --cmakedir)" \
    -DPython_EXECUTABLE="$("$python" -c 'import sys; print(sys.executable)')" -DCMAKE_BUILD_TYPE=Release
cmake --build .build/default-model --parallel "${BUILD_JOBS:-4}"
cmake --install .build/default-model
exec "$python" reproduce_default.py "$@"
