#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mode="${1:-train}"

if [[ "$mode" == "-h" || "$mode" == "--help" ]]; then
    printf '%s\n' 'Usage: ./reproduce.sh [train|verify|prepare] [--suite SUITE] [--work-dir DIRECTORY] [--eternafold-ranks N]'
    exit 0
fi
if [[ "$mode" != train && "$mode" != verify && "$mode" != prepare ]]; then
    echo "Unknown mode: $mode" >&2
    exit 2
fi

suite=all
args=("$@")
for ((i=1; i<${#args[@]}; i++)); do
    case "${args[i]}" in
        --suite) suite="${args[i+1]:-}" ;;
        --suite=*) suite="${args[i]#--suite=}" ;;
    esac
done

if [[ -n "${MXFOLD2_PYTHON:-}" ]]; then
    python="$MXFOLD2_PYTHON"
else
    command -v uv >/dev/null || { echo 'Install uv or set MXFOLD2_PYTHON to a prepared Python 3.13 environment.' >&2; exit 1; }
    if [[ ! -x .venv/bin/python ]]; then
        uv venv --python 3.13 .venv
    fi
    python="$PWD/.venv/bin/python"
    if [[ "$mode" == train ]]; then
        torch_index=https://download.pytorch.org/whl/cu126
        if [[ "$suite" == eternafold ]]; then
            torch_index=https://download.pytorch.org/whl/cpu
        fi
        uv pip install --python "$python" --index-url "$torch_index" 'torch==2.7.1'
        uv pip install --python "$python" -r requirements.txt
    else
        uv pip install --python "$python" 'numpy==2.3.1' 'scipy==1.16.0' 'pandas==2.3.0' 'matplotlib==3.10.3'
    fi
fi

if [[ "$mode" == train ]]; then
    if [[ "$suite" != eternafold ]]; then
        "$python" -c 'import torch; assert torch.cuda.is_available(), "A CUDA GPU is required"'
    fi
    cmake -S source/mxfold2/src -B .build -Dpybind11_DIR="$("$python" -m pybind11 --cmakedir)" \
        -DPython_EXECUTABLE="$("$python" -c 'import sys; print(sys.executable)')" -DCMAKE_BUILD_TYPE=Release
    cmake --build .build --parallel "${BUILD_JOBS:-4}"
    cmake --install .build
fi

export PYTHONPATH="$PWD/source${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
exec "$python" reproduce.py "$@"
