#!/usr/bin/env bash
set -euo pipefail
GL165_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$GL165_PROJECT_ROOT"
GL165_PYTHON="${GL165_PYTHON:-$HOME/miniconda3/envs/isaac_wbc/bin/python}"
if [[ ! -x "$GL165_PYTHON" ]]; then
    echo 'Set GL165_PYTHON to the Python executable in an Isaac Lab 2.2 / Isaac Sim 5.0 environment.' >&2
    exit 1
fi
# This host's conda ICU needs the conda C++ ABI, before Kit loads system libraries.
GL165_CXX_LIB="${GL165_PYTHON%/bin/python}/lib/libstdc++.so.6"
if [[ -f "$GL165_CXX_LIB" ]]; then
    export LD_PRELOAD="$GL165_CXX_LIB${LD_PRELOAD:+:$LD_PRELOAD}"
fi
export PYTHONUNBUFFERED=1
"$GL165_PYTHON" tools/check_gl165_resources.py
GL165_START_ARGS=(--from_scratch)
for GL165_ARG in "$@"; do
    if [[ "$GL165_ARG" == --resume || "$GL165_ARG" == --resume=* ]]; then
        GL165_START_ARGS=()
    fi
done
exec "$GL165_PYTHON" legged_lab/scripts/train.py --task walk_gl165_wbc --headless \
    "${GL165_START_ARGS[@]}" "$@"
