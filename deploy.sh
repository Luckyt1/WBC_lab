#!/usr/bin/env bash
set -euo pipefail
WBC_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${ELF3_BOOTSTRAP_PYTHON:-python3}" "$WBC_ROOT/scripts/manage.py" "$@"
