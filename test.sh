#!/usr/bin/env bash
# this_file: test.sh
set -euo pipefail
cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")"
test_python=$(uvx hatch run hatch-test.py3.12:python -c 'import sys; print(sys.executable)')
if [[ -f ../uubed-rs/Cargo.toml ]]; then
    wheels=$(mktemp -d)
    trap 'rm -rf "$wheels"' EXIT
    (cd ../uubed-rs && uvx maturin build --release --out "$wheels" --interpreter python3.12)
    uv pip install --python "$test_python" --reinstall-package uubed-rs "$wheels"/*.whl
else
    uv pip install --python "$test_python" 'uubed-rs>=1.0.12'
fi
uvx hatch test -py 3.12
