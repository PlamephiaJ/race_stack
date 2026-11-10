#!/usr/bin/env bash
# Build the CUDA native dependency with the same Python used by ROS 2 Humble.
# An existing RangeLibc checkout may be supplied as the first argument.
set -euo pipefail

pf_python="${ROS_PYTHON:-/usr/bin/python3}"
pf_source="${1:-}"
pf_temp="$(mktemp -d)"
cleanup() {
    if [[ -n "$pf_temp" ]]; then rm -rf "$pf_temp"; fi
}
trap cleanup EXIT

if [[ -z "$pf_source" ]]; then
    pf_source="$pf_temp/range_libc"
    git clone https://github.com/kctess5/range_libc.git "$pf_source"
    git -C "$pf_source" checkout f55480a7044367c219745480b436b8b7117f8282
fi

# Always build from a clean copy: CPU object files must never be reused by a
# CUDA build. Do not modify the caller's checkout or cached build artifacts.
"$pf_python" - "$pf_source" "$pf_temp/build_source" <<'PY'
from pathlib import Path
import os
import shutil
import sys

source, target = map(Path, sys.argv[1:])
shutil.copytree(source, target, ignore=shutil.ignore_patterns(
    '.git', 'build', '*.so', '__pycache__', '*.egg-info'))
setup = target / 'pywrapper/setup.py'
text = setup.read_text()
# These upstream flags contain unexpanded shell expressions, not compiler settings.
text = '\n'.join(line for line in text.splitlines() if '-DCMAKE_C' not in line)
arch = os.environ.get('CUDA_ARCH', '87')
if not arch.isdigit():
    raise SystemExit('CUDA_ARCH must be digits, e.g. 87 for Jetson Orin')
text = text.replace('-arch=sm_87', '-arch=sm_' + arch)
setup.write_text(text + '\n')
PY

cd "$pf_temp/build_source/pywrapper"
export WITH_CUDA=ON
"$pf_python" -m pip install --force-reinstall --no-deps --no-build-isolation .
"$pf_python" -c 'import range_libc; assert range_libc.SHOULD_USE_CUDA, "CPU RangeLibc is not allowed"'
