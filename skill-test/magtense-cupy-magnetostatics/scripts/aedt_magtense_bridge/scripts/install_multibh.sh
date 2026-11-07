#!/usr/bin/env bash
# Linux / WSL2: requires gfortran, make, numpy, F2PY, meson and Ninja.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
repo="${1:-./MagTense}"
if [[ ! -f "${repo}/Makefile" ]]; then
    git clone https://github.com/cmt-dtu-energy/MagTense.git "$repo"
fi
python patch_magtense.py --repo "$repo" --build --fc gfortran
python -m pip uninstall -y magtense
python -m pip install -e "$repo/python"
python native_smoke.py
echo 'Multi-BH MagTense installed and smoke-tested.'
