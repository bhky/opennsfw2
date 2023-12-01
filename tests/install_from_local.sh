#!/usr/bin/env bash
set -e

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install opencv-stubs matplotlib-stubs  # For mypy.
python3 -m pip install .