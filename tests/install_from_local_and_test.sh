#!/usr/bin/env bash
set -e

./tests/install_from_local.sh
export KERAS_BACKEND="$1"
./tests/run_code_checks.sh