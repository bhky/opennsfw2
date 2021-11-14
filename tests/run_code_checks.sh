#!/usr/bin/env bash
set -e

find opennsfw2 -iname "*.py" | grep -v -e "__init__.py" | xargs -L 1 pylint --errors-only
find opennsfw2 -iname "*.py" | grep -v -e "__init__.py" | xargs -L 1 pylint --exit-zero
find opennsfw2 -iname "*.py" | grep -v -e "__init__.py" | xargs -L 1 mypy --strict --check-untyped-defs --implicit-reexport
find tests -iname "*.py" | grep -v -e "__init__.py" | xargs -L 1 python3 -m unittest