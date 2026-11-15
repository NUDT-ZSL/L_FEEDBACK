#!/usr/bin/env bash
# Deterministic end-to-end demo: four render configurations plus error cases.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

$PY -m volfog --mode single   --output-dir output/demo --output-prefix sser
$PY -m volfog --mode single   --output-dir output/demo --output-prefix spar --parallel --threads 4
$PY -m volfog --mode multiple --output-dir output/demo --output-prefix mser
$PY -m volfog --mode multiple --output-dir output/demo --output-prefix mpar --parallel --threads 4
$PY scripts/acceptance.py
