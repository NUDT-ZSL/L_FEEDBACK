#!/usr/bin/env bash
# Deterministic end-to-end demo: generate -> mix -> verify -> unit tests.
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
WORKDIR="demo_project"

python -c "import shutil; shutil.rmtree('${WORKDIR}', ignore_errors=True)"
python -m offline_mixer generate "${WORKDIR}" --seed 20261117
python -m offline_mixer mix "${WORKDIR}/timeline.json" \
    -o "${WORKDIR}/mix_output.wav" \
    -r "${WORKDIR}/mix_report.json"
python -m offline_mixer verify "${WORKDIR}/mix_report.json"
python -m unittest tests.test_pipeline -v
