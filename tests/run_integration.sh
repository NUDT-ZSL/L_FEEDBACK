#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_exe="${1:-${repo_root}/build/cpu/character_renderer}"
test_root="${2:-${repo_root}/build/cpu/integration_data}"

cmake \
  "-DTEST_EXE=${test_exe}" \
  "-DTEST_ROOT=${test_root}" \
  -P "${repo_root}/tests/run_integration.cmake"

