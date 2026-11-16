#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${1:-$root/build/linux-release}"
out_dir="$root/demo_output"

"$build_dir/asset_generator" --preset normal \
    --out "$out_dir/normal.ska"
"$build_dir/skelpreview" offline "$out_dir/normal.ska" \
    --start-frame 0 --end-frame 5 \
    --width 640 --height 480 \
    --output-dir "$out_dir/frames" \
    --report "$out_dir/frames/report.json"

echo
echo "Open the interactive preview with:"
echo "  $build_dir/skelpreview preview $out_dir/normal.ska"

