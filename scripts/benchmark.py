#!/usr/bin/env python3
"""Generate the 8x5-minute sample and measure wall time plus peak RSS."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

try:
    import resource
except ImportError:  # Windows
    resource = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_samples import make_performance


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = make_performance(root)
    command = [
        sys.executable,
        "-m",
        "cutmaster",
        "--manifest",
        str(root / "manifest.json"),
        "--state-dir",
        str(root / "state"),
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    elapsed = time.perf_counter() - started
    usage_kib = (
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        if resource is not None
        else None
    )
    result = {
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "child_peak_rss_mib": usage_kib / 1024.0 if usage_kib is not None else None,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "manifest": manifest,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
