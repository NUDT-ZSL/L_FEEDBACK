"""Command line interface for the batch loudness normalizer."""

from __future__ import annotations

import argparse
import json
import math
import sys

from .pipeline import FileResult, Params, probe_file, process_file
from .report import build_report, render
from .scheduler import run_ordered


def _timeout_result(item: FileResult) -> FileResult:
    item.status = "timeout"
    item.error = {"stage": "process", "reason": "per-file timeout exceeded"}
    return item


def _probe_timeout(item: tuple) -> FileResult:
    index, entry, _params = item
    return FileResult(
        index=index,
        path=entry["path"],
        offset_ms=float(entry.get("offset_ms", 0.0)),
        status="timeout",
        error={"stage": "probe", "reason": "per-file timeout exceeded"},
    )


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="loudnorm-batch",
        description="Batch loudness normalization (ITU-R BS.1770-4 / EBU R128 style).",
    )
    parser.add_argument("--manifest", required=True, help="JSON manifest of input files")
    parser.add_argument("--out-dir", default="out", help="output directory for WAV files")
    parser.add_argument("--report", default="report.json", help="JSON report path")
    parser.add_argument("--target-lufs", type=float, default=-23.0)
    parser.add_argument("--true-peak-ceiling", type=float, default=-1.0, help="dBTP")
    parser.add_argument("--gain-min", type=float, default=-24.0, help="dB")
    parser.add_argument("--gain-max", type=float, default=24.0, help="dB")
    parser.add_argument(
        "--total-duration-ms",
        type=float,
        default=None,
        help="unified timeline length (default: max offset+duration over inputs)",
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=30.0, help="per-file seconds")
    args = parser.parse_args(argv)

    with open(args.manifest, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    entries = manifest["files"]

    params = Params(
        target_lufs=args.target_lufs,
        ceiling_dbtp=args.true_peak_ceiling,
        gain_min_db=args.gain_min,
        gain_max_db=args.gain_max,
        out_dir=args.out_dir,
        timeout_s=args.timeout,
    )

    # Phase 1: probe everything (bounded concurrency, per-file timeout).
    probe_items = [(i, e, params) for i, e in enumerate(entries)]
    results: list[FileResult] = run_ordered(
        probe_items,
        lambda item: probe_file(item[0], item[1], item[2]),
        _probe_timeout,
        jobs=args.jobs,
        timeout_s=params.timeout_s + 5.0,
    )

    # Unified timeline: longest offset+duration unless overridden.
    if args.total_duration_ms is not None:
        params.total_ms = args.total_duration_ms
    else:
        total = 0.0
        for r in results:
            if r.metadata and r.metadata.get("duration_s") is not None:
                total = max(total, r.offset_ms + math.ceil(r.metadata["duration_s"] * 1000.0))
        params.total_ms = total

    # Phase 2: full processing.
    results = run_ordered(
        results,
        lambda r: process_file(r, params),
        _timeout_result,
        jobs=args.jobs,
        timeout_s=params.timeout_s,
    )

    report = build_report(results, params, args.jobs)
    with open(args.report, "w", encoding="utf-8") as fh:
        fh.write(render(report))

    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"processed {len(results)} files: {summary}")
    print(f"report written to {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(run())
