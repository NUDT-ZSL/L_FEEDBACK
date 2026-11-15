"""Command line entry point.

Usage::

    python -m audiopipeline.cli manifest.json output.wav [-c config.json]
                                               [-r report.json]

Exit code is non-zero for fatal manifest / configuration errors; per-event
failures are recorded in the report while the process still exits 0.
"""

from __future__ import annotations

import argparse
import sys

from .config import Config
from .errors import ERROR_CATALOG, PipelineError
from .pipeline import Pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audiopipeline",
        description="互动剧情音频离线对齐与响度统一处理器",
    )
    parser.add_argument("manifest", help="事件清单 JSON 路径")
    parser.add_argument("output", help="输出 48kHz/24bit 立体声 WAV 路径")
    parser.add_argument(
        "-c", "--config", default=None, help="可选配置文件 JSON 路径"
    )
    parser.add_argument(
        "-r",
        "--report",
        default=None,
        help="报告输出路径(默认与输出音频同目录，后缀 .report.json)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report_path = args.report or _default_report_path(args.output)
    try:
        config = Config.from_file(args.config) if args.config else Config()
        pipeline = Pipeline(config)
        report = pipeline.run(args.manifest, args.output, report_path)
    except PipelineError as exc:
        sys.stderr.write(f"致命错误 {exc.code}: {exc.reason or ERROR_CATALOG[exc.code].message}\n")
        return 2

    summary = report["summary"]
    sys.stdout.write(
        "处理完成: 总事件 {total} / 成功 {succeeded} / 跳过 {skipped}; "
        "整体响度 {lufs} LUFS, 真峰值 {tp} dBTP\n".format(
            total=summary["total_events"],
            succeeded=summary["succeeded"],
            skipped=summary["skipped"],
            lufs=summary["overall_loudness_lufs"],
            tp=summary["overall_true_peak_dbtp"],
        )
    )
    sys.stdout.write(f"音频: {args.output}\n报告: {report_path}\n")
    return 0


def _default_report_path(output_path: str) -> str:
    if output_path.lower().endswith(".wav"):
        return output_path[:-4] + ".report.json"
    return output_path + ".report.json"


if __name__ == "__main__":
    raise SystemExit(main())
