"""Command line entry point."""

from __future__ import annotations

import argparse
import sys

from .errors import DubReplaceError
from .runner import run_job


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dub-replace",
        description="按方言配音批量替换过场动画目标语言配音轨（不变音高 WSOLA 时间伸缩，多声道 PCM 输出）。",
    )
    parser.add_argument("manifest", help="作业清单 JSON 路径")
    parser.add_argument("-q", "--quiet", action="store_true", help="仅输出错误信息")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        exit_code, summary = run_job(args.manifest)
    except DubReplaceError as exc:
        print(f"[FATAL] {exc.code}: {exc.message}", file=sys.stderr)
        print(f"提示: {exc.hint}", file=sys.stderr)
        return 2
    if not args.quiet:
        print(
            f"作业完成: 共 {summary['total']} 个动画，成功 {summary['succeeded']}，"
            f"失败 {summary['failed']}，跳过(已成功) {summary['skipped']}"
        )
        for result in summary["results"]:
            tag = {"success": "OK ", "failed": "ERR", "skipped": "SKIP"}[result["status"]]
            line = f"  [{tag}] {result['animation_id']} -> {result['report_path']}"
            if result["error"]:
                line += f" ({result['error']['code']})"
            print(line)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
