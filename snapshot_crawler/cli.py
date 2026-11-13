"""Command line entry point for the snapshot crawler."""
from __future__ import annotations

import argparse
import sys

from .config import Config
from .crawler import Crawler
from .db import utc_now_iso
from .report import build_report, write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="snapshot-crawler",
        description="Offline batch archiver producing a SQLite snapshot store.",
    )
    parser.add_argument("--config", required=True, help="JSON config path")
    parser.add_argument("--output-dir", help="override output directory")
    parser.add_argument("--timeout", type=float, help="per-request timeout (s)")
    parser.add_argument("--max-retries", type=int, help="retry cap per URL")
    parser.add_argument(
        "--circuit-threshold", type=int, help="consecutive failures to trip"
    )
    parser.add_argument(
        "--circuit-cooldown", type=float, help="circuit cooldown seconds"
    )
    parser.add_argument("--max-bytes", type=int, help="max response body bytes")
    parser.add_argument(
        "--exit-after",
        type=int,
        default=None,
        help="hard-exit via os._exit after the Nth request (recovery test)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides = {
        "output_dir": args.output_dir,
        "timeout": args.timeout,
        "max_retries": args.max_retries,
        "circuit_threshold": args.circuit_threshold,
        "circuit_cooldown": args.circuit_cooldown,
        "max_bytes": args.max_bytes,
    }
    config = Config.load(args.config, overrides)
    started_at = utc_now_iso()
    crawler = Crawler(config, exit_after=args.exit_after)
    crawler.run()
    ended_at = utc_now_iso()
    crawler.db.close()
    report = build_report(config, started_at, ended_at)
    path = write_report(config, report)
    print(f"[done] report written to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
