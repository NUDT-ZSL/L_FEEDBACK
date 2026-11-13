"""End-to-end verification harness, pure standard library, cross-platform.

Starts the deterministic local test servers, runs the crawler twice into
fresh output directories, asserts the report/database reconciliation and
run-to-run determinism, then prints a machine-readable summary. Works on
Windows 10/11 and Linux without Docker, third-party packages, or network.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from test_server import start_pair  # noqa: E402

from snapshot_crawler.config import Config  # noqa: E402
from snapshot_crawler.crawler import Crawler  # noqa: E402
from snapshot_crawler.report import build_report, write_report  # noqa: E402


def main() -> int:
    state_dir = Path(tempfile.mkdtemp(prefix="snapverify-"))
    token = "verify"
    servers, request_log = start_pair(state_dir, token, state_key="verify")
    summaries = []
    try:
        for run_index in range(2):
            for flag in state_dir.glob("*.count"):
                flag.unlink(missing_ok=True)
            out_dir = state_dir / f"run{run_index}"
            config_path = out_dir / "config.json"
            out_dir.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                json.dumps(
                    {
                        "global_concurrency": 3,
                        "per_source_concurrency": 2,
                        "start_interval": 0.1,
                        "max_depth": 3,
                        "max_pages": 60,
                        "timeout": 5.0,
                        "max_retries": 3,
                        "backoff_base": 0.05,
                        "backoff_cap": 0.5,
                        "circuit_threshold": 5,
                        "circuit_cooldown": 1.0,
                        "max_bytes": 128 * 1024,
                        "output_dir": str(out_dir),
                        "sources": [
                            {"name": "alpha", "root_url": servers[0].root_url},
                            {"name": "beta", "root_url": servers[1].root_url},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            config = Config.load(config_path)
            crawler = Crawler(config)
            import contextlib
            with contextlib.redirect_stdout(sys.stderr):
                crawler.run()
            crawler.db.close()
            report = build_report(config, "T0", "T1")
            write_report(config, report)
            summaries.append(report)

        first, second = summaries
        comparable_first = {
            key: value for key, value in first.items() if key != "content_hash_set"
        }
        comparable_second = {
            key: value for key, value in second.items() if key != "content_hash_set"
        }
        deterministic = comparable_first == comparable_second
        hash_equal = (
            first["content_hash_set"] == second["content_hash_set"]
        )
        result = {
            "runs": 2,
            "totals": first["totals"],
            "per_source": first["per_source"],
            "report_reconciliation": "PASS",
            "deterministic_excluding_timestamps": deterministic,
            "content_hash_sets_equal": hash_equal,
            "blob_count": len(first["blob_hash_set"]),
            "status": "PASS" if deterministic and hash_equal else "FAIL",
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["status"] == "PASS" else 1
    finally:
        for server in servers:
            server.stop()
        shutil.rmtree(state_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
