"""Detect active and latent dependency-confusion exposure in manifests."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from .client import ApiError, IndexClient, IndexUnavailable
from .manifest import detect_external_policy, discover_manifests, parse_manifest
from .resolver import Resolver
from .versions import Version


def _public_baseline(resolver: Resolver, package: str) -> str | None:
    metadata = resolver.package("public", package)
    if not metadata:
        return None
    normal_versions = [
        record["version"]
        for record in metadata.get("versions", [])
        if not record.get("injected")
    ]
    if not normal_versions:
        return None
    return max(normal_versions, key=Version)


def detect(
    directory: str | Path,
    output: str | Path,
    client: IndexClient,
    *,
    scan_delay: float = 0.0,
) -> dict[str, Any]:
    directory_path = Path(directory)
    output_path = Path(output)
    if not directory_path.is_dir():
        raise FileNotFoundError(str(directory_path))

    initial_status = client.status()
    initial_snapshot = str(initial_status.get("snapshot_sha256", ""))
    if scan_delay > 0:
        time.sleep(scan_delay)

    manifests = discover_manifests(directory_path)
    findings: list[dict[str, Any]] = []
    warnings: list[str] = []
    requirements_count = 0
    resolved_count = 0
    unresolved_count = 0
    package_names: set[str] = set()
    risk_names: set[str] = set()
    latent_names: set[str] = set()

    for manifest in manifests:
        parsed = parse_manifest(manifest)
        warnings.extend(f"{manifest}: {warning}" for warning in parsed.warnings)
        source_order = detect_external_policy(manifest) or parsed.source_order
        resolver = Resolver(client, source_order)
        for result in resolver.resolve_all(parsed.requirements):
            requirements_count += 1
            package_names.add(result["package"])
            if result["resolved"]:
                resolved_count += 1
            else:
                unresolved_count += 1

            finding: dict[str, Any] = {
                "manifest": str(manifest),
                "line_number": result["line_number"],
                "package": result["package"],
                "requirement": result["requirement"],
                "resolved": result["resolved"],
                "source": result["source"],
                "version": result["version"],
                "private_highest": result["private_highest"],
                "public_highest": result["public_highest"],
                "public_injected_versions": result["public_injected_versions"],
            }

            if (
                result["resolved"]
                and result["source"] == "public"
                and result["private_present"]
            ):
                finding["kind"] = "risk"
                risk_names.add(result["package"])
                findings.append(finding)
                continue

            if (
                result["resolved"]
                and result["source"] != "public"
                and result["private_present"]
                and result["public_injected_versions"]
            ):
                baseline = _public_baseline(resolver, result["package"])
                private_highest = result["private_highest"]
                if baseline is None or (
                    private_highest is not None and Version(private_highest) > Version(baseline)
                ):
                    finding["kind"] = "latent"
                    finding["private_above_public_baseline"] = baseline
                    latent_names.add(result["package"])
                    findings.append(finding)

    final_status = client.status()
    final_snapshot = str(final_status.get("snapshot_sha256", ""))
    stale = initial_snapshot != final_snapshot

    report = {
        "directory": str(directory_path.resolve()),
        "manifests": [str(path) for path in manifests],
        "snapshot_sha256": initial_snapshot,
        "final_snapshot_sha256": final_snapshot,
        "stale": stale,
        "risks": sorted(risk_names),
        "latent": sorted(latent_names),
        "findings": findings,
        "warnings": warnings,
        "summary": {
            "packages": len(package_names),
            "requirements": requirements_count,
            "resolved": resolved_count,
            "unresolved": unresolved_count,
            "risks": len(risk_names),
            "risk_count": len(risk_names),
            "latent": len(latent_names),
            "latent_count": len(latent_names),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="扫描清单并检测依赖混淆风险")
    parser.add_argument("--dir", required=True, help="待扫描清单目录")
    parser.add_argument("--output", required=True, help="report.json 输出路径")
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    parser.add_argument(
        "--scan-delay",
        type=float,
        default=float(os.environ.get("DEPCONFUSION_SCAN_DELAY", "0")),
        help="测试用：取初始快照后等待的秒数",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    directory = Path(args.dir)
    if not directory.is_dir():
        print(f"目录不存在: {directory}", file=sys.stderr)
        return 1
    client = IndexClient(args.base_url)
    try:
        report = detect(directory, args.output, client, scan_delay=args.scan_delay)
    except IndexUnavailable as exc:
        print(f"索引服务不可达: {exc}", file=sys.stderr)
        return 2
    except (ApiError, OSError, ValueError) as exc:
        print(f"检测失败: {exc}", file=sys.stderr)
        return 1

    for warning in report.get("warnings", []):
        print(f"警告: {warning}", file=sys.stderr)
    print(
        f"检测完成: {args.output} risks={len(report['risks'])} "
        f"latent={len(report['latent'])} stale={report['stale']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
