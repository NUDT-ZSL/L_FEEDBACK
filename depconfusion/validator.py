"""Re-run resolution under a selected mitigation and verify artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .client import ApiError, IndexClient, IndexUnavailable
from .manifest import parse_manifest
from .resolver import Resolver, SOURCE_ORDERS
from .versions import Version


STRATEGIES = ("pin", "hash", "private-first")


class ValidationFailure(RuntimeError):
    pass


def _validate_results(
    strategy: str,
    parsed: Any,
    resolver: Resolver,
) -> tuple[int, list[str]]:
    results = resolver.resolve_all(parsed.requirements)
    errors: list[str] = []
    if not results:
        return 0, ["清单中没有可解析的依赖"]

    for result in results:
        label = result["requirement"]
        if not result["resolved"]:
            reason = result.get("reason", "没有满足约束的版本")
            errors.append(f"{label}: {reason}")
            continue

        if strategy == "pin":
            if result["operator"] != "==":
                errors.append(f"{label}: pin 验证要求精确版本（==）")
                continue
            if result["private_present"] and result["source"] != "private":
                errors.append(f"{label}: pin 未解析到私有制品")
            elif result["private_present"] and (
                result["private_highest"] is None
                or Version(str(result["version"])) != Version(str(result["private_highest"]))
            ):
                errors.append(f"{label}: pin 版本不是私有最高满足版本")

        elif strategy == "hash":
            if not result["hashes"]:
                errors.append(f"{label}: hash 验证要求 --hash=sha256:<摘要>")
                continue
            if not result["hash_verified"]:
                errors.append(f"{label}: 制品 sha256 与清单哈希不匹配")

        elif strategy == "private-first":
            if result["private_present"] and result["source"] != "private":
                errors.append(f"{label}: private-first 未优先解析私有制品")

    return len(results), errors


def validate_manifest(
    manifest_path: str | Path,
    strategy: str,
    client: IndexClient,
) -> dict[str, Any]:
    path = Path(manifest_path)
    parsed = parse_manifest(path)
    if strategy == "private-first":
        order = SOURCE_ORDERS["private-first"]
    else:
        # Hash filtering makes a malicious public candidate ineligible before
        # source preference is applied.
        order = SOURCE_ORDERS["public-first"]

    resolver = Resolver(client, order)
    count, errors = _validate_results(strategy, parsed, resolver)
    return {
        "strategy": strategy,
        "manifest": str(path),
        "resolved": count,
        "ok": not errors,
        "errors": errors,
        "warnings": parsed.warnings,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="验证依赖清单修复策略及制品哈希")
    parser.add_argument("strategy_pos", nargs="?", choices=STRATEGIES)
    parser.add_argument("-m", "--manifest", required=True)
    parser.add_argument(
        "--strategy",
        "--policy",
        dest="strategy_opt",
        choices=STRATEGIES,
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    strategy = args.strategy_opt or args.strategy_pos
    if not strategy:
        print("错误: 必须指定验证策略", file=sys.stderr)
        return 1
    manifest = Path(args.manifest)
    if not manifest.is_file():
        print(f"清单不存在: {manifest}", file=sys.stderr)
        return 1

    client = IndexClient(args.base_url)
    try:
        client.status()
        result = validate_manifest(manifest, strategy, client)
    except FileNotFoundError:
        print(f"清单不存在: {manifest}", file=sys.stderr)
        return 1
    except IndexUnavailable as exc:
        print(f"索引服务不可达: {exc}", file=sys.stderr)
        return 2
    except (ApiError, OSError, ValueError) as exc:
        print(f"验证失败: {exc}", file=sys.stderr)
        return 1

    for warning in result["warnings"]:
        print(f"警告: {warning}", file=sys.stderr)
    if result["ok"]:
        print(f"PASS: {strategy} (resolved={result['resolved']})")
        return 0
    for error in result["errors"]:
        print(f"FAIL: {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
