"""Apply pin, hash, and private-first mitigations to a manifest."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from .client import ApiError, IndexClient, IndexUnavailable
from .manifest import (
    Requirement,
    atomic_write_text,
    parse_manifest,
    policy_file_for,
    write_manifest,
)
from .versions import Version, satisfies


STRATEGIES = ("pin", "hash", "private-first")


class RepairError(RuntimeError):
    pass


def _private_candidates(client: IndexClient, requirement: Requirement) -> tuple[dict, list[dict]]:
    metadata = client.package("private", requirement.package)
    if not metadata:
        return {}, []
    candidates = [
        record
        for record in metadata.get("versions", [])
        if satisfies(str(record["version"]), requirement.operator, requirement.version)
    ]
    return metadata, candidates


def _highest_private(client: IndexClient, requirement: Requirement) -> dict:
    metadata, candidates = _private_candidates(client, requirement)
    if not metadata:
        return {}
    if not candidates:
        raise RepairError(
            f"{requirement.text}: 私有仓库没有满足约束的版本"
        )
    return max(candidates, key=lambda item: Version(str(item["version"])))


def apply_pin(manifest_path: str | Path, client: IndexClient) -> int:
    path = Path(manifest_path)
    parsed = parse_manifest(path)
    updated: list[Requirement] = []
    changed = 0
    for requirement in parsed.requirements:
        selected = _highest_private(client, requirement)
        if not selected:
            # A public-only dependency has nothing to pin in the private repo.
            updated.append(requirement)
            continue
        if not (
            requirement.operator == "==" and Version(requirement.version) == Version(selected["version"])
        ):
            changed += 1
        updated.append(
            replace(
                requirement,
                operator="==",
                version=str(selected["version"]),
                hashes=(),
            )
        )
    write_manifest(path, updated)
    return changed


def apply_hash(manifest_path: str | Path, client: IndexClient) -> int:
    path = Path(manifest_path)
    parsed = parse_manifest(path)
    updated: list[Requirement] = []
    changed = 0
    for requirement in parsed.requirements:
        selected = _highest_private(client, requirement)
        if not selected:
            updated.append(requirement)
            continue
        expected_hash = str(selected["sha256"])
        if requirement.hashes != (expected_hash,):
            changed += 1
        updated.append(replace(requirement, hashes=(expected_hash,)))
    write_manifest(path, updated)
    return changed


def apply_private_first(manifest_path: str | Path) -> int:
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    without_policy = [
        line
        for line in lines
        if line.strip().lower() != "# depconfusion-policy: private-first"
    ]
    new_text = "\n".join(["# depconfusion-policy: private-first", *without_policy]) + "\n"
    atomic_write_text(path, new_text)
    policy_path = policy_file_for(path)
    policy_path.write_text(
        json.dumps(
            {
                "strategy": "private-first",
                "source_order": ["private", "public"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 1


def apply_strategy(
    strategy: str,
    manifest_path: str | Path,
    client: IndexClient | None = None,
) -> int:
    if strategy == "pin":
        if client is None:
            raise ValueError("pin 需要索引客户端")
        return apply_pin(manifest_path, client)
    if strategy == "hash":
        if client is None:
            raise ValueError("hash 需要索引客户端")
        return apply_hash(manifest_path, client)
    if strategy == "private-first":
        return apply_private_first(manifest_path)
    raise ValueError(f"未知修复策略: {strategy}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="对依赖清单施加依赖混淆修复策略")
    parser.add_argument("strategy_pos", nargs="?", choices=STRATEGIES)
    parser.add_argument("--strategy", dest="strategy_opt", choices=STRATEGIES)
    parser.add_argument("-m", "--manifest", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    strategy = args.strategy_opt or args.strategy_pos
    if not strategy:
        print("错误: 必须指定修复策略", file=sys.stderr)
        return 1
    manifest = Path(args.manifest)
    if not manifest.is_file():
        print(f"清单不存在: {manifest}", file=sys.stderr)
        return 1
    client = IndexClient(args.base_url)
    try:
        if strategy != "private-first":
            client.status()
        changed = apply_strategy(strategy, manifest, client)
    except FileNotFoundError:
        print(f"清单不存在: {manifest}", file=sys.stderr)
        return 1
    except IndexUnavailable as exc:
        print(f"索引服务不可达: {exc}", file=sys.stderr)
        return 2
    except (ApiError, RepairError, OSError, ValueError) as exc:
        print(f"修复失败: {exc}", file=sys.stderr)
        return 1
    _ = changed
    print(f"修复完成: {strategy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
