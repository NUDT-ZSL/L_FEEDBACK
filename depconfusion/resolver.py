"""Public-then-private resolver shared by every other command."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .client import ApiError, IndexClient, IndexUnavailable
from .manifest import (
    ParseResult,
    Requirement,
    detect_external_policy,
    parse_manifest,
)
from .versions import Version, satisfies


SOURCE_ORDERS = {
    "public-first": ("public", "private"),
    "private-first": ("private", "public"),
}


class Resolver:
    def __init__(self, client: IndexClient, source_order: tuple[str, str] = ("public", "private")):
        self.client = client
        self.source_order = source_order
        self._cache: dict[tuple[str, str], dict[str, Any] | None] = {}

    def package(self, repo: str, name: str) -> dict[str, Any] | None:
        key = (repo, name)
        if key not in self._cache:
            self._cache[key] = self.client.package(repo, name)
        return self._cache[key]

    def highest_matching(
        self,
        repo: str,
        requirement: Requirement,
        *,
        honor_hashes: bool = True,
    ) -> dict[str, Any] | None:
        metadata = self.package(repo, requirement.package)
        if not metadata:
            return None
        candidates: list[dict[str, Any]] = []
        for record in metadata.get("versions", []):
            try:
                if not satisfies(str(record["version"]), requirement.operator, requirement.version):
                    continue
            except ValueError:
                continue
            if honor_hashes and requirement.hashes:
                if str(record.get("sha256", "")).lower() not in requirement.hashes:
                    continue
            candidates.append(record)
        if not candidates:
            return None
        return max(candidates, key=lambda item: Version(str(item["version"])))

    def resolve_requirement(self, requirement: Requirement) -> dict[str, Any]:
        selected_repo: str | None = None
        selected: dict[str, Any] | None = None
        for repo in self.source_order:
            candidate = self.highest_matching(repo, requirement)
            if candidate is not None:
                selected_repo = repo
                selected = candidate
                break

        public_meta = self.package("public", requirement.package)
        private_meta = self.package("private", requirement.package)
        hash_verified: bool | None = None
        if selected is not None and requirement.hashes:
            hash_verified = str(selected["sha256"]).lower() in requirement.hashes

        result: dict[str, Any] = {
            "package": requirement.package,
            "requirement": requirement.text,
            "operator": requirement.operator,
            "constraint": requirement.version,
            "hashes": list(requirement.hashes),
            "line_number": requirement.line_number,
            "resolved": selected is not None,
            "status": "resolved" if selected is not None else "unresolved",
            "source": selected_repo,
            "version": selected["version"] if selected else None,
            "sha256": selected["sha256"] if selected else None,
            "uploaded_at": selected["uploaded_at"] if selected else None,
            "injected": bool(selected.get("injected", False)) if selected else False,
            "hash_verified": hash_verified,
            "private_present": private_meta is not None,
            "public_present": public_meta is not None,
            "private_highest": _highest_from_meta(private_meta),
            "public_highest": _highest_from_meta(public_meta),
            "public_injected_versions": _injected_versions(public_meta),
        }
        if selected is None and requirement.hashes:
            result["reason"] = "没有满足版本约束及哈希的可用版本"
        elif selected is None:
            result["reason"] = "没有满足版本约束的可用版本"
        return result

    def resolve_all(self, requirements: list[Requirement]) -> list[dict[str, Any]]:
        return [self.resolve_requirement(requirement) for requirement in requirements]


def _highest_from_meta(metadata: dict[str, Any] | None) -> str | None:
    if not metadata or not metadata.get("versions"):
        return None
    return max(metadata["versions"], key=lambda item: Version(str(item["version"])))["version"]


def _injected_versions(metadata: dict[str, Any] | None) -> list[str]:
    if not metadata:
        return []
    return sorted(
        [
            str(record["version"])
            for record in metadata.get("versions", [])
            if record.get("injected")
        ],
        key=Version,
    )


def resolve_parsed(parsed: ParseResult, resolver: Resolver) -> dict[str, Any]:
    results = resolver.resolve_all(parsed.requirements)
    resolved = [item for item in results if item["resolved"]]
    unresolved = [item for item in results if not item["resolved"]]
    return {
        "source_order": list(resolver.source_order),
        "warnings": list(parsed.warnings),
        "requirements": results,
        "resolved": resolved,
        "unresolved": unresolved,
    }


def resolve_manifest(
    manifest_path: str | Path,
    client: IndexClient,
    source_order: tuple[str, str] | None = None,
) -> dict[str, Any]:
    path = Path(manifest_path)
    parsed = parse_manifest(path)
    if source_order is None:
        source_order = parsed.source_order
        external = detect_external_policy(path)
        if external:
            source_order = external
    resolver = Resolver(client, source_order)
    result = resolve_parsed(parsed, resolver)
    result["manifest"] = str(path)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="按公共优先/私有优先顺序解析依赖清单")
    parser.add_argument("manifest_pos", nargs="?", help="依赖清单路径")
    parser.add_argument("-m", "--manifest", help="依赖清单路径")
    parser.add_argument("--base-url", default="http://127.0.0.1:5000", help="索引服务地址")
    parser.add_argument(
        "--policy",
        choices=("auto", "public-first", "private-first"),
        default="auto",
        help="来源顺序；auto 会识别清单中的 private-first 策略标记",
    )
    parser.add_argument("-o", "--output", help="将解析 JSON 写入文件")
    return parser


def _emit_warnings(warnings: list[str]) -> None:
    for warning in warnings:
        print(f"警告: {warning}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_value = args.manifest or args.manifest_pos
    if not manifest_value:
        print("错误: 必须指定 --manifest", file=sys.stderr)
        return 1
    manifest_path = Path(manifest_value)
    if not manifest_path.is_file():
        print(f"清单不存在: {manifest_path}", file=sys.stderr)
        return 1

    order = None
    if args.policy != "auto":
        order = SOURCE_ORDERS[args.policy]
    client = IndexClient(args.base_url)
    try:
        client.status()
        result = resolve_manifest(manifest_path, client, order)
    except FileNotFoundError:
        print(f"清单不存在: {manifest_path}", file=sys.stderr)
        return 1
    except IndexUnavailable as exc:
        print(f"索引服务不可达: {exc}", file=sys.stderr)
        return 2
    except (ApiError, OSError, ValueError) as exc:
        print(f"解析失败: {exc}", file=sys.stderr)
        return 1

    _emit_warnings(result.get("warnings", []))
    rendered = json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
