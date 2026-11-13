"""Offline dual-repository HTTP index service."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .state import (
    DEFAULT_SEED,
    InjectionError,
    IndexState,
    REPOSITORIES,
    StateError,
)


class IndexHTTPServer(ThreadingHTTPServer):
    # POSIX needs SO_REUSEADDR to rebind immediately after the demo's short
    # TIME_WAIT window.  Windows keeps an exclusive binding so an active
    # listener is reported as a real port conflict instead of being shared.
    allow_reuse_address = os.name != "nt"
    daemon_threads = True

    def __init__(self, address: tuple[str, int], state: IndexState | None = None):
        self.state = state
        self.host = address[0]
        super().__init__(address, IndexRequestHandler)


class IndexRequestHandler(BaseHTTPRequestHandler):
    server: IndexHTTPServer

    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep service stderr quiet for deterministic demo output.
        return

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(raw)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _error(self, status: int, message: str) -> None:
        self._send_json(status, {"status": status, "error": message})

    def _segments(self) -> tuple[str, ...]:
        parsed = urllib.parse.urlparse(self.path)
        return tuple(
            urllib.parse.unquote(segment)
            for segment in parsed.path.strip("/").split("/")
            if segment
        )

    def _validate_repo(self, repo: str) -> bool:
        if repo not in REPOSITORIES:
            self._error(400, f"未知仓库: {repo}")
            return False
        return True

    def do_GET(self) -> None:
        segments = self._segments()
        if segments == ("status",):
            host, port = self.server.server_address[:2]
            self._send_json(
                HTTPStatus.OK,
                self.server.state.status(str(self.server.host), int(port), os.getpid()),
            )
            return
        if not segments:
            self._error(404, "未知资源")
            return

        repo = segments[0]
        if not self._validate_repo(repo):
            return
        if len(segments) == 2 and segments[1] == "_snapshot":
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            if query.get("rebuild", ["false"])[-1].lower() in ("1", "true", "yes"):
                snapshot = self.server.state.mark_snapshot_rebuilt(repo)
            else:
                snapshot = self.server.state.repo_snapshot(repo)
                snapshot["rebuilt"] = False
                snapshot["revision"] = self.server.state.revision
            self._send_json(HTTPStatus.OK, snapshot)
            return
        if len(segments) == 2:
            metadata = self.server.state.package_metadata(repo, segments[1])
            if metadata is None:
                self._error(404, f"仓库 {repo} 中不存在包 {segments[1]}")
                return
            self._send_json(HTTPStatus.OK, metadata)
            return
        if len(segments) == 3:
            versions = self.server.state.versions(repo, segments[1])
            selected = next(
                (record for record in versions if record["version"] == segments[2]),
                None,
            )
            if selected is None:
                self._error(404, f"仓库 {repo} 中不存在包 {segments[1]}=={segments[2]}")
                return
            selected = dict(selected)
            selected.update({"repo": repo, "name": segments[1]})
            self._send_json(HTTPStatus.OK, selected)
            return
        self._error(404, "未知资源")

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0") or 0)
        if content_length <= 0:
            return {}
        raw = self.rfile.read(content_length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError(f"请求体不是合法 JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("请求体必须是 JSON 对象")
        return payload

    def do_POST(self) -> None:
        segments = self._segments()
        if segments == ("shutdown",):
            try:
                self.server.state.persist()
            except OSError as exc:
                self._error(500, f"索引状态落盘失败: {exc}")
                return
            self._send_json(
                HTTPStatus.OK,
                {"ok": True, "message": "索引状态已落盘，服务正在关闭"},
            )

            def delayed_shutdown() -> None:
                # Give the client time to consume the complete response before
                # server_close() tears down the socket during TIME_WAIT.
                threading.Event().wait(0.25)
                self.server.shutdown()

            threading.Thread(target=delayed_shutdown, daemon=True).start()
            return

        if not segments:
            self._error(404, "未知资源")
            return
        repo = segments[0]
        if not self._validate_repo(repo):
            return

        try:
            payload = self._read_json()
        except ValueError as exc:
            self._error(400, str(exc))
            return

        if len(segments) == 2 and segments[1] == "_snapshot":
            snapshot = self.server.state.mark_snapshot_rebuilt(repo)
            self._send_json(HTTPStatus.OK, snapshot)
            return

        if len(segments) == 2 and repo == "public":
            name = payload.get("package") or payload.get("name") or segments[1]
            version = str(payload.get("version", ""))
            attack_payload = payload.get("payload")
            try:
                result = self.server.state.inject_public(
                    str(name),
                    version,
                    attack_payload if isinstance(attack_payload, str) else "",
                )
            except InjectionError as exc:
                self._error(exc.status, str(exc))
                return
            self._send_json(HTTPStatus.OK, result)
            return

        self._error(404, "未知资源")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="启动离线公共/私有 Python 包索引")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--state", default=str(Path("data") / "index_state.json"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        server = IndexHTTPServer((args.host, args.port))
    except OSError as exc:
        print(
            f"端口占用或监听失败 {args.host}:{args.port}: {exc}",
            file=sys.stderr,
        )
        return 2

    try:
        state = IndexState.load_or_create(args.state, args.seed)
    except (OSError, StateError) as exc:
        server.server_close()
        print(f"索引状态初始化失败: {exc}", file=sys.stderr)
        return 1

    server.state = state
    try:
        server.serve_forever(poll_interval=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            state.persist()
        finally:
            server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
