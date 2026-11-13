"""End-to-end offline demo.

Starts a real index service subprocess, injects a malicious public version,
detects it, applies all three mitigations, validates each one, and shuts the
service down.  The required output lines are emitted in order.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from depconfusion.client import ApiError, IndexClient, IndexUnavailable
from depconfusion.detector import detect
from depconfusion.repair import apply_strategy
from depconfusion.validator import validate_manifest


ROOT = Path(__file__).resolve().parent


def _wait_ready(proc: subprocess.Popen[str], base_url: str, timeout: float = 12.0) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            stdout, stderr = proc.communicate(timeout=1)
            details = (stderr or stdout or "").strip()
            raise RuntimeError(
                f"索引服务进程提前退出，退出码 {proc.returncode}"
                + (f": {details}" if details else "")
            )
        try:
            with urllib.request.urlopen(base_url + "/status", timeout=0.3) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("ready") is True and payload.get("pid") == proc.pid:
                return
        except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError) as exc:
            last_error = exc
        time.sleep(0.05)
    raise RuntimeError(f"等待索引服务就绪超时: {base_url} ({last_error})")


def _stop_service(
    proc: subprocess.Popen[str],
    client: IndexClient,
    *,
    request_shutdown: bool = True,
) -> None:
    if request_shutdown:
        try:
            client.shutdown()
        except Exception:
            # The endpoint may close its connection just after a successful
            # write; process exit below is the authoritative signal.
            pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=3)


def run_demo(host: str, port: int, seed: int) -> int:
    base_url = f"http://{host}:{port}"
    with tempfile.TemporaryDirectory(prefix="depconfusion-demo-") as temporary:
        work = Path(temporary)
        project_dir = work / "project"
        project_dir.mkdir(parents=True)
        manifest = project_dir / "requirements.txt"
        manifest.write_text("internal-utils>=1.0.0\n", encoding="utf-8")
        state_path = work / "index_state.json"
        report_path = work / "report.json"

        command = [
            sys.executable,
            str(ROOT / "service.py"),
            "--host",
            host,
            "--port",
            str(port),
            "--seed",
            str(seed),
            "--state",
            str(state_path),
        ]
        proc = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        client = IndexClient(base_url, attempts=1, timeout=1.0)
        service_closed = False
        service_ready = False
        try:
            try:
                _wait_ready(proc, base_url)
            except RuntimeError as exc:
                print(f"演示失败: {exc}", file=sys.stderr)
                return 2
            service_ready = True
            print("索引服务已启动")

            try:
                client.status()
                client.inject(
                    "internal-utils",
                    "9.9.9",
                    "demo-malicious-payload|internal-utils|9.9.9",
                )
            except (IndexUnavailable, ApiError, ValueError) as exc:
                print(f"演示失败: 注入攻击失败: {exc}", file=sys.stderr)
                return 1
            print("注入攻击完成")

            report = detect(project_dir, report_path, client)
            if "internal-utils" not in report["risks"]:
                print("演示失败: 未检测到 internal-utils 风险", file=sys.stderr)
                return 1
            print("检测到风险: internal-utils")

            for strategy in ("pin", "hash", "private-first"):
                try:
                    apply_strategy(strategy, manifest, client)
                except Exception as exc:
                    print(f"演示失败: 修复 {strategy} 失败: {exc}", file=sys.stderr)
                    return 1
                print(f"修复完成: {strategy}")

                try:
                    validation = validate_manifest(manifest, strategy, client)
                except Exception as exc:
                    print(f"演示失败: 验证 {strategy} 失败: {exc}", file=sys.stderr)
                    return 1
                if not validation["ok"]:
                    reasons = "; ".join(validation["errors"])
                    print(f"演示失败: 验证 {strategy}: {reasons}", file=sys.stderr)
                    return 1
                print(f"验证通过: {strategy}")

            _stop_service(proc, client)
            service_closed = True
            print("索引服务已关闭")
            return 0
        finally:
            if not service_closed:
                try:
                    _stop_service(proc, client, request_shutdown=service_ready)
                except Exception:
                    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="离线依赖混淆靶场端到端演示")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20240517)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_demo(args.host, args.port, args.seed)
    except KeyboardInterrupt:
        print("演示失败: 已中断", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
