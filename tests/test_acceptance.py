"""Acceptance tests for the offline dependency-confusion lab."""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from depconfusion.client import ApiError, IndexClient
from depconfusion.state import build_initial_repositories
from depconfusion.versions import Version, highest, satisfies


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def utf8_child_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_ready(base_url: str, proc: subprocess.Popen[str], timeout: float = 8.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            stdout, stderr = proc.communicate(timeout=1)
            raise AssertionError(
                f"service exited early ({proc.returncode}): {(stderr or stdout).strip()}"
            )
        try:
            with urllib.request.urlopen(base_url + "/status", timeout=0.2) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("ready") is True:
                return
        except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError):
            pass
        time.sleep(0.03)
    raise AssertionError(f"service did not become ready: {base_url}")


def http_status(url: str) -> int:
    try:
        with urllib.request.urlopen(url, timeout=1) as response:
            return int(response.status)
    except urllib.error.HTTPError as exc:
        code = int(exc.code)
        exc.close()
        return code


class RunningService:
    def __init__(self):
        self.temp = tempfile.TemporaryDirectory(prefix="dc-test-")
        self.work = Path(self.temp.name)
        self.port = free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.state_path = self.work / "index_state.json"
        self.client = IndexClient(self.base_url, attempts=1, timeout=1.0)
        self.proc: subprocess.Popen[str] | None = None

    def __enter__(self) -> "RunningService":
        self.proc = subprocess.Popen(
            [
                PYTHON,
                str(ROOT / "service.py"),
                "--port",
                str(self.port),
                "--state",
                str(self.state_path),
            ],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=utf8_child_env(),
        )
        wait_ready(self.base_url, self.proc)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.proc is not None:
            try:
                self.client.shutdown()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=4)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=2)
            self.proc.communicate(timeout=2)
        self.temp.cleanup()


def run_cli(script: str, args: list[str], *, timeout: float = 12.0, env=None) -> subprocess.CompletedProcess[str]:
    run_env = utf8_child_env()
    if env:
        run_env.update(env)
    return subprocess.run(
        [PYTHON, str(ROOT / script), *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=run_env,
    )


class VersionAndStateTests(unittest.TestCase):
    def test_version_ordering_and_equality(self) -> None:
        self.assertEqual(highest(["1.0", "1.10.0", "1.2.0", "1.0rc1"]), "1.10.0")
        self.assertTrue(Version("1.0") == Version("1.0.0"))
        self.assertTrue(Version("1.0rc1") < Version("1.0"))
        self.assertTrue(Version("1.0") < Version("1.0.post1"))
        self.assertTrue(satisfies("3.8.2", ">=", "1.0.0"))
        self.assertTrue(satisfies("3.8.2", "==", "3.8.2.0"))  # equal by normalized release

    def test_seed_generation_is_deterministic(self) -> None:
        first = build_initial_repositories(20240517)
        second = build_initial_repositories(20240517)
        other = build_initial_repositories(42)
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertGreaterEqual(len(first["private"]), 8)
        self.assertTrue(all(len(versions) >= 3 for versions in first["private"].values()))
        self.assertEqual(highest(list(first["private"]["internal-utils"])), "3.8.2")


class ServiceTests(unittest.TestCase):
    def test_status_package_snapshot_contract(self) -> None:
        with RunningService() as service:
            status = service.client.status()
            self.assertTrue(status["ready"])
            self.assertGreaterEqual(status["repositories"]["private"], 8)
            self.assertRegex(status["snapshot_sha256"], SHA_RE)

            metadata = service.client.package("public", "internal-utils")
            assert metadata is not None
            for key in ("name", "version", "sha256", "uploaded_at"):
                self.assertIn(key, metadata)
            self.assertRegex(metadata["sha256"], SHA_RE)
            self.assertTrue(metadata["uploaded_at"].endswith("Z"))

            self.assertEqual(http_status(service.base_url + "/nope/package"), 400)
            self.assertEqual(http_status(service.base_url + "/public/no-such-package"), 404)

            snapshot = service.client.snapshot("private")
            self.assertEqual(snapshot["package_count"], 10)
            self.assertGreaterEqual(snapshot["min_versions"], 3)
            self.assertRegex(snapshot["snapshot_sha256"], SHA_RE)
            rebuilt = service.client.rebuild_snapshot("private")
            self.assertTrue(rebuilt["rebuilt"])
            self.assertEqual(rebuilt["snapshot_sha256"], snapshot["snapshot_sha256"])

    def test_shared_injection_validation_and_snapshot_change(self) -> None:
        with RunningService() as service:
            before = service.client.status()["snapshot_sha256"]
            result = service.client.inject(
                "internal-utils", "10.0.0", "malicious payload"
            )
            self.assertEqual(result["version"], "10.0.0")
            self.assertRegex(result["sha256"], SHA_RE)
            after = service.client.status()["snapshot_sha256"]
            self.assertNotEqual(before, after)

            with self.assertRaises(ApiError) as lower:
                service.client.inject("internal-utils", "3.8.2", "too low")
            self.assertEqual(lower.exception.status, 409)
            self.assertIn("必须高于私有最高版本 3.8.2", str(lower.exception))

            with self.assertRaises(ApiError) as invalid:
                service.client.inject("internal-utils", "nine", "bad")
            self.assertEqual(invalid.exception.status, 400)

            with self.assertRaises(ApiError) as missing_private:
                service.client.inject("not-in-private", "9.9.9", "bad")
            self.assertEqual(missing_private.exception.status, 409)

            persisted = service.client.package("public", "internal-utils")
            assert persisted is not None
            self.assertEqual(persisted["version"], "10.0.0")
            self.assertTrue(persisted["injected"])

    def test_state_persists_across_service_restart(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dc-restart-") as raw:
            work = Path(raw)
            port = free_port()
            state_path = work / "state.json"
            base_url = f"http://127.0.0.1:{port}"
            first = subprocess.Popen(
                [
                    PYTHON,
                    str(ROOT / "service.py"),
                    "--port",
                    str(port),
                    "--state",
                    str(state_path),
                ],
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=utf8_child_env(),
            )
            wait_ready(base_url, first)
            client = IndexClient(base_url, attempts=1)
            client.inject("internal-utils", "10.0.0", "persisted")
            snapshot = client.status()["snapshot_sha256"]
            client.shutdown()
            first.wait(timeout=4)
            first.communicate(timeout=2)

            restarted = subprocess.Popen(
                [
                    PYTHON,
                    str(ROOT / "service.py"),
                    "--port",
                    str(port),
                    "--state",
                    str(state_path),
                ],
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=utf8_child_env(),
            )
            try:
                wait_ready(base_url, restarted)
                restarted_client = IndexClient(base_url, attempts=1)
                self.assertEqual(restarted_client.status()["snapshot_sha256"], snapshot)
                metadata = restarted_client.package("public", "internal-utils")
                assert metadata is not None
                self.assertEqual(metadata["version"], "10.0.0")
            finally:
                try:
                    IndexClient(base_url, attempts=1, timeout=0.5).shutdown()
                except Exception:
                    pass
                try:
                    restarted.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    restarted.terminate()
                    restarted.wait(timeout=3)
                restarted.communicate(timeout=2)

    def test_port_conflict_is_fast_and_does_not_touch_existing_service(self) -> None:
        with RunningService() as service:
            before = service.client.status()
            other_state = service.work / "other_state.json"
            started = time.monotonic()
            result = subprocess.run(
                [
                    PYTHON,
                    str(ROOT / "service.py"),
                    "--port",
                    str(service.port),
                    "--state",
                    str(other_state),
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
                env=utf8_child_env(),
            )
            elapsed = time.monotonic() - started
            self.assertEqual(result.returncode, 2)
            self.assertIn("端口占用或监听失败", result.stderr)
            self.assertLess(elapsed, 2.0)
            self.assertFalse(other_state.exists())
            self.assertEqual(service.client.status()["revision"], before["revision"])


class ResolverAndCliTests(unittest.TestCase):
    def test_resolver_public_then_private_private_first_and_unresolved(self) -> None:
        with RunningService() as service:
            manifest = service.work / "requirements.txt"
            manifest.write_text("internal-utils>=1.0.0\n", encoding="utf-8")
            public = run_cli(
                "resolver.py",
                ["--manifest", str(manifest), "--base-url", service.base_url],
            )
            self.assertEqual(public.returncode, 0, public.stderr)
            payload = json.loads(public.stdout)
            self.assertEqual(payload["resolved"][0]["source"], "public")
            self.assertEqual(payload["resolved"][0]["version"], "1.5.0")
            self.assertEqual(payload["unresolved"], [])

            private = run_cli(
                "resolver.py",
                [
                    "--manifest",
                    str(manifest),
                    "--policy",
                    "private-first",
                    "--base-url",
                    service.base_url,
                ],
            )
            self.assertEqual(private.returncode, 0, private.stderr)
            private_payload = json.loads(private.stdout)
            self.assertEqual(private_payload["resolved"][0]["source"], "private")
            self.assertEqual(private_payload["resolved"][0]["version"], "3.8.2")

            unresolved_manifest = service.work / "unresolved.txt"
            unresolved_manifest.write_text("internal-utils>=99.0.0\n", encoding="utf-8")
            unresolved = run_cli(
                "resolver.py",
                ["--manifest", str(unresolved_manifest), "--base-url", service.base_url],
            )
            self.assertEqual(unresolved.returncode, 0, unresolved.stderr)
            unresolved_payload = json.loads(unresolved.stdout)
            self.assertEqual(unresolved_payload["resolved"], [])
            self.assertEqual(len(unresolved_payload["unresolved"]), 1)
            self.assertEqual(unresolved_payload["unresolved"][0]["status"], "unresolved")

    def test_injector_changes_resolution_and_logs_success_failure(self) -> None:
        with RunningService() as service:
            log_path = service.work / "inject.log"
            success = run_cli(
                "injector.py",
                [
                    "--package",
                    "internal-utils",
                    "--version",
                    "9.9.9",
                    "--payload",
                    "evil",
                    "--base-url",
                    service.base_url,
                    "--log",
                    str(log_path),
                ],
            )
            self.assertEqual(success.returncode, 0, success.stderr)
            self.assertIn("注入成功", success.stdout)
            log = log_path.read_text(encoding="utf-8")
            self.assertIn("SUCCESS", log)
            self.assertRegex(log.splitlines()[-1], r"sha256=[0-9a-f]{64}$")

            manifest = service.work / "requirements.txt"
            manifest.write_text("internal-utils>=1.0.0\n", encoding="utf-8")
            resolved = run_cli(
                "resolver.py",
                ["--manifest", str(manifest), "--base-url", service.base_url],
            )
            self.assertEqual(resolved.returncode, 0, resolved.stderr)
            self.assertEqual(json.loads(resolved.stdout)["resolved"][0]["version"], "9.9.9")

            failure = run_cli(
                "injector.py",
                [
                    "--package",
                    "internal-utils",
                    "--version",
                    "1.0.0",
                    "--payload",
                    "bad",
                    "--base-url",
                    service.base_url,
                    "--log",
                    str(log_path),
                ],
            )
            self.assertEqual(failure.returncode, 1)
            self.assertIn("注入失败", failure.stderr)
            log = log_path.read_text(encoding="utf-8")
            self.assertIn("FAILURE", log)
            self.assertIn("必须高于私有最高版本", log)

    def test_missing_manifest_and_unparseable_lines(self) -> None:
        with RunningService() as service:
            missing = run_cli(
                "resolver.py",
                ["--manifest", str(service.work / "missing.txt"), "--base-url", service.base_url],
            )
            self.assertEqual(missing.returncode, 1)
            self.assertIn("清单不存在", missing.stderr)

            manifest = service.work / "mixed.txt"
            manifest.write_text(
                "this is not a requirement\n\ninternal-utils==1.5.0  # pinned\n",
                encoding="utf-8",
            )
            result = run_cli(
                "resolver.py",
                ["--manifest", str(manifest), "--base-url", service.base_url],
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("警告", result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(len(payload["resolved"]), 1)
            self.assertEqual(payload["warnings"], ["第 1 行已跳过: 应为 name==version 或 name>=version"])

    def test_unreachable_index_retries_and_exits_two(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dc-unreachable-") as raw:
            manifest = Path(raw) / "requirements.txt"
            manifest.write_text("internal-utils>=1.0.0\n", encoding="utf-8")
            unreachable_port = free_port()
            started = time.monotonic()
            result = run_cli(
                "resolver.py",
                [
                    "--manifest",
                    str(manifest),
                    "--base-url",
                    f"http://127.0.0.1:{unreachable_port}",
                ],
                timeout=8,
            )
            elapsed = time.monotonic() - started
            self.assertEqual(result.returncode, 2)
            self.assertIn("索引服务不可达", result.stderr)
            self.assertIn("已尝试 3 次", result.stderr)
            self.assertGreaterEqual(elapsed, 1.8)


class DetectorAndRepairTests(unittest.TestCase):
    def test_detector_risks_latent_summary_and_stale(self) -> None:
        with RunningService() as service:
            service.client.inject("internal-utils", "9.9.9", "evil")
            project = service.work / "project"
            project.mkdir()
            manifest = project / "requirements.txt"
            report_path = service.work / "report.json"

            manifest.write_text("internal-utils>=1.0.0\n", encoding="utf-8")
            result = run_cli(
                "detector.py",
                [
                    "--dir",
                    str(project),
                    "--output",
                    str(report_path),
                    "--base-url",
                    service.base_url,
                ],
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["risks"], ["internal-utils"])
            self.assertEqual(report["latent"], [])
            self.assertFalse(report["stale"])
            self.assertRegex(report["snapshot_sha256"], SHA_RE)
            self.assertEqual(
                report["summary"],
                {
                    "packages": 1,
                    "requirements": 1,
                    "resolved": 1,
                    "unresolved": 0,
                    "risks": 1,
                    "risk_count": 1,
                    "latent": 0,
                    "latent_count": 0,
                },
            )

            manifest.write_text("internal-utils==3.8.2\n", encoding="utf-8")
            result = run_cli(
                "detector.py",
                [
                    "--dir",
                    str(project),
                    "--output",
                    str(report_path),
                    "--base-url",
                    service.base_url,
                ],
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            latent_report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(latent_report["risks"], [])
            self.assertEqual(latent_report["latent"], ["internal-utils"])
            self.assertEqual(latent_report["summary"]["risk_count"], 0)
            self.assertEqual(latent_report["summary"]["latent_count"], 1)

            detector = subprocess.Popen(
                [
                    PYTHON,
                    str(ROOT / "detector.py"),
                    "--dir",
                    str(project),
                    "--output",
                    str(report_path),
                    "--base-url",
                    service.base_url,
                    "--scan-delay",
                    "1.2",
                ],
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=utf8_child_env(),
            )
            time.sleep(0.35)
            injected = run_cli(
                "injector.py",
                [
                    "--package",
                    "internal-utils",
                    "--version",
                    "10.0.0",
                    "--payload",
                    "mid-scan",
                    "--base-url",
                    service.base_url,
                    "--log",
                    str(service.work / "stale-inject.log"),
                ],
            )
            self.assertEqual(injected.returncode, 0, injected.stderr)
            stdout, stderr = detector.communicate(timeout=8)
            self.assertEqual(detector.returncode, 0, stderr)
            self.assertIn("stale=True", stdout)
            stale_report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertTrue(stale_report["stale"])
            self.assertNotEqual(
                stale_report["snapshot_sha256"],
                stale_report["final_snapshot_sha256"],
            )

    def test_all_repairs_and_validators_pass(self) -> None:
        with RunningService() as service:
            service.client.inject("internal-utils", "9.9.9", "evil")
            private_meta = service.client.package("private", "internal-utils")
            assert private_meta is not None
            private_sha = private_meta["sha256"]

            pin_manifest = service.work / "pin.txt"
            pin_manifest.write_text("internal-utils>=1.0.0\n", encoding="utf-8")
            pin = run_cli(
                "repair.py",
                ["pin", "--manifest", str(pin_manifest), "--base-url", service.base_url],
            )
            self.assertEqual(pin.returncode, 0, pin.stderr)
            self.assertIn("internal-utils==3.8.2", pin_manifest.read_text(encoding="utf-8"))
            validated = run_cli(
                "validator.py",
                ["pin", "--manifest", str(pin_manifest), "--base-url", service.base_url],
            )
            self.assertEqual(validated.returncode, 0, validated.stderr)
            self.assertIn("PASS: pin", validated.stdout)

            hash_manifest = service.work / "hash.txt"
            hash_manifest.write_text("internal-utils>=1.0.0\n", encoding="utf-8")
            hashed = run_cli(
                "repair.py",
                ["hash", "--manifest", str(hash_manifest), "--base-url", service.base_url],
            )
            self.assertEqual(hashed.returncode, 0, hashed.stderr)
            self.assertIn(f"--hash=sha256:{private_sha}", hash_manifest.read_text(encoding="utf-8"))
            hash_validated = run_cli(
                "validator.py",
                ["hash", "--manifest", str(hash_manifest), "--base-url", service.base_url],
            )
            self.assertEqual(hash_validated.returncode, 0, hash_validated.stderr)
            self.assertIn("PASS: hash", hash_validated.stdout)

            private_manifest = service.work / "private-first.txt"
            private_manifest.write_text("internal-utils>=1.0.0\n", encoding="utf-8")
            private_first = run_cli(
                "repair.py",
                [
                    "private-first",
                    "--manifest",
                    str(private_manifest),
                ],
            )
            self.assertEqual(private_first.returncode, 0, private_first.stderr)
            self.assertIn(
                "# depconfusion-policy: private-first",
                private_manifest.read_text(encoding="utf-8"),
            )
            self.assertTrue((service.work / ".depconfusion-policy.json").is_file())
            auto_resolved = run_cli(
                "resolver.py",
                ["--manifest", str(private_manifest), "--base-url", service.base_url],
            )
            self.assertEqual(auto_resolved.returncode, 0, auto_resolved.stderr)
            self.assertEqual(json.loads(auto_resolved.stdout)["resolved"][0]["source"], "private")
            private_validated = run_cli(
                "validator.py",
                [
                    "private-first",
                    "--manifest",
                    str(private_manifest),
                    "--base-url",
                    service.base_url,
                ],
            )
            self.assertEqual(private_validated.returncode, 0, private_validated.stderr)
            self.assertIn("PASS: private-first", private_validated.stdout)

    def test_repairs_and_validators_fail_with_reasons(self) -> None:
        with RunningService() as service:
            service.client.inject("internal-utils", "9.9.9", "evil")
            plain = service.work / "plain.txt"
            plain.write_text("internal-utils>=1.0.0\n", encoding="utf-8")

            pin_fail = run_cli(
                "validator.py",
                ["pin", "--manifest", str(plain), "--base-url", service.base_url],
            )
            self.assertNotEqual(pin_fail.returncode, 0)
            self.assertIn("FAIL", pin_fail.stderr)
            self.assertIn("精确版本", pin_fail.stderr)

            hash_fail = run_cli(
                "validator.py",
                ["hash", "--manifest", str(plain), "--base-url", service.base_url],
            )
            self.assertNotEqual(hash_fail.returncode, 0)
            self.assertIn("FAIL", hash_fail.stderr)
            self.assertIn("--hash=sha256", hash_fail.stderr)

            impossible = service.work / "impossible.txt"
            impossible.write_text("internal-utils>=99.0.0\n", encoding="utf-8")
            hash_repair_fail = run_cli(
                "repair.py",
                ["hash", "--manifest", str(impossible), "--base-url", service.base_url],
            )
            self.assertEqual(hash_repair_fail.returncode, 1)
            self.assertIn("私有仓库没有满足约束的版本", hash_repair_fail.stderr)

            unresolved = service.work / "unresolved.txt"
            unresolved.write_text("internal-utils==99.0.0\n", encoding="utf-8")
            unresolved_fail = run_cli(
                "validator.py",
                ["pin", "--manifest", str(unresolved), "--base-url", service.base_url],
            )
            self.assertNotEqual(unresolved_fail.returncode, 0)
            self.assertIn("FAIL", unresolved_fail.stderr)


class DemoLifecycleTests(unittest.TestCase):
    def test_demo_port_conflict_does_not_touch_existing_service(self) -> None:
        with RunningService() as service:
            before = service.client.status()
            failed_demo = run_cli("demo.py", ["--port", str(service.port)], timeout=20)
            self.assertNotEqual(failed_demo.returncode, 0)
            self.assertIn("端口占用或监听失败", failed_demo.stderr)
            after = service.client.status()
            self.assertEqual(after["pid"], before["pid"])
            self.assertEqual(after["revision"], before["revision"])

    def test_demo_output_order_and_restart_on_same_port(self) -> None:
        port = free_port()
        expected = [
            "索引服务已启动",
            "注入攻击完成",
            "检测到风险: internal-utils",
            "修复完成: pin",
            "验证通过: pin",
            "修复完成: hash",
            "验证通过: hash",
            "修复完成: private-first",
            "验证通过: private-first",
            "索引服务已关闭",
        ]
        first = run_cli("demo.py", ["--port", str(port)], timeout=20)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(first.stdout.strip().splitlines(), expected)

        second = run_cli("demo.py", ["--port", str(port)], timeout=20)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(second.stdout.strip().splitlines(), expected)


if __name__ == "__main__":
    unittest.main()
