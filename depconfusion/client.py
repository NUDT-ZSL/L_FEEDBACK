"""Small standard-library HTTP client with deterministic retry behavior."""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable


class IndexUnavailable(RuntimeError):
    def __init__(self, message: str):
        super().__init__(message)


class ApiError(RuntimeError):
    def __init__(self, message: str, status: int, payload: Any = None):
        super().__init__(message)
        self.status = status
        self.payload = payload


class IndexClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:5000",
        *,
        timeout: float = 1.0,
        attempts: int = 3,
        retry_interval: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.attempts = attempts
        self.retry_interval = retry_interval
        self._sleep = sleep

    def _request_once(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        url = self.base_url + path
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json; charset=utf-8"
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
                content_type = response.headers.get("Content-Type", "")
                if "application/json" in content_type:
                    return json.loads(raw.decode("utf-8"))
                return raw.decode("utf-8")
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            try:
                decoded = json.loads(raw.decode("utf-8")) if raw else None
            except ValueError:
                decoded = raw.decode("utf-8", errors="replace")
            if isinstance(decoded, dict):
                message = str(decoded.get("error") or decoded.get("message") or exc.reason)
            else:
                message = str(decoded or exc.reason)
            raise ApiError(f"HTTP {exc.code}: {message}", exc.code, decoded) from exc

    def request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, self.attempts + 1):
            try:
                return self._request_once(method, path, payload)
            except ApiError:
                raise
            except (
                urllib.error.URLError,
                socket.timeout,
                TimeoutError,
                ConnectionError,
                OSError,
            ) as exc:
                last_error = exc
                if attempt < self.attempts:
                    self._sleep(self.retry_interval)
        raise IndexUnavailable(
            f"索引服务不可达（已尝试 {self.attempts} 次）: {self.base_url}: {last_error}"
        )

    def status(self) -> dict[str, Any]:
        result = self.request("GET", "/status")
        if not isinstance(result, dict):
            raise RuntimeError("索引服务 /status 返回格式非法")
        return result

    def package(self, repo: str, name: str) -> dict[str, Any] | None:
        path = f"/{urllib.parse.quote(repo)}/{urllib.parse.quote(name)}"
        try:
            result = self.request("GET", path)
        except ApiError as exc:
            if exc.status == 404:
                return None
            raise
        return result

    def snapshot(self, repo: str) -> dict[str, Any]:
        result = self.request("GET", f"/{urllib.parse.quote(repo)}/_snapshot")
        return result

    def rebuild_snapshot(self, repo: str) -> dict[str, Any]:
        result = self.request("POST", f"/{urllib.parse.quote(repo)}/_snapshot", {})
        return result

    def inject(self, name: str, version: str, payload: str) -> dict[str, Any]:
        path = f"/public/{urllib.parse.quote(name)}"
        result = self.request(
            "POST",
            path,
            {"package": name, "version": version, "payload": payload},
        )
        return result

    def shutdown(self) -> dict[str, Any]:
        result = self.request("POST", "/shutdown", {})
        return result
