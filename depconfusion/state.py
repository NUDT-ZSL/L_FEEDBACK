"""Deterministic index data and JSON persistence."""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from typing import Any

from .manifest import atomic_write_text, normalize_name
from .versions import Version, highest, parse_version


DEFAULT_SEED = 20240517
REPOSITORIES = ("public", "private")

# The two internal packages deliberately share a name in both repositories.
_PUBLIC_SPECS: dict[str, list[str]] = {
    "internal-utils": ["1.0.0", "1.2.0", "1.5.0"],
    "internal-auth": ["1.0.0", "1.1.0"],
    "internal-models": ["1.8.0", "2.0.1"],
    "internal-logging": ["0.9.0"],
    "requests": ["2.31.0"],
    "urllib3": ["2.2.1"],
    "flask": ["3.0.2"],
    "click": ["8.1.7"],
    "pytest": ["8.1.1"],
    "setuptools": ["69.0.3"],
    "jinja2": ["3.1.3"],
    "certifi": ["2024.2.2"],
    "idna": ["3.6"],
}

_PRIVATE_SPECS: dict[str, list[str]] = {
    "internal-utils": ["1.2.0", "2.4.0", "3.8.2"],
    "internal-auth": [],
    "internal-models": [],
    "internal-metrics": [],
    "internal-logging": [],
    "internal-config": [],
    "internal-cache": [],
    "internal-common": [],
    "internal-storage": [],
    "internal-cli": [],
}

_BASE_TIME = datetime(2024, 1, 1, tzinfo=timezone.utc)


class StateError(RuntimeError):
    pass


class InjectionError(ValueError):
    def __init__(self, message: str, status: int = 409):
        super().__init__(message)
        self.status = status


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _timestamp_for(seed: int, repo: str, name: str, version: str) -> str:
    offset_hash = hashlib.sha256(f"{seed}:{repo}:{name}:{version}".encode("utf-8")).hexdigest()
    seconds = int(offset_hash[:8], 16) % (365 * 24 * 3600)
    stamp = _BASE_TIME + timedelta(seconds=seconds)
    return stamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def _record(repo: str, name: str, version: str, seed: int, injected: bool = False) -> dict[str, Any]:
    payload = f"payload|repo={repo}|name={name}|version={version}|seed={seed}"
    return {
        "version": version,
        "sha256": _digest(payload),
        "uploaded_at": _timestamp_for(seed, repo, name, version),
        "injected": injected,
        "payload": payload,
    }


def _generated_private_versions(seed: int, name: str) -> list[str]:
    name_seed = int(hashlib.sha256(f"{seed}:{name}".encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(name_seed)
    count = 3 + rng.randrange(3)
    versions: list[str] = []
    for index in range(count):
        major = 1 + index // 2
        minor = rng.randrange(1, 8)
        patch = rng.randrange(0, 6)
        candidate = f"{major}.{minor}.{patch}"
        while candidate in versions:
            patch += 1
            candidate = f"{major}.{minor}.{patch}"
        versions.append(candidate)
    return versions


def build_initial_repositories(seed: int) -> dict[str, dict[str, dict[str, Any]]]:
    repositories: dict[str, dict[str, dict[str, Any]]] = {"public": {}, "private": {}}
    for name, versions in _PUBLIC_SPECS.items():
        repositories["public"][name] = {
            version: _record("public", name, version, seed) for version in versions
        }
    for name, versions in _PRIVATE_SPECS.items():
        if not versions:
            versions = _generated_private_versions(seed, name)
        repositories["private"][name] = {
            version: _record("private", name, version, seed) for version in versions
        }
    return repositories


def _canonical_repository(repo: str, packages: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "repo": repo,
        "packages": {
            name: [
                {
                    "version": record["version"],
                    "sha256": record["sha256"],
                    "uploaded_at": record["uploaded_at"],
                    "injected": bool(record.get("injected", False)),
                }
                for record in sorted(versions.values(), key=lambda item: Version(item["version"]))
            ]
            for name, versions in sorted(packages.items())
        },
    }


class IndexState:
    """Thread-safe in-memory view of the persisted JSON state file."""

    def __init__(self, data: dict[str, Any], path: str | Path):
        self.path = Path(path)
        self.data = data
        self._lock = RLock()
        self._validate()

    @classmethod
    def load_or_create(cls, path: str | Path, seed: int = DEFAULT_SEED) -> "IndexState":
        state_path = Path(path)
        if state_path.exists():
            try:
                data = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise StateError(f"索引状态文件无法读取: {state_path}: {exc}") from exc
            return cls(data, state_path)
        data = {
            "schema": 1,
            "seed": int(seed),
            "revision": 0,
            "repositories": build_initial_repositories(int(seed)),
        }
        state = cls(data, state_path)
        state.persist()
        return state

    def _validate(self) -> None:
        if not isinstance(self.data, dict):
            raise StateError("索引状态必须是 JSON 对象")
        if self.data.get("schema") != 1:
            raise StateError("不支持的索引状态 schema")
        repositories = self.data.get("repositories")
        if not isinstance(repositories, dict):
            raise StateError("索引状态缺少 repositories")
        for repo in REPOSITORIES:
            if not isinstance(repositories.get(repo), dict):
                raise StateError(f"索引状态缺少仓库: {repo}")
        for repo, packages in repositories.items():
            for name, versions in packages.items():
                if not isinstance(versions, dict) or not versions:
                    raise StateError(f"仓库 {repo} 的包 {name} 没有版本")
                for version, record in versions.items():
                    parse_version(version)
                    if record.get("version") != version:
                        raise StateError(f"包 {name} 的版本字段不一致")
                    digest = str(record.get("sha256", ""))
                    if len(digest) != 64 or digest.lower() != digest:
                        raise StateError(f"包 {name}=={version} 的 sha256 非法")
                    try:
                        int(digest, 16)
                    except ValueError as exc:
                        raise StateError(f"包 {name}=={version} 的 sha256 非法") from exc

    @property
    def seed(self) -> int:
        return int(self.data.get("seed", DEFAULT_SEED))

    @property
    def revision(self) -> int:
        return int(self.data.get("revision", 0))

    def repository(self, repo: str) -> dict[str, dict[str, Any]]:
        if repo not in REPOSITORIES:
            raise KeyError(repo)
        return self.data["repositories"][repo]

    def package_count(self, repo: str) -> int:
        return len(self.repository(repo))

    def versions(self, repo: str, name: str) -> list[dict[str, Any]]:
        normalized = normalize_name(name)
        with self._lock:
            return [
                dict(record)
                for record in sorted(
                    self.repository(repo).get(normalized, {}).values(),
                    key=lambda item: Version(item["version"]),
                )
            ]

    def package_metadata(self, repo: str, name: str) -> dict[str, Any] | None:
        versions = self.versions(repo, name)
        if not versions:
            return None
        selected = versions[-1]
        return {
            "repo": repo,
            "name": normalize_name(name),
            "version": selected["version"],
            "sha256": selected["sha256"],
            "uploaded_at": selected["uploaded_at"],
            "injected": bool(selected.get("injected", False)),
            "versions": versions,
        }

    def highest_private_version(self, name: str) -> str | None:
        versions = self.versions("private", name)
        return highest([record["version"] for record in versions])

    def repo_snapshot(self, repo: str) -> dict[str, Any]:
        with self._lock:
            packages = self.repository(repo)
            canonical = _canonical_repository(repo, packages)
            digest = hashlib.sha256(
                json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            ).hexdigest()
            version_counts = {name: len(versions) for name, versions in sorted(packages.items())}
            return {
                "repo": repo,
                "package_count": len(packages),
                "min_versions": min(version_counts.values()) if version_counts else 0,
                "version_counts": version_counts,
                "snapshot_sha256": digest,
            }

    def snapshot_hash(self) -> str:
        with self._lock:
            canonical = {
                repo: _canonical_repository(repo, self.repository(repo))
                for repo in REPOSITORIES
            }
            return hashlib.sha256(
                json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            ).hexdigest()

    def status(self, host: str, port: int, pid: int) -> dict[str, Any]:
        with self._lock:
            return {
                "ready": True,
                "schema": 1,
                "host": host,
                "port": port,
                "pid": pid,
                "seed": self.seed,
                "revision": self.revision,
                "repositories": {
                    repo: self.package_count(repo) for repo in REPOSITORIES
                },
                "snapshot_sha256": self.snapshot_hash(),
            }

    def inject_public(self, name: str, version: str, payload: str) -> dict[str, Any]:
        normalized = normalize_name(name)
        if not normalized:
            raise InjectionError("包名不能为空", 400)
        if not isinstance(payload, str) or payload == "":
            raise InjectionError("payload 不能为空", 400)
        try:
            injected_version = parse_version(version)
        except ValueError as exc:
            raise InjectionError(str(exc), 400) from exc
        version = injected_version.text

        with self._lock:
            private_versions = self.repository("private").get(normalized)
            if not private_versions:
                raise InjectionError(f"私有仓库不存在包 {normalized}", 409)
            private_highest = highest([record["version"] for record in private_versions.values()])
            assert private_highest is not None
            if injected_version <= Version(private_highest):
                raise InjectionError(
                    f"注入版本 {version} 必须高于私有最高版本 {private_highest}",
                    409,
                )
            public_versions = self.repository("public").setdefault(normalized, {})
            if version in public_versions:
                raise InjectionError(f"公共仓库已存在 {normalized}=={version}", 409)

            payload_digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            sequence = self.revision + 1
            offset_hash = hashlib.sha256(
                f"{self.seed}:inject:{normalized}:{version}:{sequence}".encode("utf-8")
            ).hexdigest()
            stamp = _BASE_TIME + timedelta(seconds=int(offset_hash[:8], 16) % (365 * 24 * 3600))
            record = {
                "version": version,
                "sha256": payload_digest,
                "uploaded_at": stamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "injected": True,
                "payload": payload,
            }
            public_versions[version] = record
            self.data["revision"] = sequence
            self.persist()
            return {
                "repo": "public",
                "name": normalized,
                "version": version,
                "sha256": payload_digest,
                "uploaded_at": record["uploaded_at"],
                "injected": True,
            }

    def mark_snapshot_rebuilt(self, repo: str) -> dict[str, Any]:
        if repo not in REPOSITORIES:
            raise KeyError(repo)
        with self._lock:
            self.data["revision"] = self.revision + 1
            self.persist()
            snapshot = self.repo_snapshot(repo)
            snapshot["rebuilt"] = True
            snapshot["revision"] = self.revision
            return snapshot

    def persist(self) -> None:
        with self._lock:
            text = json.dumps(self.data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
            atomic_write_text(self.path, text)


def validate_injection_for_state(state: IndexState, name: str, version: str, payload: str) -> None:
    """Public validation hook shared by service and CLI.

    The actual mutation performs the same checks under one lock; this helper lets
    the CLI preflight and keeps the error messages identical.
    """

    normalized = normalize_name(name)
    try:
        candidate = parse_version(version)
    except ValueError as exc:
        raise InjectionError(str(exc), 400) from exc
    private_highest = state.highest_private_version(normalized)
    if private_highest is None:
        raise InjectionError(f"私有仓库不存在包 {normalized}", 409)
    if candidate <= Version(private_highest):
        raise InjectionError(f"注入版本 {version} 必须高于私有最高版本 {private_highest}", 409)
    if state.package_metadata("public", normalized) and any(
        record["version"] == version for record in state.versions("public", normalized)
    ):
        raise InjectionError(f"公共仓库已存在 {normalized}=={version}", 409)
    if not payload:
        raise InjectionError("payload 不能为空", 400)
