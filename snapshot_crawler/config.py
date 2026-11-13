"""JSON configuration loading and command line parsing."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceConfig:
    name: str
    root_url: str
    per_source_concurrency: int
    start_interval: float


@dataclass
class Config:
    sources: list[SourceConfig]
    global_concurrency: int
    max_depth: int
    max_pages: int
    timeout: float
    max_retries: int
    backoff_base: float
    backoff_cap: float
    circuit_threshold: int
    circuit_cooldown: float
    max_bytes: int
    user_agent: str
    output_dir: str

    @staticmethod
    def load(path: str | Path, overrides: dict | None = None) -> "Config":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        overrides = overrides or {}

        def pick(key, default):
            if key in overrides and overrides[key] is not None:
                return overrides[key]
            return data.get(key, default)

        sources = []
        for idx, item in enumerate(data["sources"]):
            sources.append(
                SourceConfig(
                    name=str(item.get("name", f"source{idx}")),
                    root_url=item["root_url"],
                    per_source_concurrency=int(
                        item.get(
                            "per_source_concurrency",
                            data.get("per_source_concurrency", 2),
                        )
                    ),
                    start_interval=float(
                        item.get(
                            "start_interval",
                            data.get("start_interval", 0.0),
                        )
                    ),
                )
            )

        cfg = Config(
            sources=sources,
            global_concurrency=int(pick("global_concurrency", 4)),
            max_depth=int(pick("max_depth", 3)),
            max_pages=int(pick("max_pages", 100)),
            timeout=float(pick("timeout", 10.0)),
            max_retries=int(pick("max_retries", 3)),
            backoff_base=float(pick("backoff_base", 0.2)),
            backoff_cap=float(pick("backoff_cap", 5.0)),
            circuit_threshold=int(pick("circuit_threshold", 5)),
            circuit_cooldown=float(pick("circuit_cooldown", 2.0)),
            max_bytes=int(pick("max_bytes", 5 * 1024 * 1024)),
            user_agent=str(pick("user_agent", "snapshot-crawler/1.0")),
            output_dir=str(pick("output_dir", "out")),
        )
        if not cfg.sources:
            raise ValueError("config must contain at least one source")
        if cfg.global_concurrency < 1 or any(
            s.per_source_concurrency < 1 for s in cfg.sources
        ):
            raise ValueError("concurrency values must be >= 1")
        if cfg.max_depth < 0 or cfg.max_pages < 1:
            raise ValueError("max_depth must be >= 0 and max_pages >= 1")
        return cfg
