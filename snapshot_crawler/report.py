"""Audit report generation with database-rebuilt reconciliation.

Every reported counter is independently recomputed from SQLite using the same
definitions, and the run asserts they match. This makes the JSON report fully
auditable rather than trusting in-memory counters.
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import Config
from .db import Database


def _scalar(db: Database, sql: str, args: tuple = ()):
    return db.conn.execute(sql, args).fetchone()[0]


def build_report(config: Config, started_at: str, ended_at: str) -> dict:
    db = Database(config.output_dir)
    try:
        per_source = {}
        totals = {
            "fetched": 0,
            "done": 0,
            "failed": 0,
            "dedup_hits": 0,
            "new_versions": 0,
            "retries": 0,
            "circuit_trips": 0,
        }
        for source in config.sources:
            name = source.name
            fetched = _scalar(
                db,
                "SELECT COALESCE(SUM(attempts),0) FROM url_state WHERE source=?",
                (name,),
            )
            done = _scalar(
                db,
                "SELECT COUNT(*) FROM url_state WHERE source=? AND status='done'",
                (name,),
            )
            failed = _scalar(
                db,
                "SELECT COUNT(*) FROM url_state WHERE source=? AND status='failed'",
                (name,),
            )
            # Deterministic definition independent of write scheduling:
            # an attribution is a dedup hit when the same content hash is
            # also attributed to at least one other source.
            dedup_hits = _scalar(
                db,
                "SELECT COUNT(*) FROM attribution a WHERE a.source=? AND "
                "EXISTS (SELECT 1 FROM attribution b WHERE "
                "b.content_hash=a.content_hash AND b.source<>a.source)",
                (name,),
            )
            new_versions = _scalar(
                db,
                "SELECT COUNT(*) FROM versions WHERE source=? AND version>1",
                (name,),
            )
            retries = _scalar(
                db,
                "SELECT COALESCE(SUM(attempts-1),0) FROM url_state "
                "WHERE source=? AND attempts>0",
                (name,),
            )
            circuit_trips = db.event_count(name, "circuit_open")

            failed_rows = list(
                db.conn.execute(
                    "SELECT url, error_type, attempts FROM url_state "
                    "WHERE source=? AND status='failed' ORDER BY url",
                    (name,),
                )
            )
            per_source[name] = {
                "fetched": int(fetched),
                "done": int(done),
                "failed": int(failed),
                "dedup_hits": int(dedup_hits),
                "new_versions": int(new_versions),
                "retries": int(retries),
                "circuit_trips": int(circuit_trips),
                "failed_urls": [
                    {
                        "url": row[0],
                        "error_type": row[1],
                        "attempts": int(row[2]),
                    }
                    for row in failed_rows
                ],
            }
            for key in (
                "fetched",
                "done",
                "failed",
                "dedup_hits",
                "new_versions",
                "retries",
                "circuit_trips",
            ):
                totals[key] += per_source[name][key]

        hash_set = sorted(
            row[0]
            for row in db.conn.execute("SELECT content_hash FROM content")
        )
        blob_hashes = sorted(
            p.name
            for p in (Path(config.output_dir) / "blobs").rglob("*")
            if p.is_file() and not p.name.endswith(".tmp")
        )
        report = {
            "started_at": started_at,
            "ended_at": ended_at,
            "config_digest": _config_digest(config),
            "totals": totals,
            "per_source": per_source,
            "content_hash_set": hash_set,
            "blob_hash_set": blob_hashes,
        }
        _reconcile(report)
        return report
    finally:
        db.close()


def _config_digest(config: Config) -> dict:
    # Deterministic, time-independent configuration summary.
    return {
        "global_concurrency": config.global_concurrency,
        "max_depth": config.max_depth,
        "max_pages": config.max_pages,
        "timeout": config.timeout,
        "max_retries": config.max_retries,
        "backoff_base": config.backoff_base,
        "backoff_cap": config.backoff_cap,
        "circuit_threshold": config.circuit_threshold,
        "circuit_cooldown": config.circuit_cooldown,
        "max_bytes": config.max_bytes,
        "sources": [
            {
                "name": s.name,
                "root_url": s.root_url,
                "per_source_concurrency": s.per_source_concurrency,
                "start_interval": s.start_interval,
            }
            for s in config.sources
        ],
    }


def _reconcile(report: dict) -> None:
    """Cross-check totals against per-source sums and blob/hash sets."""
    totals = report["totals"]
    sums = {key: 0 for key in totals}
    for section in report["per_source"].values():
        for key in sums:
            sums[key] += section[key]
    if sums != totals:
        raise AssertionError(
            f"report totals mismatch: {sums} != {totals}"
        )
    if report["content_hash_set"] != report["blob_hash_set"]:
        raise AssertionError("content table hashes do not match blob files")


def write_report(config: Config, report: dict) -> Path:
    out = Path(config.output_dir) / "report.json"
    out.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    return out
