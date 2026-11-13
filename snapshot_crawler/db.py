"""SQLite persistence: content, URL state machine, attribution and versions.

A single writer lock guards all mutations; worker threads share one connection.
The content table is keyed only by the normalized content hash (no source
column). Attribution rows are unique per (source, url, content_hash).
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS content (
    content_hash   TEXT PRIMARY KEY,
    is_text        INTEGER NOT NULL,
    encoding_used  TEXT NOT NULL,
    encoding_failed INTEGER NOT NULL,
    size_bytes     INTEGER NOT NULL,
    oversized      INTEGER NOT NULL,
    first_seen     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS url_state (
    source      TEXT NOT NULL,
    url         TEXT NOT NULL,
    depth       INTEGER NOT NULL,
    status      TEXT NOT NULL CHECK (status IN ('pending','inflight','done','failed')),
    attempts    INTEGER NOT NULL DEFAULT 0,
    error_type  TEXT,
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (source, url)
);
CREATE TABLE IF NOT EXISTS attribution (
    source        TEXT NOT NULL,
    url           TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    fetched_at    TEXT NOT NULL,
    http_status   INTEGER,
    last_modified TEXT,
    etag          TEXT,
    PRIMARY KEY (source, url, content_hash),
    FOREIGN KEY (content_hash) REFERENCES content(content_hash)
);
CREATE TABLE IF NOT EXISTS versions (
    source       TEXT NOT NULL,
    url          TEXT NOT NULL,
    version      INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    PRIMARY KEY (source, url, version),
    FOREIGN KEY (content_hash) REFERENCES content(content_hash)
);
CREATE TABLE IF NOT EXISTS current_pointer (
    source       TEXT NOT NULL,
    url          TEXT NOT NULL,
    version      INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    PRIMARY KEY (source, url)
);
CREATE TABLE IF NOT EXISTS source_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    source     TEXT NOT NULL,
    event_type TEXT NOT NULL,
    detail     TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS url_cache_headers (
    source        TEXT NOT NULL,
    url           TEXT NOT NULL,
    etag          TEXT,
    last_modified TEXT,
    updated_at    TEXT NOT NULL,
    PRIMARY KEY (source, url)
);
CREATE INDEX IF NOT EXISTS idx_attribution_hash ON attribution(content_hash);
CREATE INDEX IF NOT EXISTS idx_url_status ON url_state(source, status);
"""


def utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


class Database:
    def __init__(self, output_dir: str | Path):
        self.path = Path(output_dir) / "snapshot.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(
            str(self.path), check_same_thread=False, isolation_level=None
        )
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    # ---------- recovery ----------
    def reset_inflight(self) -> int:
        with self._lock:
            cur = self.conn.execute(
                "UPDATE url_state SET status='pending', updated_at=? "
                "WHERE status='inflight'",
                (utc_now_iso(),),
            )
            return cur.rowcount

    # ---------- URL state machine ----------
    def enqueue_url(self, source: str, url: str, depth: int) -> bool:
        """Insert a pending URL. Returns False if it was already known."""
        with self._lock:
            cur = self.conn.execute(
                "INSERT OR IGNORE INTO url_state(source,url,depth,status,"
                "attempts,updated_at) VALUES(?,?,?,'pending',0,?)",
                (source, url, depth, utc_now_iso()),
            )
            return cur.rowcount > 0

    def mark_inflight(self, source: str, url: str) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE url_state SET status='inflight', updated_at=? "
                "WHERE source=? AND url=?",
                (utc_now_iso(), source, url),
            )

    def record_attempt(self, source: str, url: str) -> None:
        """Increment the attempt counter for one actual HTTP request."""
        with self._lock:
            self.conn.execute(
                "UPDATE url_state SET attempts=attempts+1, updated_at=? "
                "WHERE source=? AND url=?",
                (utc_now_iso(), source, url),
            )

    def mark_done(self, source: str, url: str) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE url_state SET status='done', error_type=NULL, "
                "updated_at=? WHERE source=? AND url=?",
                (utc_now_iso(), source, url),
            )

    def mark_failed(self, source: str, url: str, error_type: str) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE url_state SET status='failed', error_type=?, "
                "updated_at=? WHERE source=? AND url=?",
                (error_type, utc_now_iso(), source, url),
            )

    def get_attempts(self, source: str, url: str) -> int:
        with self._lock:
            row = self.conn.execute(
                "SELECT attempts FROM url_state WHERE source=? AND url=?",
                (source, url),
            ).fetchone()
            return int(row[0]) if row else 0

    def pending_urls(self, source: str) -> list[tuple[str, int]]:
        with self._lock:
            return list(
                self.conn.execute(
                    "SELECT url, depth FROM url_state WHERE source=? AND "
                    "status='pending' ORDER BY depth, url",
                    (source,),
                )
            )

    def count_seen(self, source: str) -> int:
        with self._lock:
            return self.conn.execute(
                "SELECT COUNT(*) FROM url_state WHERE source=?", (source,)
            ).fetchone()[0]

    # ---------- content / attribution / versions ----------
    def content_exists(self, content_hash: str) -> bool:
        with self._lock:
            return self.conn.execute(
                "SELECT 1 FROM content WHERE content_hash=?", (content_hash,)
            ).fetchone() is not None

    def insert_content(
        self,
        content_hash: str,
        is_text: bool,
        encoding_used: str,
        encoding_failed: bool,
        size_bytes: int,
        oversized: bool,
    ) -> bool:
        """Insert a content row. Returns True if newly inserted (not dedup)."""
        with self._lock:
            cur = self.conn.execute(
                "INSERT OR IGNORE INTO content(content_hash,is_text,"
                "encoding_used,encoding_failed,size_bytes,oversized,first_seen)"
                " VALUES(?,?,?,?,?,?,?)",
                (
                    content_hash,
                    1 if is_text else 0,
                    encoding_used,
                    1 if encoding_failed else 0,
                    size_bytes,
                    1 if oversized else 0,
                    utc_now_iso(),
                ),
            )
            return cur.rowcount > 0

    def add_attribution(
        self,
        source: str,
        url: str,
        content_hash: str,
        http_status: int,
        last_modified: str | None,
        etag: str | None,
    ) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT OR IGNORE INTO attribution(source,url,content_hash,"
                "fetched_at,http_status,last_modified,etag) "
                "VALUES(?,?,?,?,?,?,?)",
                (
                    source,
                    url,
                    content_hash,
                    utc_now_iso(),
                    http_status,
                    last_modified,
                    etag,
                ),
            )

    def add_version_if_changed(
        self, source: str, url: str, content_hash: str
    ) -> tuple[bool, int]:
        """Append a version when the hash differs from the current pointer.

        Returns (is_new_version, version_number). Unchanged hash returns
        (False, current_version).
        """
        with self._lock:
            row = self.conn.execute(
                "SELECT version FROM current_pointer WHERE source=? AND url=?",
                (source, url),
            ).fetchone()
            now = utc_now_iso()
            if row is None:
                self.conn.execute(
                    "INSERT INTO versions(source,url,version,content_hash,"
                    "created_at) VALUES(?,?,1,?,?)",
                    (source, url, content_hash, now),
                )
                self.conn.execute(
                    "INSERT INTO current_pointer(source,url,version,"
                    "content_hash,updated_at) VALUES(?,?,1,?,?)",
                    (source, url, content_hash, now),
                )
                return True, 1
            current_version = int(row[0])
            current_hash = self.conn.execute(
                "SELECT content_hash FROM current_pointer WHERE source=? AND url=?",
                (source, url),
            ).fetchone()[0]
            if current_hash == content_hash:
                return False, current_version
            next_version = current_version + 1
            self.conn.execute(
                "INSERT INTO versions(source,url,version,content_hash,"
                "created_at) VALUES(?,?,?,?,?)",
                (source, url, next_version, content_hash, now),
            )
            self.conn.execute(
                "UPDATE current_pointer SET version=?, content_hash=?, "
                "updated_at=? WHERE source=? AND url=?",
                (next_version, content_hash, now, source, url),
            )
            return True, next_version

    def get_cache_headers(self, source: str, url: str) -> dict:
        with self._lock:
            row = self.conn.execute(
                "SELECT etag, last_modified FROM url_cache_headers "
                "WHERE source=? AND url=?",
                (source, url),
            ).fetchone()
        if row is None:
            return {}
        headers = {}
        if row[0]:
            headers["If-None-Match"] = row[0]
        if row[1]:
            headers["If-Modified-Since"] = row[1]
        return headers

    def upsert_cache_headers(
        self, source: str, url: str, etag: str | None,
        last_modified: str | None,
    ) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO url_cache_headers(source,url,etag,"
                "last_modified,updated_at) VALUES(?,?,?,?,?) "
                "ON CONFLICT(source,url) DO UPDATE SET etag=excluded.etag,"
                "last_modified=excluded.last_modified,updated_at=excluded.updated_at",
                (source, url, etag, last_modified, utc_now_iso()),
            )

    def log_event(self, source: str, event_type: str, detail: str) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO source_events(source,event_type,detail,created_at)"
                " VALUES(?,?,?,?)",
                (source, event_type, detail, utc_now_iso()),
            )

    def event_count(self, source: str | None = None, event_type: str | None = None) -> int:
        with self._lock:
            sql = "SELECT COUNT(*) FROM source_events WHERE 1=1"
            args: list = []
            if source is not None:
                sql += " AND source=?"
                args.append(source)
            if event_type is not None:
                sql += " AND event_type=?"
                args.append(event_type)
            return self.conn.execute(sql, args).fetchone()[0]
