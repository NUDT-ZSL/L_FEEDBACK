import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

from snapshot_crawler.config import Config
from snapshot_crawler.crawler import Crawler
from snapshot_crawler.report import build_report

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_server import start_pair  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def write_config(path: Path, servers, output_dir: Path, **overrides):
    data = {
        "global_concurrency": overrides.get("global_concurrency", 4),
        "per_source_concurrency": overrides.get("per_source_concurrency", 2),
        "start_interval": overrides.get("start_interval", 0.05),
        "max_depth": overrides.get("max_depth", 3),
        "max_pages": overrides.get("max_pages", 60),
        "timeout": overrides.get("timeout", 4.0),
        "max_retries": overrides.get("max_retries", 3),
        "backoff_base": overrides.get("backoff_base", 0.05),
        "backoff_cap": overrides.get("backoff_cap", 0.5),
        "circuit_threshold": overrides.get("circuit_threshold", 5),
        "circuit_cooldown": overrides.get("circuit_cooldown", 1.0),
        "max_bytes": overrides.get("max_bytes", 256 * 1024),
        "output_dir": str(output_dir),
        "sources": [
            {"name": "alpha", "root_url": servers[0].root_url},
            {"name": "beta", "root_url": servers[1].root_url},
        ],
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


class IntegrationBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state_dir = Path(tempfile.mkdtemp())
        cls.token = f"t{os.getpid()}-{cls.__name__}"
        cls.servers, cls.request_log = start_pair(
            cls.state_dir, cls.token, state_key=cls.__name__
        )

    @classmethod
    def tearDownClass(cls):
        for server in cls.servers:
            server.stop()
        shutil.rmtree(cls.state_dir, ignore_errors=True)

    def make_run_dir(self) -> Path:
        return Path(tempfile.mkdtemp(dir=self.state_dir))

    def reset_server_state(self) -> None:
        for flag in self.state_dir.glob("*.count"):
            flag.unlink(missing_ok=True)

    def run_crawler(self, out_dir: Path, **overrides):
        cfg_path = out_dir / "config.json"
        write_config(cfg_path, self.servers, out_dir, **overrides)
        config = Config.load(cfg_path)
        crawler = Crawler(config)
        started = datetime.now(timezone.utc).isoformat()
        crawler.run()
        ended = datetime.now(timezone.utc).isoformat()
        crawler.db.close()
        return config, build_report(config, started, ended)

    @staticmethod
    def _url_rows(out_dir: Path) -> int:
        conn = sqlite3.connect(out_dir / "snapshot.db")
        try:
            return conn.execute("SELECT COUNT(*) FROM url_state").fetchone()[0]
        finally:
            conn.close()


class FullPipelineTests(IntegrationBase):
    def test_01_full_run_and_report_reconciliation(self):
        out_dir = self.make_run_dir()
        config, report = self.run_crawler(out_dir)
        self.assertEqual(report["totals"]["done"] + report["totals"]["failed"],
                         self._url_rows(out_dir))
        # Every failed URL carries a final error type.
        for section in report["per_source"].values():
            for item in section["failed_urls"]:
                self.assertTrue(item["error_type"])
        # Hash sets equal between content table and blob files.
        self.assertEqual(
            report["content_hash_set"], report["blob_hash_set"]
        )
        self.assertGreater(len(report["content_hash_set"]), 3)

    def test_02_cross_source_dedup(self):
        out_dir = self.make_run_dir()
        _config, report = self.run_crawler(out_dir)
        # Both origins host /shared with identical normalized content.
        conn = sqlite3.connect(out_dir / "snapshot.db")
        rows = conn.execute(
            "SELECT source, content_hash FROM attribution WHERE url LIKE "
            "'%/shared'"
        ).fetchall()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][1], rows[1][1])
        content_cols = [
            r[1] for r in conn.execute("PRAGMA table_info(content)")
        ]
        self.assertNotIn("source", content_cols)
        # The shared hash has attributions from two distinct sources.
        cross_source = conn.execute(
            "SELECT COUNT(DISTINCT source) FROM attribution "
            "WHERE content_hash=?", (rows[0][1],)
        ).fetchone()[0]
        self.assertEqual(cross_source, 2)
        conn.close()

    def test_03_same_origin_and_depth_limits(self):
        out_dir = self.make_run_dir()
        self.run_crawler(out_dir, max_depth=1)
        conn = sqlite3.connect(out_dir / "snapshot.db")
        urls = [r[0] for r in conn.execute("SELECT url FROM url_state")]
        # External domain never queued.
        self.assertFalse(any("example.invalid" in u for u in urls))
        # Depth-2 pages not reached with max_depth=1.
        self.assertFalse(any(u.endswith("/deep/2") for u in urls))
        # Query retained, fragment dropped.
        self.assertTrue(any("/query?a=1&b=2" in u for u in urls))
        self.assertFalse(any("#" in u for u in urls))
        conn.close()

    def test_04_oversized_streaming_and_flag(self):
        out_dir = self.make_run_dir()
        max_bytes = 128 * 1024
        self.run_crawler(out_dir, max_bytes=max_bytes)
        conn = sqlite3.connect(out_dir / "snapshot.db")
        row = conn.execute(
            "SELECT size_bytes, oversized FROM content WHERE content_hash IN "
            "(SELECT content_hash FROM attribution WHERE url LIKE '%/oversized')"
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], max_bytes)
        self.assertEqual(row[1], 1)
        hash_value = conn.execute(
            "SELECT content_hash FROM attribution WHERE url LIKE '%/oversized'"
        ).fetchone()[0]
        on_disk = out_dir / "blobs" / hash_value[:2] / hash_value[2:4] / hash_value
        self.assertEqual(on_disk.stat().st_size, max_bytes)
        conn.close()

    def test_05_retry_then_success_and_4xx_no_retry(self):
        out_dir = self.make_run_dir()
        self.reset_server_state()
        _config, report = self.run_crawler(out_dir)
        conn = sqlite3.connect(out_dir / "snapshot.db")
        flaky = conn.execute(
            "SELECT status, attempts FROM url_state WHERE url LIKE '%/flaky'"
        ).fetchone()
        self.assertEqual(flaky[0], "done")
        self.assertGreaterEqual(flaky[1], 2)
        missing = conn.execute(
            "SELECT status, attempts, error_type FROM url_state "
            "WHERE url LIKE '%/missing'"
        ).fetchone()
        self.assertEqual(missing[0], "failed")
        self.assertEqual(missing[1], 1)  # 404 consumes no retry budget
        self.assertEqual(missing[2], "http_404")
        conn.close()
        # Retries actually counted in report.
        self.assertGreaterEqual(report["totals"]["retries"], 1)

    def test_06_circuit_breaker_trips_and_recovers(self):
        out_dir = self.make_run_dir()
        self.reset_server_state()
        # threshold=2: /always500 exhausts retries (failures count separately
        # on different URLs), then /reset trips the breaker.
        _config, report = self.run_crawler(
            out_dir,
            circuit_threshold=2,
            circuit_cooldown=0.8,
            max_retries=1,
        )
        conn = sqlite3.connect(out_dir / "snapshot.db")
        trips = conn.execute(
            "SELECT COUNT(*) FROM source_events WHERE event_type='circuit_open'"
        ).fetchone()[0]
        self.assertGreaterEqual(trips, 1)
        recovered = conn.execute(
            "SELECT COUNT(*) FROM source_events "
            "WHERE event_type='circuit_recovered'"
        ).fetchone()[0]
        self.assertGreaterEqual(recovered, 1)
        # The breaker never blocks the whole source: normal pages still done.
        done_pages = conn.execute(
            "SELECT COUNT(*) FROM url_state WHERE status='done'"
        ).fetchone()[0]
        self.assertGreater(done_pages, 5)
        self.assertGreaterEqual(report["totals"]["circuit_trips"], 1)
        conn.close()

    def test_07_304_creates_no_version_or_content(self):
        out_dir = self.make_run_dir()
        config, _report = self.run_crawler(out_dir)
        etag_url = self.servers[0].root_url + "etag"
        from snapshot_crawler.blobstore import BlobStore
        from snapshot_crawler.fetcher import fetch_url

        result = fetch_url(
            etag_url,
            self.servers[0].root_url,
            BlobStore(out_dir),
            config.timeout,
            config.max_bytes,
            config.user_agent,
            conditional_headers={"If-None-Match": '"v1-fixed-etag"'},
        )
        self.assertEqual(result.http_status, 304)
        self.assertTrue(result.not_modified)
        self.assertIsNone(result.content_hash)
        conn = sqlite3.connect(out_dir / "snapshot.db")
        before = conn.execute(
            "SELECT COUNT(*) FROM versions WHERE url=?", (etag_url,)
        ).fetchone()[0]
        self.assertEqual(before, 1)  # initial fetch created exactly one version
        # The conditional 304 request must not add a version or content row.
        after = conn.execute(
            "SELECT COUNT(*) FROM versions WHERE url=?", (etag_url,)
        ).fetchone()[0]
        self.assertEqual(after, before)
        not_modified_events = conn.execute(
            "SELECT COUNT(*) FROM source_events WHERE event_type='not_modified'"
        ).fetchone()[0]
        self.assertEqual(not_modified_events, 0)  # raw fetch does not write DB
        versions = conn.execute(
            "SELECT COUNT(*) FROM versions WHERE url=?"
            , (etag_url,),
        ).fetchone()[0]
        self.assertEqual(versions, 1)
        conn.close()

    def test_08_deterministic_two_runs(self):
        first = self.make_run_dir()
        second = self.make_run_dir()
        self.reset_server_state()
        _cfg1, report1 = self.run_crawler(first)
        self.reset_server_state()
        _cfg2, report2 = self.run_crawler(second)
        for key in ("started_at", "ended_at"):
            report1.pop(key)
            report2.pop(key)
        self.assertEqual(report1, report2)


class VersioningTests(IntegrationBase):
    def test_hash_change_creates_version_and_keeps_old_blob(self):
        from snapshot_crawler.blobstore import BlobStore
        from snapshot_crawler.fetcher import fetch_url

        out_dir = self.make_run_dir()
        self.reset_server_state()
        write_config(out_dir / "config.json", self.servers, out_dir)
        config = Config.load(out_dir / "config.json")
        db = __import__("snapshot_crawler.db", fromlist=["Database"]).Database(
            out_dir
        )
        blobs = BlobStore(out_dir)
        url = self.servers[0].root_url + "versioned"
        hashes = []
        for _ in range(3):
            result = fetch_url(
                url,
                self.servers[0].root_url,
                blobs,
                config.timeout,
                config.max_bytes,
                config.user_agent,
            )
            self.assertEqual(result.http_status, 200)
            blobs.write_atomic(result.content_hash, result.temp_path)
            db.insert_content(
                result.content_hash,
                result.is_text,
                result.encoding_used,
                result.encoding_failed,
                result.size_bytes,
                result.oversized,
            )
            db.add_attribution(
                "alpha", url, result.content_hash, 200,
                result.last_modified, result.etag,
            )
            is_new, version = db.add_version_if_changed(
                "alpha", url, result.content_hash
            )
            hashes.append(result.content_hash)
            if version == 1:
                self.assertTrue(is_new)
        db.close()

        self.assertEqual(len(set(hashes)), 3)
        conn = sqlite3.connect(out_dir / "snapshot.db")
        versions = conn.execute(
            "SELECT version, content_hash FROM versions "
            "WHERE url=? ORDER BY version", (url,)
        ).fetchall()
        self.assertEqual([v[0] for v in versions], [1, 2, 3])
        pointer = conn.execute(
            "SELECT version, content_hash FROM current_pointer WHERE url=?",
            (url,),
        ).fetchone()
        self.assertEqual(pointer[0], 3)
        self.assertEqual(pointer[1], hashes[-1])
        conn.close()
        # Old blobs remain on disk even though only the latest is referenced.
        for digest in hashes:
            self.assertTrue(blobs.path_for(digest).is_file())


class NotModifiedCrawlerTests(IntegrationBase):
    def test_worker_304_keeps_single_version(self):
        from snapshot_crawler.crawler import SourceState

        out_dir = self.make_run_dir()
        config, _report = self.run_crawler(out_dir)
        url = self.servers[0].root_url + "etag"
        crawler = Crawler(config)
        source_cfg = next(s for s in config.sources if s.name == "alpha")
        state = SourceState(cfg=source_cfg)
        # Re-drive the worker for an already done URL; the cache header table
        # now holds the ETag, so the server answers 304.
        crawler.db.enqueue_url("alpha", url, 0)
        crawler.db.mark_inflight("alpha", url)
        crawler._process_with_retry(state, url, 0)
        crawler.db.close()

        conn = sqlite3.connect(out_dir / "snapshot.db")
        status = conn.execute(
            "SELECT status, attempts FROM url_state WHERE url=?", (url,)
        ).fetchone()
        self.assertEqual(status[0], "done")
        versions = conn.execute(
            "SELECT COUNT(*) FROM versions WHERE url=?", (url,)
        ).fetchone()[0]
        self.assertEqual(versions, 1)
        events = conn.execute(
            "SELECT COUNT(*) FROM source_events "
            "WHERE event_type='not_modified' AND source='alpha'"
        ).fetchone()[0]
        self.assertEqual(events, 1)
        conn.close()


class RateLimitTests(IntegrationBase):
    def test_per_source_interval_and_inflight_caps(self):
        out_dir = self.make_run_dir()
        interval = 0.25
        config, _report = self.run_crawler(
            out_dir,
            start_interval=interval,
            per_source_concurrency=2,
            global_concurrency=3,
            backoff_base=0.02,
        )
        # 1) Strict per-source start interval using monotonic arrival stamps.
        for label in ("alpha", "beta"):
            mono = sorted(
                e["mono"]
                for e in self.request_log.entries
                if e["origin"] == label
            )
            gaps = [b - a for a, b in zip(mono, mono[1:])]
            self.assertTrue(gaps)
            # Small scheduling slack only; nothing close to a missing interval.
            self.assertGreaterEqual(min(gaps), interval - 0.05)

        # 2) Reconstruct peak overlap from arrival/completion timestamps.
        finished = [
            e for e in self.request_log.entries if e["completed_mono"]
        ]

        def peak_overlap(entries):
            events = []
            for entry in entries:
                events.append((entry["mono"], 1))
                events.append((entry["completed_mono"], -1))
            events.sort(key=lambda item: (item[0], item[1]))
            current = peak = 0
            for _t, delta in events:
                current += delta
                peak = max(peak, current)
            return peak

        self.assertLessEqual(peak_overlap(finished), config.global_concurrency)
        for label in ("alpha", "beta"):
            source_entries = [e for e in finished if e["origin"] == label]
            self.assertLessEqual(
                peak_overlap(source_entries), 2
            )


class RecoveryTests(IntegrationBase):
    def test_hard_exit_then_resume_and_temp_cleanup(self):
        out_dir = self.make_run_dir()
        cfg_path = out_dir / "config.json"
        write_config(
            cfg_path, self.servers, out_dir, start_interval=0.3,
            per_source_concurrency=1, global_concurrency=2,
        )
        env = dict(os.environ)
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "snapshot_crawler",
                "--config",
                str(cfg_path),
                "--exit-after",
                "2",
            ],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("exit-after", proc.stdout)
        conn = sqlite3.connect(out_dir / "snapshot.db")
        inflight = conn.execute(
            "SELECT COUNT(*) FROM url_state WHERE status='inflight'"
        ).fetchone()[0]
        self.assertGreaterEqual(inflight, 1)
        conn.close()

        # Drop an orphan temp file to prove startup cleanup.
        blob_dir = out_dir / "blobs"
        orphan = blob_dir / ".orphan.tmp"
        orphan.write_bytes(b"stale")

        config = Config.load(cfg_path)
        crawler = Crawler(config)
        removed, reset = crawler.startup()
        self.assertGreaterEqual(removed, 1)
        self.assertGreaterEqual(reset, 1)
        self.assertFalse(orphan.exists())
        crawler.run()
        crawler.db.close()

        conn = sqlite3.connect(out_dir / "snapshot.db")
        leftovers = conn.execute(
            "SELECT COUNT(*) FROM url_state WHERE status IN "
            "('inflight','pending')"
        ).fetchone()[0]
        self.assertEqual(leftovers, 0)
        done_or_failed = conn.execute(
            "SELECT COUNT(*) FROM url_state WHERE status IN ('done','failed')"
        ).fetchone()[0]
        self.assertGreater(done_or_failed, 0)
        conn.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
