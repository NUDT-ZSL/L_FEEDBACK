from __future__ import annotations

import os
import threading

from replay_exporter import ErrorCode, ExporterError, export
from replay_exporter.locks import PLAN_CACHE, acquire_export_lock

from .base import ExporterTestCase


class ConcurrencyTests(ExporterTestCase):
    def test_duplicate_export_to_same_directory_is_rejected(self):
        output_dir = os.path.join(self.temp_root, "busy")
        os.makedirs(output_dir, exist_ok=True)
        lock = acquire_export_lock(output_dir)
        try:
            with self.assertRaises(ExporterError) as caught:
                export(self.timeline_path, output_dir)
            self.assertEqual(caught.exception.code, ErrorCode.OUTPUT_DIR_BUSY)
        finally:
            lock.release()

    def test_concurrent_exports_to_different_directories(self):
        results = {}
        errors = []

        def worker(name: str) -> None:
            target = os.path.join(self.temp_root, name)
            try:
                results[name] = export(self.timeline_path, target)
            except ExporterError as exc:  # pragma: no cover - failure path
                errors.append(exc.to_dict())

        threads = [
            threading.Thread(target=worker, args=(f"worker_{index}",))
            for index in range(3)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        self.assertEqual(len(results), 3)
        for name, result in results.items():
            self.assertTrue(result["ok"], msg=name)
            self.assertFalse(os.path.exists(
                os.path.join(self.temp_root, name, "export.lock")
            ))

    def test_plan_cache_is_shared_and_locked(self):
        from replay_exporter.engine import parse_and_plan

        plan_a, _ = parse_and_plan(self.timeline_path)
        plan_b, _ = parse_and_plan(self.timeline_path)
        self.assertIs(plan_a, plan_b)
        self.assertGreaterEqual(PLAN_CACHE.size(), 1)

    def test_stale_lock_from_dead_pid_is_taken_over(self):
        import json

        output_dir = os.path.join(self.temp_root, "stale")
        os.makedirs(output_dir, exist_ok=True)
        stale = {
            "token": "dead-token",
            "pid": 2147480000,  # virtually guaranteed not to exist
            "host": "test-host",
            "heartbeat_at": "2000-01-01T00:00:00+00:00",
        }
        with open(os.path.join(output_dir, "export.lock"), "w", encoding="utf-8") as handle:
            json.dump(stale, handle)
        lock = acquire_export_lock(output_dir)
        self.assertNotEqual(lock.token, "dead-token")
        lock.release()
