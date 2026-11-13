import tempfile
import unittest
from pathlib import Path

from snapshot_crawler.blobstore import BlobStore
from snapshot_crawler.db import Database


class BlobAndDbTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.blobs = BlobStore(self.dir)
        self.db = Database(self.dir)

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_atomic_blob_write_and_temp_cleanup(self):
        fd, tmp_name = self.blobs.new_temp_file()
        with __import__("os").fdopen(fd, "wb") as fh:
            fh.write(b"hello")
        leftover = self.blobs.cleanup_temp()
        self.assertGreaterEqual(leftover, 1)
        self.assertFalse(Path(tmp_name).exists())

        fd, tmp_name = self.blobs.new_temp_file()
        with __import__("os").fdopen(fd, "wb") as fh:
            fh.write(b"payload")
        digest = BlobStore.hash_bytes(b"payload")
        target = self.blobs.write_atomic(digest, tmp_name)
        self.assertTrue(target.is_file())
        self.assertFalse(Path(tmp_name).exists())

    def test_inflight_reset_on_restart(self):
        self.db.enqueue_url("s1", "http://x/", 0)
        self.db.mark_inflight("s1", "http://x/")
        self.db.enqueue_url("s1", "http://x/p", 1)
        self.db.mark_done("s1", "http://x/p")
        reset = self.db.reset_inflight()
        self.assertEqual(reset, 1)
        pending = dict(self.db.pending_urls("s1"))
        self.assertIn("http://x/", pending)

    def test_content_dedup_and_attribution_uniqueness(self):
        inserted_1 = self.db.insert_content("h1", True, "utf-8", False, 10, False)
        inserted_2 = self.db.insert_content("h1", True, "utf-8", False, 10, False)
        self.assertTrue(inserted_1)
        self.assertFalse(inserted_2)
        self.db.add_attribution("a", "u1", "h1", 200, None, None)
        self.db.add_attribution("b", "u2", "h1", 200, None, None)
        # Re-adding same (source,url,hash) must not duplicate.
        self.db.add_attribution("a", "u1", "h1", 200, None, None)
        count = self.db.conn.execute(
            "SELECT COUNT(*) FROM attribution"
        ).fetchone()[0]
        self.assertEqual(count, 2)
        cols = [
            row[1]
            for row in self.db.conn.execute("PRAGMA table_info(content)")
        ]
        self.assertNotIn("source", cols)

    def test_version_pointer_updates_only_on_hash_change(self):
        self.db.insert_content("h1", True, "utf-8", False, 1, False)
        self.db.insert_content("h2", True, "utf-8", False, 1, False)
        is_new, version = self.db.add_version_if_changed("a", "u", "h1")
        self.assertTrue(is_new)
        self.assertEqual(version, 1)
        is_new, version = self.db.add_version_if_changed("a", "u", "h1")
        self.assertFalse(is_new)
        self.assertEqual(version, 1)
        is_new, version = self.db.add_version_if_changed("a", "u", "h2")
        self.assertTrue(is_new)
        self.assertEqual(version, 2)
        pointer = self.db.conn.execute(
            "SELECT version, content_hash FROM current_pointer "
            "WHERE source='a' AND url='u'"
        ).fetchone()
        self.assertEqual(tuple(pointer), (2, "h2"))


if __name__ == "__main__":
    unittest.main()
