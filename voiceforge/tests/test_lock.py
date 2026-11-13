import json
import os
import shutil
import tempfile
import unittest

from voiceforge.lock import LockUnavailableError, ProcessLock


class LockTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="vf-lock-")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_live_owner_is_rejected_and_released_on_exit(self):
        first = ProcessLock(self.root)
        first.acquire()
        self.assertTrue(os.path.exists(first.path))
        second = ProcessLock(self.root)
        with self.assertRaises(LockUnavailableError):
            second.acquire()
        first.release()
        third = ProcessLock(self.root)
        third.acquire()
        third.release()

    def test_stale_lock_of_dead_process_is_taken_over(self):
        lock = ProcessLock(self.root)
        lock.acquire()
        lock.release()
        with open(lock.path, "w", encoding="utf-8") as handle:
            json.dump({"pid": 2147483000, "host": "test",
                       "platform": "test", "program": "voiceforge"}, handle)
        takeover = ProcessLock(self.root)
        takeover.acquire()
        self.assertTrue(os.path.exists(takeover.path))
        takeover.release()
        self.assertFalse(os.path.exists(takeover.path))


if __name__ == "__main__":
    unittest.main()

