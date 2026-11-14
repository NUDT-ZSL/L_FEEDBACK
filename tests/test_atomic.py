import os
import tempfile
import unittest

from terrain_baker.atomicio import AtomicStage
from terrain_baker.errors import IOFailure


class AtomicStageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.target = os.path.join(self.tmp.name, "out")
        os.makedirs(self.target, exist_ok=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_commit_replaces_files(self):
        with AtomicStage(self.target) as stage:
            p = stage.path("a.txt")
            with open(p, "w") as handle:
                handle.write("new")
            stage.register("a.txt")
            stage.commit()
        self.assertEqual(open(os.path.join(self.target, "a.txt")).read(), "new")
        self.assertFalse(os.path.exists(self.target + ".staging"))

    def test_failure_cleans_staging_and_preserves_target(self):
        good = os.path.join(self.target, "keep.txt")
        with open(good, "w") as handle:
            handle.write("original")

        def fail_on_commit(relative):
            # Fail before the first replacement so nothing reaches the target.
            raise OSError("simulated write failure")

        with self.assertRaises(IOFailure):
            with AtomicStage(self.target, inject_failure=fail_on_commit) as stage:
                for name in ("safe.txt", "boom.txt"):
                    with open(stage.path(name), "w") as handle:
                        handle.write("x")
                    stage.register(name)
                stage.commit()
        with open(good) as handle:
            self.assertEqual(handle.read(), "original")
        self.assertFalse(os.path.exists(os.path.join(self.target, "safe.txt")))
        self.assertFalse(os.path.exists(self.target + ".staging"))

    def test_stale_manifest_files_removed(self):
        stale = os.path.join(self.target, "old.bin")
        with open(stale, "wb") as handle:
            handle.write(b"old")
        with AtomicStage(self.target) as stage:
            with open(stage.path("new.bin"), "wb") as handle:
                handle.write(b"new")
            stage.register("new.bin")
            stage.commit(previous_manifest={"old.bin"})
        self.assertFalse(os.path.exists(stale))
        self.assertTrue(os.path.exists(os.path.join(self.target, "new.bin")))

    def test_path_escape_rejected(self):
        with AtomicStage(self.target) as stage:
            with self.assertRaises(IOFailure):
                stage.path("../escape.txt")


if __name__ == "__main__":
    unittest.main()
