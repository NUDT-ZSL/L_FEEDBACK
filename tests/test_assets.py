import os
import unittest

from audiopack.assets import AssetError, AssetStore, FailureInjector
from audiopack.config import Config
from tests.helpers import make_workspace, cleanup


class AssetLoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root, self.assets, _, _, _, self.config = make_workspace()
        self.config = Config(max_workers=3, max_retries=2)

    def tearDown(self) -> None:
        cleanup(self.root)

    def _path(self, voice_key: str, lang: str) -> str:
        return os.path.abspath(
            AssetStore(self.config).asset_path(self.assets, voice_key, lang)
        )

    def test_parallel_load_succeeds_and_caches(self) -> None:
        paths = [self._path("kill", lang) for lang in ("zh", "en", "ja")]
        store = AssetStore(self.config)
        clips, failures = store.load_many(paths)
        self.assertEqual(failures, [])
        self.assertEqual(set(clips), set(paths))
        for clip in clips.values():
            self.assertEqual(clip.sample_rate, 48000)
            self.assertEqual(len(clip.content_hash), 64)
        # Cache returns the same instance on repeated load.
        self.assertIs(store.load_with_retry(paths[0]), clips[paths[0]])

    def test_transient_failure_retries_then_succeeds(self) -> None:
        path = self._path("kill", "zh")
        injector = FailureInjector(transient={path}, transient_attempts=2)
        store = AssetStore(self.config, injector)
        clip = store.load_with_retry(path)
        self.assertGreater(clip.frames, 0)
        self.assertEqual(injector.attempts[path], 3)

    def test_transient_failure_exhausting_retries_fails_with_code(self) -> None:
        path = self._path("kill", "en")
        injector = FailureInjector(transient={path}, transient_attempts=99)
        store = AssetStore(self.config, injector)
        with self.assertRaises(AssetError) as ctx:
            store.load_with_retry(path)
        self.assertEqual(ctx.exception.code, "E_LOAD_FAILED")
        self.assertTrue(ctx.exception.retryable)
        self.assertEqual(injector.attempts[path], self.config.max_retries + 1)

    def test_missing_asset_is_terminal(self) -> None:
        path = self._path("victory", "ja")
        injector = FailureInjector(missing={path})
        store = AssetStore(self.config, injector)
        with self.assertRaises(AssetError) as ctx:
            store.load_with_retry(path)
        self.assertEqual(ctx.exception.code, "E_MISSING_ASSET")
        self.assertFalse(ctx.exception.retryable)

    def test_corrupt_asset_is_terminal(self) -> None:
        path = self._path("defend", "zh")
        injector = FailureInjector(corrupt={path})
        store = AssetStore(self.config, injector)
        with self.assertRaises(AssetError) as ctx:
            store.load_with_retry(path)
        self.assertEqual(ctx.exception.code, "E_CORRUPT_ASSET")

    def test_physically_corrupt_wav_raises_corrupt_code(self) -> None:
        path = self._path("ult", "en")
        with open(path, "wb") as handle:
            handle.write(b"not-a-wav-file")
        store = AssetStore(self.config)
        with self.assertRaises(AssetError) as ctx:
            store.load_with_retry(path)
        self.assertEqual(ctx.exception.code, "E_CORRUPT_ASSET")

    def test_load_many_collects_all_failures(self) -> None:
        missing = self._path("victory", "ja")
        corrupt = self._path("ult", "en")
        good = self._path("kill", "zh")
        injector = FailureInjector(missing={missing}, corrupt={corrupt})
        store = AssetStore(self.config, injector)
        clips, failures = store.load_many([missing, corrupt, good])
        codes = sorted(error.code for error in failures)
        self.assertEqual(codes, ["E_CORRUPT_ASSET", "E_MISSING_ASSET"])
        self.assertEqual(list(clips), [good])


if __name__ == "__main__":
    unittest.main()
