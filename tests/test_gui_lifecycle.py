from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from paint_review.params import preset_params
from paint_review.ui import PaintReviewWindow


def wait_until(app: QApplication, predicate, timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        QTest.qWait(50)
    return bool(predicate())


class GuiLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temporary.name)
        self.window = PaintReviewWindow()
        self.window.output_dir = self.output_dir
        self.window.show()
        QTest.qWait(100)

    def tearDown(self) -> None:
        if self.window.isVisible():
            self.window.close()
            wait_until(self.app, lambda: not self.window.isVisible(), 20.0)
        QTest.qWait(50)
        self.temporary.cleanup()

    def test_preset_switch_cancels_export_then_rebuilds(self) -> None:
        self.window._start_export()
        self.assertTrue(
            wait_until(
                self.app,
                lambda: self.window._export_thread is not None
                and self.window._export_thread.isRunning(),
                5.0,
            )
        )
        self.window.panel.preset_combo.setCurrentText("珠光漆")
        self.assertEqual(self.window._pending_preset, "珠光漆")
        self.assertTrue(
            wait_until(
                self.app,
                lambda: self.window._export_thread is None
                and self.window.preset_name == "珠光漆"
                and self.window.preview_worker is not None,
                25.0,
            )
        )
        self.assertEqual(
            self.window.params.material_hash(),
            preset_params("珠光漆").material_hash(),
        )
        self.assertTrue(self.window.preview_worker.isRunning())
        self.assertEqual(list(self.output_dir.glob("*.png")), [])
        self.assertFalse((self.output_dir / "render_log.jsonl").exists())

    def test_close_cancels_export_and_leaves_no_threads(self) -> None:
        self.window._start_export()
        self.assertTrue(
            wait_until(
                self.app,
                lambda: self.window._export_thread is not None
                and self.window._export_thread.isRunning(),
                5.0,
            )
        )
        self.window.close()
        self.assertTrue(self.window.isVisible())
        self.assertTrue(
            wait_until(self.app, lambda: not self.window.isVisible(), 25.0)
        )
        self.assertIsNone(self.window._export_thread)
        self.assertIsNone(self.window.preview_worker)
        self.assertTrue(self.window.renderer.closed)
        self.assertEqual(list(self.output_dir.glob("*.png")), [])


if __name__ == "__main__":
    unittest.main()

