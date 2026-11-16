from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PIL import Image
from PyQt5.QtCore import QEventLoop, QTimer
from PyQt5.QtWidgets import QApplication

from paint_review.export import ExportJob
from paint_review.params import OrbitCamera, preset_params
from paint_review.renderer import PaintRenderer


class ExportJobTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _run_export(self, renderer: PaintRenderer, output_dir: Path) -> dict:
        loop = QEventLoop()
        results = {}
        job = ExportJob(
            renderer,
            preset_params("金属漆"),
            OrbitCamera(),
            output_dir,
            "金属漆",
            width=320,
            height=180,
        )
        job.completed.connect(lambda record: results.update(record))
        job.failed.connect(lambda message: results.update({"failure": message}))
        job.finished.connect(loop.quit)
        QTimer.singleShot(20000, loop.quit)
        job.start()
        loop.exec_()
        if job.isRunning():
            job.request_cancel()
            job.wait()
        self.assertNotIn("failure", results, results.get("failure"))
        self.assertFalse(job.isRunning())
        job.deleteLater()
        return results

    def test_export_writes_png_and_jsonl_with_software_versions(self) -> None:
        renderer = PaintRenderer(96, 96, 256, 128)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                output_dir = Path(temporary)
                first = self._run_export(renderer, output_dir)
                second = self._run_export(renderer, output_dir)
                first_path = Path(first["output_path"])
                second_path = Path(second["output_path"])
                self.assertTrue(first_path.exists())
                self.assertTrue(second_path.exists())
                self.assertIn(first["params_hash"][:12], first_path.name)
                self.assertTrue(np.array_equal(
                    np.asarray(Image.open(first_path).convert("RGB")),
                    np.asarray(Image.open(second_path).convert("RGB")),
                ))
                log_lines = (output_dir / "render_log.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
                self.assertEqual(len(log_lines), 2)
                first_record = json.loads(log_lines[0])
                self.assertEqual(first_record["params_hash"], first["params_hash"])
                self.assertEqual(
                    first_record["software"]["numpy"],
                    np.__version__,
                )
                self.assertIn("render_ms", first_record["stats"])
                self.assertIn("covered_pixels", first_record["stats"])
                self.assertIn("grain_generation_ms", first_record["stats"])
        finally:
            renderer.close()


if __name__ == "__main__":
    unittest.main()

