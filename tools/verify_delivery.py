from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from PyQt5.QtCore import QCoreApplication, QEventLoop, QTimer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paint_review.export import ExportJob  # noqa: E402
from paint_review.params import OrbitCamera, preset_params  # noqa: E402
from paint_review.renderer import PaintRenderer  # noqa: E402


def run_full_export(renderer: PaintRenderer, preset: str, output_dir: Path) -> dict:
    loop = QEventLoop()
    result = {}
    job = ExportJob(
        renderer,
        preset_params(preset),
        OrbitCamera(),
        output_dir,
        preset,
        width=1920,
        height=1080,
    )
    job.completed.connect(lambda record: result.update(record))
    job.failed.connect(lambda message: result.update({"failure": message}))
    job.finished.connect(loop.quit)
    QTimer.singleShot(120000, loop.quit)
    job.start()
    loop.exec_()
    if job.isRunning():
        job.request_cancel()
        job.wait()
    if "failure" in result:
        raise RuntimeError(result["failure"])
    if not result:
        raise RuntimeError("Export job produced no result")
    return result


def pixel_digest(path: Path) -> str:
    pixels = np.ascontiguousarray(np.asarray(Image.open(path).convert("RGB")))
    return hashlib.sha256(pixels.tobytes()).hexdigest()


def main() -> int:
    app = QCoreApplication.instance() or QCoreApplication([])
    output_dir = ROOT / "artifacts" / "verification"
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("paint_review_*.png"):
        stale.unlink()
    log_path = output_dir / "render_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    first_renderer = PaintRenderer(96, 96, 512, 256)
    second_renderer = PaintRenderer(96, 96, 512, 256)
    records = []
    try:
        for preset in ("素色漆", "金属漆", "珠光漆"):
            record = run_full_export(first_renderer, preset, output_dir)
            record["pixel_sha256"] = pixel_digest(Path(record["output_path"]))
            records.append(record)
            print(
                "{}: {:.0f} ms, {}".format(
                    preset, record["stats"]["render_ms"], Path(record["output_path"]).name
                )
            )
        repeat = run_full_export(second_renderer, "金属漆", output_dir)
        repeat["pixel_sha256"] = pixel_digest(Path(repeat["output_path"]))
        first_metal = next(r for r in records if r["preset"] == "金属漆")
        identical = first_metal["pixel_sha256"] == repeat["pixel_sha256"]
        if not identical:
            raise AssertionError("Independent 1920x1080 exports differ by pixel")
        summary = {
            "numpy": np.__version__,
            "resolution": [1920, 1080],
            "preset_records": records,
            "repeat_record": repeat,
            "independent_repeat_pixel_identical": identical,
        }
        summary_path = ROOT / "artifacts" / "verification_summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print("Independent repeat pixel-identical:", identical)
        print("Summary:", summary_path)
        print("Log:", log_path)
        return 0
    finally:
        first_renderer.close()
        second_renderer.close()


if __name__ == "__main__":
    sys.exit(main())

