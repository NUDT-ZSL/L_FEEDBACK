import json
import os
import subprocess
import sys
import tracemalloc
import unittest

from rhythm_mixer.config import load_inputs
from rhythm_mixer.engine import run

from tests._helpers import make_fixture, read_output

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class DeterminismTests(unittest.TestCase):
    def test_repeated_runs_byte_identical(self):
        input_dir, output_a = make_fixture()
        output_b = os.path.join(os.path.dirname(output_a), "output_b")
        run(input_dir, output_a)
        run(input_dir, output_b)
        for name in ("mix_output.wav", "beat_alignment_report.json"):
            with open(os.path.join(output_a, name), "rb") as handle:
                first = handle.read()
            with open(os.path.join(output_b, name), "rb") as handle:
                second = handle.read()
            self.assertEqual(first, second, name)

    def test_report_contains_no_absolute_paths(self):
        input_dir, output_dir = make_fixture()
        run(input_dir, output_dir)
        report_path = os.path.join(output_dir, "beat_alignment_report.json")
        with open(report_path, encoding="utf-8") as handle:
            raw = handle.read()
        self.assertNotIn(input_dir, raw)
        self.assertNotIn(output_dir, raw)

    def test_synth_fixture_is_seed_deterministic(self):
        from rhythm_mixer.make_samples import generate
        import tempfile

        dir_a = tempfile.mkdtemp(prefix="synth_a_")
        dir_b = tempfile.mkdtemp(prefix="synth_b_")
        generate(dir_a)
        generate(dir_b)
        for name in ("drums.wav", "bass.wav", "melody.wav",
                     "chart.json", "events.json"):
            with open(os.path.join(dir_a, name), "rb") as handle:
                a = handle.read()
            with open(os.path.join(dir_b, name), "rb") as handle:
                b = handle.read()
            self.assertEqual(a, b, name)


class ResourceTests(unittest.TestCase):
    def _long_chart_fixture(self, duration_s=120.0):
        input_dir, output_dir = make_fixture()
        chart_path = os.path.join(input_dir, "chart.json")
        with open(chart_path, encoding="utf-8") as handle:
            chart = json.load(handle)
        chart["total_duration_ms"] = round(duration_s * 1000.0, 3)
        with open(chart_path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(chart, handle)

        chart_loaded = load_inputs(input_dir)
        input_bytes = sum(
            layer.samples.nbytes for layer in chart_loaded.layers.values()
        )
        return input_dir, output_dir, input_bytes, duration_s

    def test_peak_mixing_memory_within_five_inputs(self):
        input_dir, output_dir, input_bytes, duration_s = \
            self._long_chart_fixture()

        tracemalloc.start()
        baseline = tracemalloc.take_snapshot()
        run(input_dir, output_dir)
        peak = tracemalloc.get_traced_memory()[1]
        end = tracemalloc.take_snapshot()
        tracemalloc.stop()

        # Growth attributable to the mixing run, excluding fixture allocations
        # already live before the snapshot comparison.
        stats = end.compare_to(baseline, "lineno")
        growth = sum(stat.size_diff for stat in stats if stat.size_diff > 0)
        limit = input_bytes * 5
        self.assertLess(
            growth,
            limit,
            f"mixing growth {growth} B exceeds 5x input {limit} B for a "
            f"{duration_s:g} s output",
        )
        self.assertLess(peak, input_bytes * 20)  # overall ceiling sanity

        data, info, report, _, _ = read_output(output_dir)
        self.assertEqual(info.frames, int(round(duration_s * 44100)))
        self.assertEqual(data.shape[0], report["total_samples"])

    def test_single_thread_env_forced_in_subprocess(self):
        input_dir, output_dir = make_fixture()
        env = dict(os.environ)
        for key in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        ):
            env.pop(key, None)
        code = (
            "from rhythm_mixer.cli import build_parser; "
            "import rhythm_mixer.cli as c, os; "
            "p=build_parser(); "
            "rc=c.main(['-i', os.environ['RM_IN'], '-o', os.environ['RM_OUT']]); "
            "print('THREADS', os.environ.get('OMP_NUM_THREADS'), "
            "os.environ.get('OPENBLAS_NUM_THREADS')); "
            "raise SystemExit(rc)"
        )
        env["RM_IN"] = input_dir
        env["RM_OUT"] = output_dir
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("THREADS 1 1", result.stdout)


if __name__ == "__main__":
    unittest.main()
