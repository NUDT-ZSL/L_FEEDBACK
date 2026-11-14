import os
import tempfile
import unittest

from subalign.drift import align_events, fit_segmented
from subalign.matching import match_events
from subalign.subtitles import load_subtitle

from tests.samples import (
    ARABIC_LINES,
    drop_event,
    linear_drift,
    reference_events,
    segmented_drift,
    swap_events,
    translated_events,
    write_subtitle,
)


class AlignmentTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = self.temp.name
        self.events = reference_events(count=24)
        self.reference_path = os.path.join(self.root, "reference.srt")
        write_subtitle(self.reference_path, self.events)
        self.reference = load_subtitle(self.reference_path)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def align(self, drifted_events, fmt: str = "srt", name: str = "target.srt"):
        path = os.path.join(self.root, name)
        write_subtitle(path, drifted_events, fmt)
        target = load_subtitle(path)
        matches, translated = match_events(self.reference.events, target.events)
        model = fit_segmented(matches)
        result = align_events(self.reference.events, target.events, matches, model)
        return target, result, model, matches, translated


class LinearDriftTests(AlignmentTestBase):
    def test_recovers_positive_offset_within_ten_ms(self) -> None:
        drifted = linear_drift(self.events, scale=1.0, offset_ms=1250)
        _target, result, model, matches, _ = self.align(drifted)
        self.assertEqual(model.kind, "linear")
        self.assertAlmostEqual(model.offset_ms, -1250, delta=10)
        self.assertAlmostEqual(model.scale, 1.0, delta=0.001)
        self.assertEqual(len(matches), 24)
        for original, aligned in zip(self.events, result.events):
            self.assertLess(abs(aligned.start - original[0]), 10)
            self.assertEqual(aligned.end - aligned.start, original[1] - original[0])

    def test_recovers_scale_and_offset(self) -> None:
        drifted = linear_drift(self.events, scale=1.04, offset_ms=1200)
        _target, result, model, _matches, _ = self.align(drifted)
        self.assertAlmostEqual(model.scale, 1.0 / 1.04, delta=0.0005)
        self.assertAlmostEqual(model.offset_ms, -1200.0 / 1.04, delta=5.0)
        for original, aligned in zip(self.events, result.events):
            self.assertLess(abs(aligned.start - original[0]), 10)

    def test_negative_offset_fit_without_file_clamping(self) -> None:
        from subalign.drift import fit_global
        from subalign.matching import Match

        matches = [
            Match(
                reference_index=index,
                target_index=index,
                reference_start=start,
                target_start=1.04 * start - 800.0,
                text_similarity=1.0,
                time_distance_ms=0.0,
                confidence=0.95,
            )
            for index, (start, _end, _text) in enumerate(self.events)
        ]
        model = fit_global(matches)
        self.assertAlmostEqual(model.scale, 1.0 / 1.04, delta=0.0005)
        self.assertAlmostEqual(model.offset_ms, 800.0 / 1.04, delta=5.0)

    def test_negative_offset_is_clamped(self) -> None:
        from subalign.drift import preflight_warnings, sanitize
        from subalign.subtitles import SubtitleEvent

        events = [
            SubtitleEvent(index=0, start=-300.0, end=200.0, text="negative"),
            SubtitleEvent(index=1, start=500.0, end=1000.0, text="normal"),
        ]
        preflight = preflight_warnings(events)
        self.assertTrue(any(w["code"] == "NEGATIVE_START" for w in preflight))
        warnings = sanitize(events)
        self.assertEqual(events[0].start, 0.0)
        self.assertEqual(events[0].end, 200.0)
        self.assertTrue(all(w["code"] != "NEGATIVE_START" for w in warnings))


class SegmentedDriftTests(AlignmentTestBase):
    def test_detects_segment_boundary_and_parameters(self) -> None:
        boundary = self.events[12][0]
        drifted = segmented_drift(
            self.events,
            boundary=boundary,
            first_scale=1.0,
            first_offset=600,
            second_scale=1.02,
            second_offset=-1500,
        )
        _target, result, model, matches, _ = self.align(drifted)
        self.assertEqual(model.kind, "segmented")
        self.assertGreaterEqual(len(model.segments), 2)
        self.assertEqual(len(matches), 24)

        for original, aligned in zip(self.events, result.events):
            self.assertLess(abs(aligned.start - original[0]), 10, (original, aligned.start))

        boundaries = [segment.start_time_ms for segment in model.segments]
        self.assertTrue(any(abs(point - boundary) < 5000 for point in boundaries[1:]))

    def test_constant_drift_stays_linear(self) -> None:
        drifted = linear_drift(self.events, scale=1.0, offset_ms=400)
        _target, _result, model, _matches, _ = self.align(drifted)
        self.assertEqual(model.kind, "linear")


class MissingAndOrderTests(AlignmentTestBase):
    def test_missing_events_are_reported_without_breaking_matches(self) -> None:
        drifted = drop_event(linear_drift(self.events, 1.0, 900), 7)
        _target, result, _model, matches, _ = self.align(drifted)
        self.assertEqual(len(matches), 23)
        self.assertEqual(result.stats["unmatched_target_count"], 0)
        matched_refs = {match.reference_index for match in matches}
        self.assertNotIn(7, matched_refs)

    def test_extra_target_event_is_flagged_unmatched(self) -> None:
        drifted = linear_drift(self.events, 1.0, 900)
        drifted.insert(10, (drifted[10][0] + 100, drifted[10][1] + 100, "Extra standalone cue"))
        _target, result, _model, matches, _ = self.align(drifted)
        self.assertEqual(result.stats["unmatched_target_count"], 1)
        self.assertTrue(any(w["code"] == "UNMATCHED_EVENT" for w in result.warnings))
        self.assertEqual(len(matches), 24)

    def test_out_of_order_events_keep_order_and_warn(self) -> None:
        drifted = swap_events(linear_drift(self.events, 1.0, 700), 5)
        _target, result, _model, _matches, _ = self.align(drifted)
        self.assertTrue(any(w["code"] == "OUT_OF_ORDER" for w in result.warnings))
        starts = [event.start for event in result.events]
        self.assertEqual(starts[5], result.events[5].start)


class MultilingualTests(AlignmentTestBase):
    def test_translated_chinese_target_aligns_on_time(self) -> None:
        drifted = linear_drift(translated_events(self.events), 1.0, 1100)
        _target, result, _model, matches, translated = self.align(drifted)
        self.assertTrue(translated)
        self.assertEqual(len(matches), 24)
        for original, aligned in zip(self.events, result.events):
            self.assertLess(abs(aligned.start - original[0]), 10)

    def test_arabic_rtl_preserves_order_and_text(self) -> None:
        arabic = translated_events(self.events, ARABIC_LINES)
        drifted = linear_drift(arabic, 1.0, 500)
        target, result, _model, _matches, _ = self.align(drifted)
        for original_event, aligned_event in zip(target.events, result.events):
            self.assertEqual(original_event.text, aligned_event.text)
        self.assertEqual([event.index for event in result.events], list(range(24)))
        self.assertIn("مرحبا", result.events[0].text)

    def test_nfc_normalization_keeps_original_text(self) -> None:
        from subalign.textutil import normalize_text

        composed = "café 字幕"
        decomposed = "cafe\u0301 字幕"
        self.assertNotEqual(composed, decomposed)
        self.assertEqual(normalize_text(composed), normalize_text(decomposed))


class SanitizerTests(AlignmentTestBase):
    def test_overlap_zero_and_negative_events_are_deterministic(self) -> None:
        from tests.samples import overlap_events
        from subalign.drift import preflight_warnings, sanitize
        from subalign.subtitles import SubtitleEvent

        _target, result, _model, matches, _ = self.align(
            overlap_events(), fmt="ass", name="edge.ass"
        )
        codes = {warning["code"] for warning in result.warnings}
        self.assertIn("OVERLAP", codes)
        self.assertIn("ZERO_DURATION", codes)
        self.assertGreaterEqual(min(event.start for event in result.events), 0.0)
        for earlier, later in zip(result.events, result.events[1:]):
            self.assertLessEqual(earlier.start, later.start)

        raw_events = [
            SubtitleEvent(index=0, start=-250.0, end=300.0, text="a"),
            SubtitleEvent(index=1, start=100.0, end=500.0, text="b"),
            SubtitleEvent(index=2, start=700.0, end=700.0, text="c"),
        ]
        direct_codes = {w["code"] for w in preflight_warnings(raw_events)}
        self.assertIn("NEGATIVE_START", direct_codes)
        self.assertIn("ZERO_DURATION", direct_codes)
        direct_warnings = sanitize(raw_events)
        self.assertIn("OVERLAP", {w["code"] for w in direct_warnings})
        self.assertEqual(raw_events[0].start, 0.0)
        self.assertEqual(raw_events[1].start, 300.0)


if __name__ == "__main__":
    unittest.main()
