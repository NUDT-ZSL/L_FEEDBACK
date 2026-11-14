import copy
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from voice_mixer.errors import ProjectValidationError
from voice_mixer.parser import parse_project
from voice_mixer.sample_project import build_sample_project
from voice_mixer.timeline import TimelineError, build_timeline, tick_to_sample


class ParserTimelineTest(unittest.TestCase):
    def setUp(self):
        self.project_dict = build_sample_project()

    def _parse(self):
        return parse_project(copy.deepcopy(self.project_dict))

    def test_valid_project_parses(self):
        project = self._parse()
        self.assertEqual(project.sample_rate, 16000)
        self.assertEqual(set(project.channels), {"squad", "command", "recon"})
        self.assertEqual(len(project.events), 13)

    def test_bad_sample_rate(self):
        self.project_dict["sample_rate"] = 4000
        with self.assertRaises(ProjectValidationError) as context:
            self._parse()
        self.assertEqual(context.exception.code, "E_SAMPLE_RATE")

    def test_unknown_channel_reference(self):
        self.project_dict["speakers"][0]["bound_channel"] = "ghost"
        with self.assertRaises(ProjectValidationError):
            self._parse()
        try:
            self._parse()
        except ProjectValidationError as exc:
            errors = exc.errors
        self.assertTrue(any(e["code"] == "E_UNKNOWN_REF" for e in errors))

    def test_tick_order_error(self):
        self.project_dict["events"][-1]["tick"] = 1
        with self.assertRaises(ProjectValidationError):
            self._parse()
        try:
            self._parse()
        except ProjectValidationError as exc:
            errors = exc.errors
        self.assertTrue(any(e["code"] == "E_TICK_ORDER" for e in errors))

    def test_schema_error_has_path(self):
        del self.project_dict["channels"]
        with self.assertRaises(ProjectValidationError):
            self._parse()
        try:
            self._parse()
        except ProjectValidationError as exc:
            errors = exc.errors
        self.assertTrue(any(e["code"] == "E_SCHEMA" for e in errors))

    def test_tick_to_sample_millisecond_precision(self):
        self.assertEqual(tick_to_sample(1, 16000, 1000), 16)
        self.assertEqual(tick_to_sample(0, 16000, 1000), 0)

    def test_unclosed_start_is_event_error(self):
        self.project_dict["events"] = [
            {"tick": 10, "type": "start", "speaker": "alpha"},
        ]
        project = self._parse()
        with self.assertRaises(TimelineError) as context:
            build_timeline(project)
        self.assertTrue(any(e["code"] == "E_EVENT" for e in context.exception.errors))

    def test_end_without_start_is_event_error(self):
        self.project_dict["events"] = [
            {"tick": 10, "type": "end", "speaker": "alpha"},
        ]
        project = self._parse()
        with self.assertRaises(TimelineError) as context:
            build_timeline(project)
        self.assertTrue(any(e["code"] == "E_EVENT" for e in context.exception.errors))

    def test_switch_splits_fragment_and_keeps_duration(self):
        self.project_dict["events"] = [
            {"tick": 0, "type": "start", "speaker": "alpha"},
            {"tick": 100, "type": "switch", "speaker": "alpha", "channel": "command"},
            {"tick": 200, "type": "end", "speaker": "alpha"},
        ]
        project = self._parse()
        timeline = build_timeline(project)
        self.assertEqual(len(timeline.fragments), 2)
        self.assertEqual(timeline.fragments[0].channel_id, "squad")
        self.assertEqual(timeline.fragments[1].channel_id, "command")
        self.assertEqual(timeline.fragments[0].start_sample, 0)
        self.assertEqual(timeline.fragments[1].end_sample, tick_to_sample(200, 16000, 1000))

    def test_same_tick_start_end_zero_length_reported(self):
        self.project_dict["events"] = [
            {"tick": 100, "type": "start", "speaker": "alpha"},
            {"tick": 100, "type": "end", "speaker": "alpha"},
        ]
        project = self._parse()
        with self.assertRaises(TimelineError) as context:
            build_timeline(project)
        self.assertTrue(any("zero-length" in e["message"] for e in context.exception.errors))

    def test_blocked_segment_is_retained_but_marked(self):
        self.project_dict["events"] = [
            {"tick": 0, "type": "start", "speaker": "alpha"},
            {"tick": 50, "type": "block", "speaker": "alpha"},
            {"tick": 100, "type": "unblock", "speaker": "alpha"},
            {"tick": 150, "type": "end", "speaker": "alpha"},
        ]
        project = self._parse()
        timeline = build_timeline(project)
        self.assertEqual(len(timeline.fragments), 3)
        self.assertFalse(timeline.fragments[0].blocked)
        self.assertTrue(timeline.fragments[1].blocked)
        self.assertFalse(timeline.fragments[2].blocked)


if __name__ == "__main__":
    unittest.main()
