from __future__ import annotations

import unittest

from pe_audit.binary import call_local, ret
from pe_audit.reachability import path_for_function, reachability
from tests.support import make_analysis


def _call(index: int) -> bytes:
    return call_local(index).encode() + ret().encode()


class ReachabilityTest(unittest.TestCase):
    def test_basic_reachability_and_unreachable_orphan(self) -> None:
        names = ["entry", "middle", "target", "isolated"]
        index = {name: position for position, name in enumerate(names)}
        analysis = make_analysis(
            {
                "entry": _call(index["middle"]),
                "middle": _call(index["target"]),
                "target": ret().encode(),
                "isolated": ret().encode(),
            },
            local_names=names,
        )
        result = reachability(analysis, ["entry"])
        self.assertEqual(result["unreachable"], ["isolated"])
        target = path_for_function(result, "target")
        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual(target["path"], ["entry", "middle", "target"])
        self.assertEqual(
            [edge["callee"] for edge in target["edges"]],
            ["middle", "target"],
        )

    def test_cycle_terminates_and_paths_have_no_duplicates(self) -> None:
        names = ["entry", "a", "b"]
        index = {name: position for position, name in enumerate(names)}
        analysis = make_analysis(
            {
                "entry": _call(index["a"]),
                "a": _call(index["b"]),
                "b": _call(index["a"]),
            },
            local_names=names,
        )
        result = reachability(analysis, ["entry"])
        self.assertEqual(result["cycles"], [["a", "b"]])
        for item in result["reachable"]:
            self.assertEqual(len(item["path"]), len(set(item["path"])))

    def test_self_recursion_is_cycle_and_simple_path(self) -> None:
        names = ["f"]
        analysis = make_analysis(
            {"f": _call(0)},
            local_names=names,
        )
        result = reachability(analysis, ["f"])
        self.assertEqual(result["cycles"], [["f"]])
        self.assertEqual(result["reachable"][0]["path"], ["f"])
        self.assertEqual(result["reachable"][0]["edges"], [])
        self.assertEqual(result["edges"][0]["callee"], "f")

    def test_missing_entry_is_reported_not_raised(self) -> None:
        analysis = make_analysis({"f": ret().encode()})
        result = reachability(analysis, ["missing"])
        self.assertEqual(result["missing_entries"], ["missing"])
        self.assertTrue(result["graph_incomplete"])
        self.assertEqual(result["unreachable"], ["f"])

    def test_broken_call_marks_graph_incomplete(self) -> None:
        analysis = make_analysis({"f": call_local(99).encode() + ret().encode()})
        result = reachability(analysis, ["f"])
        self.assertTrue(result["graph_incomplete"])
        self.assertTrue(result["broken_edges"])
        self.assertEqual(result["reachable_incomplete_functions"], ["f"])

    def test_edges_are_sorted_and_deterministic(self) -> None:
        names = ["entry", "a", "b"]
        index = {name: position for position, name in enumerate(names)}
        analysis = make_analysis(
            {
                "entry": _call(index["b"]) + _call(index["a"]),
                "a": ret().encode(),
                "b": ret().encode(),
            },
            local_names=names,
        )
        first = reachability(analysis, ["entry"])
        second = reachability(analysis, ["entry"])
        self.assertEqual(first, second)
        self.assertEqual(
            [(edge["caller"], edge["callee"]) for edge in first["edges"]],
            [("entry", "a"), ("entry", "b")],
        )


if __name__ == "__main__":
    unittest.main()
