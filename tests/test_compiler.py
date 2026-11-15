from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from matgraph.compiler import compile_graph
from matgraph.errors import PipelineError
from matgraph.limits import Limits
from matgraph.samples import generate_samples, sample_graph


def _constant_graph() -> dict:
    return {
        "schema": "material-graph/v1",
        "name": "constant-normal",
        "nodes": [
            {"id": "c1", "type": "constant", "value": [0.05, 0.1, 0.15]},
            {"id": "c2", "type": "constant", "value": [0.15, 0.3, 0.45]},
            {"id": "tint", "type": "constant", "value": [2.0, 2.0, 2.0]},
            {
                "id": "mul",
                "type": "multiply",
                "inputs": {"a": "c1", "b": "tint"},
            },
            {
                "id": "base",
                "type": "add",
                "inputs": {"a": "mul", "b": "c2"},
            },
            {"id": "rough_a", "type": "constant", "value": 0.2},
            {"id": "rough_b", "type": "constant", "value": 0.8},
            {"id": "half", "type": "constant", "value": 0.5},
            {
                "id": "rough",
                "type": "mix",
                "inputs": {"a": "rough_a", "b": "rough_b", "factor": "half"},
            },
            {"id": "metal", "type": "constant", "value": 0.1},
            {"id": "emissive", "type": "constant", "value": [0.0, 0.0, 0.0]},
            {"id": "tangent", "type": "constant", "value": [0.5, 0.5, 1.0]},
            {"id": "strength", "type": "constant", "value": 1.0},
            {
                "id": "normal",
                "type": "normal_map",
                "inputs": {"normal": "tangent", "strength": "strength"},
            },
        ],
        "outputs": {
            "base_color": "base",
            "normal": "normal",
            "roughness": "rough",
            "metallic": "metal",
            "emissive": "emissive",
        },
    }


def _minimal_graph() -> dict:
    return {
        "schema": "material-graph/v1",
        "name": "minimal",
        "nodes": [
            {"id": "base", "type": "constant", "value": [0.1, 0.2, 0.3]},
            {"id": "normal", "type": "constant", "value": [0.0, 0.0, 1.0]},
            {"id": "rough", "type": "constant", "value": 0.5},
            {"id": "metal", "type": "constant", "value": 0.0},
            {"id": "emissive", "type": "constant", "value": [0.0, 0.0, 0.0]},
        ],
        "outputs": {
            "base_color": "base",
            "normal": "normal",
            "roughness": "rough",
            "metallic": "metal",
            "emissive": "emissive",
        },
    }


class CompilerTests(unittest.TestCase):
    def assert_graph_error(self, graph: dict, code: str, node_index=None, limits=None):
        with self.assertRaises(PipelineError) as caught:
            compile_graph(graph, Path("."), limits or Limits())
        error = caught.exception
        self.assertEqual(error.code, code)
        if node_index is not None:
            self.assertEqual(error.node_index, node_index)
        self.assertEqual(error.stage, "graph")
        envelope = error.as_envelope()
        self.assertFalse(envelope["ok"])
        self.assertEqual(envelope["error"]["code"], code)
        return error

    def test_constant_folding_eliminates_constant_instructions(self):
        compiled = compile_graph(_constant_graph(), Path("."))
        self.assertEqual(compiled.instruction_order(), ["normal"])
        self.assertEqual(compiled.outputs["base_color"].kind, "constant")
        self.assertEqual(compiled.outputs["base_color"].value, (0.25, 0.5, 0.75))
        self.assertEqual(compiled.outputs["roughness"].value, 0.5)
        folded = {folded.node_id for folded in compiled.folded_nodes}
        self.assertIn("base", folded)
        self.assertIn("mul", folded)
        self.assertIn("rough", folded)
        self.assertNotIn("normal", folded)

    def test_stable_order_is_independent_of_node_array_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            generate_samples(Path(temporary))
            graph = sample_graph()
            first = compile_graph(graph, Path(temporary))
            shuffled = copy.deepcopy(graph)
            shuffled["nodes"].reverse()
            second = compile_graph(shuffled, Path(temporary))
            self.assertEqual(first.instruction_order(), second.instruction_order())

    def test_output_type_mismatch_is_structured(self):
        graph = _minimal_graph()
        del graph["nodes"][2]
        graph["outputs"]["roughness"] = "base"
        self.assert_graph_error(graph, "E_OUTPUT_TYPE", 0)

    def test_cycle_is_detected(self):
        graph = _minimal_graph()
        graph["nodes"].extend(
            [
                {
                    "id": "cycle_a",
                    "type": "multiply",
                    "inputs": {"a": "cycle_b", "b": "base"},
                },
                {
                    "id": "cycle_b",
                    "type": "multiply",
                    "inputs": {"a": "cycle_a", "b": "base"},
                },
            ]
        )
        graph["outputs"]["base_color"] = "cycle_a"
        self.assert_graph_error(graph, "E_CYCLE")

    def test_unreachable_node_is_detected(self):
        graph = _minimal_graph()
        graph["nodes"].append({"id": "dead", "type": "constant", "value": 3.0})
        self.assert_graph_error(graph, "E_NO_OUTPUT_PATH", 5)

    def test_missing_input_is_detected(self):
        graph = _minimal_graph()
        graph["nodes"].append(
            {
                "id": "bad_add",
                "type": "add",
                "inputs": {"a": "base"},
            }
        )
        graph["outputs"]["base_color"] = "bad_add"
        self.assert_graph_error(graph, "E_MISSING_INPUT", 5)

    def test_dimension_mismatch_is_detected(self):
        graph = _minimal_graph()
        graph["nodes"].extend(
            [
                {"id": "uvlike", "type": "constant", "value": [0.0, 0.0]},
                {
                    "id": "bad_add",
                    "type": "add",
                    "inputs": {"a": "base", "b": "uvlike"},
                },
            ]
        )
        graph["outputs"]["base_color"] = "bad_add"
        self.assert_graph_error(graph, "E_INPUT_DIMENSION", 6)

    def test_duplicate_id_is_detected(self):
        graph = _minimal_graph()
        graph["nodes"][1]["id"] = "base"
        self.assert_graph_error(graph, "E_DUPLICATE_ID", 1)

    def test_depth_limit_is_detected(self):
        graph = _minimal_graph()
        graph["nodes"].extend(
            [
                {
                    "id": "add_one",
                    "type": "add",
                    "inputs": {"a": "base", "b": "base"},
                },
                {
                    "id": "add_two",
                    "type": "add",
                    "inputs": {"a": "add_one", "b": "base"},
                },
            ]
        )
        graph["outputs"]["base_color"] = "add_two"
        self.assert_graph_error(
            graph,
            "E_GRAPH_TOO_DEEP",
            limits=Limits(max_graph_depth=2),
        )

    def test_node_limit_is_detected(self):
        graph = _minimal_graph()
        self.assert_graph_error(graph, "E_LIMIT", 2, Limits(max_nodes=2))

    def test_input_reference_is_detected(self):
        graph = _minimal_graph()
        graph["nodes"].append(
            {
                "id": "bad_add",
                "type": "add",
                "inputs": {"a": "base", "b": "does_not_exist"},
            }
        )
        graph["outputs"]["base_color"] = "bad_add"
        self.assert_graph_error(graph, "E_INPUT_NOT_FOUND", 5)

    def test_fresnel_is_monotonic_toward_grazing(self):
        from matgraph.evaluate import fresnel_schlick

        cosine = np.asarray([1.0, 0.7, 0.4, 0.2, 0.05, 0.0])
        values = fresnel_schlick(cosine, np.full_like(cosine, 1.45))
        self.assertTrue(np.all(np.diff(values) >= 0.0))
        self.assertGreater(values[-1], values[0])
        self.assertTrue(np.all(values >= 0.0))
        self.assertTrue(np.all(values <= 1.0))


if __name__ == "__main__":
    unittest.main()
