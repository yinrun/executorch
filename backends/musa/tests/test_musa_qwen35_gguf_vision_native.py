"""Host/meta coverage of the explicit vision export conversion contract."""

import unittest

import torch
import torch.nn.functional as functional

from executorch.backends.musa.vision_gelu import VISION_GELU
from executorch.backends.musa.vision_layer_norm import VISION_LAYER_NORM
from executorch.backends.musa.vision_softmax import VISION_SOFTMAX
from examples.musa.aoti_qwen35_gguf_vision_native import use_native_vision_math


class VisionMathFixture(torch.nn.Module):
    def __init__(self, normalizations=49):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(1024, device="meta"))
        self.bias = torch.nn.Parameter(torch.empty(1024, device="meta"))
        self.normalizations = normalizations

    def forward(self, hidden, scores, expanded, merged):
        for _ in range(self.normalizations):
            hidden = functional.layer_norm(
                hidden, (1024,), self.weight, self.bias, 1e-6
            )
        for _ in range(24):
            scores = functional.softmax(scores, dim=-1, dtype=torch.float32)
            expanded = functional.gelu(expanded, approximate="tanh")
        return hidden, scores, expanded, functional.gelu(merged)


class TestNativeVisionExport(unittest.TestCase):
    def export_fixture(self, rows=800, normalizations=49, merged_rows=None):
        arguments = (
            torch.empty((rows, 1024), device="meta"),
            torch.empty((16, rows, rows), device="meta"),
            torch.empty((rows, 4096), device="meta"),
            torch.empty((merged_rows or rows // 4, 4096), device="meta"),
        )
        return torch.export.export(VisionMathFixture(normalizations), arguments)

    def test_both_grids_preserve_parameters_and_output_metadata(self):
        for rows in (800, 1200):
            exported = self.export_fixture(rows)
            parameters = dict(exported.state_dict)
            output_before = tuple(exported.graph.nodes)[-1].args
            counts = use_native_vision_math(exported)
            self.assertEqual(
                counts, dict(layer_norm=49, softmax=24, gelu_tanh=24, gelu_erf=1)
            )
            targets = [node.target for node in exported.graph.nodes]
            self.assertEqual(targets.count(VISION_LAYER_NORM), 49)
            self.assertEqual(targets.count(VISION_SOFTMAX), 24)
            self.assertEqual(targets.count(VISION_GELU), 25)
            self.assertEqual(tuple(exported.graph.nodes)[-1].args, output_before)
            self.assertEqual(set(exported.state_dict), set(parameters))
            for name, parameter in parameters.items():
                self.assertIs(exported.state_dict[name], parameter)

    def test_incomplete_graph_is_not_partially_rewritten(self):
        exported = self.export_fixture(normalizations=48)
        original = str(exported.graph)
        with self.assertRaisesRegex(ValueError, "complete 24-block"):
            use_native_vision_math(exported)
        self.assertEqual(str(exported.graph), original)

    def test_mixed_grids_are_not_partially_rewritten(self):
        exported = self.export_fixture(merged_rows=300)
        original = str(exported.graph)
        with self.assertRaisesRegex(ValueError, "consistent"):
            use_native_vision_math(exported)
        self.assertEqual(str(exported.graph), original)

    def test_already_converted_graph_is_not_silently_accepted(self):
        exported = self.export_fixture()
        use_native_vision_math(exported)
        original = str(exported.graph)
        with self.assertRaisesRegex(ValueError, "complete 24-block"):
            use_native_vision_math(exported)
        self.assertEqual(str(exported.graph), original)


if __name__ == "__main__":
    unittest.main()
