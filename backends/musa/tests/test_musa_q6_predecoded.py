"""Lossless, explicitly owned Q6 preprocessing and opaque export contract."""

import unittest

import numpy as np
import torch

from executorch.backends.musa.gguf import dequantize_gguf
from executorch.backends.musa.gguf_packed import GgufEmbedding, GgufLinear
from executorch.backends.musa.q6_predecoded import (
    Q6_PREDECODED,
    dequantize_predecoded_q6,
    predecode_q6,
    predecoded_shape,
)


class TestQ6PredecodedMetadata(unittest.TestCase):
    def test_shape_and_export(self):
        packed = torch.empty((9, 3200), dtype=torch.uint8, device="meta")
        self.assertEqual(predecoded_shape(packed), (9, 2560))
        for module, value in (
            (GgufLinear(packed, Q6_PREDECODED), torch.empty((1, 2560), device="meta")),
            (
                GgufEmbedding(packed, Q6_PREDECODED),
                torch.empty((1,), dtype=torch.int64, device="meta"),
            ),
        ):
            exported = torch.export.export(module, (value,), strict=True)
            calls = [
                node for node in exported.graph.nodes if node.op == "call_function"
            ]
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0].args[-1], 114)
        with self.assertRaises(RuntimeError):
            predecoded_shape(torch.empty((2, 210), dtype=torch.uint8, device="meta"))
        with self.assertRaises(ValueError):
            predecode_q6(torch.empty((2, 210), dtype=torch.uint8, device="meta"))


@unittest.skipUnless(
    hasattr(torch, "musa") and torch.musa.is_available(), "requires MUSA"
)
class TestQ6PredecodedGpu(unittest.TestCase):
    def test_all_levels_scales_and_independent_ownership(self):
        generator = np.random.default_rng(1729)
        data = generator.integers(0, 256, (1024, 210), dtype=np.uint8)
        bits = generator.integers(0, 65536, 1024, dtype=np.uint16)
        bits &= np.uint16(0xFBFF)
        bits[:8] = [0, 0x8000, 1, 0x8001, 0x3C00, 0xBC00, 0x7BFF, 0xFBFF]
        data[:, 208:] = bits.view(np.uint8).reshape(-1, 2)
        raw = torch.from_numpy(data.reshape(128, -1)).to("musa")
        before = raw.clone()
        packed = predecode_q6(raw)
        expected = dequantize_gguf(raw, "q6_k")
        actual = dequantize_predecoded_q6(packed)
        self.assertTrue(
            torch.equal(actual.view(torch.int32), expected.view(torch.int32))
        )
        self.assertTrue(torch.equal(raw, before))
        self.assertNotEqual(raw.data_ptr(), packed.data_ptr())
        raw.zero_()
        self.assertTrue(
            torch.equal(
                dequantize_predecoded_q6(packed).view(torch.int32),
                actual.view(torch.int32),
            )
        )

    def test_linear_embedding_and_shared_buffer(self):
        raw = torch.zeros((9, 420), dtype=torch.uint8, device="musa")
        raw.view(-1, 210)[:, 209] = 0x3C
        raw.view(-1, 210)[:, 192:208] = 1
        packed = predecode_q6(raw)
        linear = GgufLinear(packed, Q6_PREDECODED)
        embedding = GgufEmbedding(packed, Q6_PREDECODED)
        self.assertEqual(
            linear.packed_weight.data_ptr(), embedding.packed_weight.data_ptr()
        )
        indices = torch.tensor([8, 0], device="musa")
        self.assertTrue(
            torch.equal(embedding(indices), GgufEmbedding(raw, "q6_k")(indices))
        )
        value = torch.ones((2, 512), device="musa")
        self.assertTrue(torch.equal(linear(value), GgufLinear(raw, "q6_k")(value)))


if __name__ == "__main__":
    unittest.main()
