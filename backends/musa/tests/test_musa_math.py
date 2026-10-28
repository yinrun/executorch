import unittest
from unittest import mock

import torch

from executorch.backends.musa import MusaBackend
from executorch.backends.musa import musa_backend


class MathPair(torch.nn.Module):
    def __init__(self, beta=1, threshold=20):
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, value):
        return torch.exp(value), torch.nn.functional.softplus(
            value, self.beta, self.threshold
        )


class TestMusaMathDecomposition(unittest.TestCase):
    def test_native_fallback_symbols(self):
        kernels = MusaBackend.get_supported_fallback_kernels()
        self.assertIn("aoti_torch_musa_exp_out", kernels)
        self.assertIn("aoti_torch_musa_softplus_out", kernels)

    def test_meta_produces_contiguous_non_aliasing_outputs(self):
        value = torch.empty_strided((2, 3), (1, 2), device="meta")
        for operation in (musa_backend._MUSA_EXP, musa_backend._MUSA_SOFTPLUS):
            result = operation(value)
            self.assertEqual(result.shape, value.shape)
            self.assertEqual(result.dtype, torch.float32)
            self.assertTrue(result.is_contiguous())
            self.assertIsNot(result, value)
            self.assertEqual(value.stride(), (1, 2))

    def test_cpu_decomposition_does_not_call_musa_ops(self):
        value = torch.tensor([-4.0, -1.0, 0.0, 1.0, 4.0])
        with mock.patch.object(musa_backend, "_MUSA_EXP") as exp, mock.patch.object(
            musa_backend, "_MUSA_SOFTPLUS"
        ) as softplus:
            actual = musa_backend._decompose_musa_exp(value)
            torch.testing.assert_close(actual, torch.exp(value))
            actual = musa_backend._decompose_musa_softplus(value, 2, 10)
            torch.testing.assert_close(
                actual, torch.nn.functional.softplus(value, 2, 10)
            )
            exp.assert_not_called()
            softplus.assert_not_called()


@unittest.skipUnless(
    hasattr(torch, "musa") and torch.musa.is_available(), "requires MUSA"
)
class TestMusaMathDevice(unittest.TestCase):
    def test_fp32_native_selection_and_fallbacks(self):
        decompositions = MusaBackend.get_decomposition_table()
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            value = torch.randn((2, 3), device="musa", dtype=dtype)
            for beta, threshold in ((1, 20), (2, 20), (1, 10)):
                with self.subTest(dtype=dtype, beta=beta, threshold=threshold):
                    graph = torch.export.export(MathPair(beta, threshold), (value,))
                    graph = graph.run_decompositions(decompositions)
                    targets = {node.target for node in graph.graph.nodes}
                    outputs = next(
                        node.args[0]
                        for node in graph.graph.nodes
                        if node.op == "output"
                    )
                    self.assertEqual(
                        outputs[0].target == musa_backend._MUSA_EXP,
                        dtype == torch.float32,
                    )
                    self.assertEqual(
                        musa_backend._MUSA_SOFTPLUS in targets,
                        dtype == torch.float32 and beta == 1 and threshold == 20,
                    )
                    for node in graph.graph.nodes:
                        if node.target in (
                            musa_backend._MUSA_EXP,
                            musa_backend._MUSA_SOFTPLUS,
                        ):
                            self.assertEqual(
                                node.args[0].meta["val"].dtype, torch.float32
                            )
                    self.assertTrue(
                        all(node.meta["val"].dtype == dtype for node in outputs)
                    )

    def test_eager_layout_values_and_nonmutation(self):
        base = torch.linspace(-20, 20, 18, device="musa").reshape(3, 6)
        for value in (
            base, base.t(), base[:, ::2], base[:1].expand(4, 6), base[:0], base[0, 0]
        ):
            before = value.clone()
            for operation, reference in (
                (musa_backend._MUSA_EXP, torch.exp),
                (musa_backend._MUSA_SOFTPLUS, torch.nn.functional.softplus),
            ):
                result = operation(value)
                self.assertTrue(result.is_contiguous())
                torch.testing.assert_close(result, reference(value), atol=0, rtol=0)
                torch.testing.assert_close(value, before, atol=0, rtol=0)
                if value.numel():
                    self.assertNotEqual(result.data_ptr(), value.data_ptr())


if __name__ == "__main__":
    unittest.main()
