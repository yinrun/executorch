# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch import nn
from torch.export import export

from executorch.backends.cuda.cuda_weight_collector import (
    decode_cuda_aoti_metadata,
)
from executorch.backends.musa import MusaBackend


class TestMusaAotiFallbacks(unittest.TestCase):
    def test_registers_native_slimtensor_fallback(self) -> None:
        fallbacks = MusaBackend.get_supported_fallback_kernels()
        self.assertIn("aoti_torch_musa_addmm_out", fallbacks)


@unittest.skipUnless(torch.musa.is_available(), "MUSA is not available")
class TestMusaAoti(unittest.TestCase):
    def test_pointwise_aoti_compile(self) -> None:
        class AddOne(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + 1

        example = torch.randn(2, 8, device="musa")
        exported = export(AddOne().eval(), (example,))
        result = MusaBackend.preprocess(
            exported,
            [MusaBackend.generate_method_name_compile_spec("forward")],
        )

        self.assertGreater(len(result.processed_bytes), 0)
        self.assertIn("aoti_musa_blob", result.data_store_output.external_data)

    def test_weighted_linear_aoti_compile(self) -> None:
        class WeightedLinear(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(8, 4)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)

        example = torch.randn(2, 8, device="musa")
        module = WeightedLinear().eval().to("musa")
        exported = export(module, (example,))
        result = MusaBackend.preprocess(
            exported,
            [MusaBackend.generate_method_name_compile_spec("forward")],
        )

        metadata = decode_cuda_aoti_metadata(result.processed_bytes)
        self.assertGreaterEqual(len(metadata.entries), 1)
        self.assertIn("aoti_musa_blob", result.data_store_output.external_data)

        entries = {entry.fqn: entry for entry in metadata.entries}
        self.assertIn("linear.weight", entries)
        self.assertEqual(entries["linear.weight"].sizes, (4, 8))
