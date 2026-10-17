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
        self.assertIn("aoti_torch_musa_mm_out", fallbacks)
        self.assertIn("aoti_torch_musa_bmm_out", fallbacks)
        self.assertIn("aoti_torch_musa_convolution", fallbacks)


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

    def test_conv2d_aoti_compile(self) -> None:
        class Conv2dModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(3, 4, 3, padding=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.conv(x)

        example = torch.randn(1, 3, 8, 8, device="musa")
        module = Conv2dModel().eval().to("musa")
        exported = export(module, (example,))
        result = MusaBackend.preprocess(
            exported,
            [MusaBackend.generate_method_name_compile_spec("forward")],
        )

        self.assertGreater(len(result.processed_bytes), 0)
        self.assertIn("aoti_musa_blob", result.data_store_output.external_data)

    def test_conv3d_aoti_compile(self) -> None:
        class Conv3dModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv3d(2, 3, 3, padding=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.conv(x)

        example = torch.randn(1, 2, 4, 4, 4, device="musa")
        module = Conv3dModel().eval().to("musa")
        exported = export(module, (example,))
        result = MusaBackend.preprocess(
            exported,
            [MusaBackend.generate_method_name_compile_spec("forward")],
        )

        self.assertGreater(len(result.processed_bytes), 0)
        self.assertIn("aoti_musa_blob", result.data_store_output.external_data)

    def test_conv_transpose2d_aoti_compile(self) -> None:
        class ConvTranspose2dModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.ConvTranspose2d(
                    2, 3, 3, stride=2, padding=1, output_padding=1
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.conv(x)

        example = torch.randn(1, 2, 4, 4, device="musa")
        module = ConvTranspose2dModel().eval().to("musa")
        exported = export(module, (example,))
        result = MusaBackend.preprocess(
            exported,
            [MusaBackend.generate_method_name_compile_spec("forward")],
        )

        self.assertGreater(len(result.processed_bytes), 0)
        self.assertIn("aoti_musa_blob", result.data_store_output.external_data)
