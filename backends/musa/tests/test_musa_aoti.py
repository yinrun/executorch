# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch import nn
from torch.export import Dim, export

from executorch.backends.cuda.cuda_weight_collector import (
    decode_cuda_aoti_metadata,
)
from executorch.backends.musa import MusaBackend
from executorch.backends.musa.musa_backend import (
    _decompose_scaled_dot_product_attention,
)


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

    def test_dynamic_embedding_aoti_compile(self) -> None:
        class EmbeddingModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = nn.Embedding(64, 16)
                self.classifier = nn.Linear(16, 8)

            def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
                return self.classifier(self.embedding(token_ids).mean(dim=1))

        example = torch.randint(64, (2, 4), device="musa", dtype=torch.int64)
        module = EmbeddingModel().eval().to("musa")
        exported = export(
            module, (example,), dynamic_shapes=({0: Dim("batch", min=1, max=3)},)
        )
        result = MusaBackend.preprocess(
            exported,
            [MusaBackend.generate_method_name_compile_spec("forward")],
        )

        self.assertGreater(len(result.processed_bytes), 0)
        self.assertIn("aoti_musa_blob", result.data_store_output.external_data)

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

    def test_transformer_encoder_aoti_compile(self) -> None:
        module = (
            nn.TransformerEncoderLayer(
                d_model=16,
                nhead=4,
                dim_feedforward=32,
                dropout=0.0,
                batch_first=True,
            )
            .eval()
            .to("musa")
        )
        example = torch.randn(1, 4, 16, device="musa")
        exported = export(module, (example,))
        result = MusaBackend.preprocess(
            exported,
            [MusaBackend.generate_method_name_compile_spec("forward")],
        )

        self.assertGreater(len(result.processed_bytes), 0)
        self.assertIn("aoti_musa_blob", result.data_store_output.external_data)

    def test_causal_gqa_aoti_compile(self) -> None:
        class CausalGqa(nn.Module):
            def forward(
                self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
            ) -> torch.Tensor:
                return torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, is_causal=True, enable_gqa=True
                )

        query = torch.randn(1, 4, 4, 8, device="musa")
        key = torch.randn(1, 2, 4, 8, device="musa")
        value = torch.randn(1, 2, 4, 8, device="musa")
        exported = export(CausalGqa().eval(), (query, key, value))
        result = MusaBackend.preprocess(
            exported,
            [MusaBackend.generate_method_name_compile_spec("forward")],
        )

        self.assertGreater(len(result.processed_bytes), 0)
        self.assertIn("aoti_musa_blob", result.data_store_output.external_data)

    def test_sdpa_decomposition(self) -> None:
        mask = torch.tensor(
            [[True, True, False, True, False]] * 3, device="musa"
        )
        options = (
            {"attn_mask": mask},
            {"attn_mask": mask.to(torch.float32) * -0.5},
            {"is_causal": True},
            {"scale": 0.3},
            {"enable_gqa": True},
        )
        for dtype, tolerance in (
            (torch.float32, 1e-4),
            (torch.float16, 1e-2),
            (torch.bfloat16, 4e-2),
        ):
            query = torch.randn(1, 4, 3, 8, device="musa", dtype=dtype)
            key = torch.randn(1, 4, 5, 8, device="musa", dtype=dtype)
            value = torch.randn(1, 4, 5, 8, device="musa", dtype=dtype)
            for kwargs in options:
                with self.subTest(dtype=dtype, kwargs=kwargs):
                    kwargs = dict(kwargs)
                    mask_tensor = kwargs.get("attn_mask")
                    if mask_tensor is not None and mask_tensor.dtype != torch.bool:
                        kwargs["attn_mask"] = mask_tensor.to(dtype)
                    current_key = key[:, :2] if kwargs.get("enable_gqa") else key
                    current_value = value[:, :2] if kwargs.get("enable_gqa") else value
                    expected = torch.nn.functional.scaled_dot_product_attention(
                        query, current_key, current_value, **kwargs
                    )
                    actual = _decompose_scaled_dot_product_attention(
                        query, current_key, current_value, **kwargs
                    )
                    torch.testing.assert_close(
                        actual, expected, atol=tolerance, rtol=tolerance
                    )
