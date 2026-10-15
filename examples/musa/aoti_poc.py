# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile

import torch
from torch import nn
from torch.export import export

from executorch.backends.musa import MusaBackend
from executorch.backends.cuda.cuda_weight_collector import decode_cuda_aoti_metadata


class AddOne(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1


def _extract_named_data(result, key: str) -> bytes:
    entry = result.data_store_output.pte_data[key]
    return bytes(result.data_store_output.buffers[entry.buffer_index])


def main() -> None:
    if not torch.musa.is_available():
        raise RuntimeError("MUSA is not available")

    example = torch.randn(4, 8, device="musa")
    exported = export(AddOne().eval(), (example,))
    compile_spec = [MusaBackend.generate_method_name_compile_spec("forward")]
    result = MusaBackend.preprocess(exported, compile_spec)
    if result.processed_bytes.startswith(b"ETCUDAFQN0"):
        metadata = decode_cuda_aoti_metadata(result.processed_bytes)
        so_key = metadata.variants[0].so_blob_key
    else:
        so_key = result.processed_bytes.decode("utf-8").splitlines()[0]
    so_data = _extract_named_data(result, so_key)

    with tempfile.NamedTemporaryFile(suffix=".wrapper.so", delete=False) as output:
        output.write(so_data)
        so_path = output.name
    try:
        from torch_musa._export import aot_load

        compiled = aot_load(so_path, "musa:0")
        actual = compiled(example)
        expected = example + 1
        print(
            {
                "device": str(actual.device),
                "shape": tuple(actual.shape),
                "max_error": (actual - expected).abs().max().item(),
                "so_key": so_key,
                "external_data": list(result.data_store_output.external_data),
            }
        )
    finally:
        os.unlink(so_path)


if __name__ == "__main__":
    main()
