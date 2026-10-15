# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import hashlib
import os
import typing
from typing import Any, Dict, List

import torch
from executorch.backends.aoti.aoti_backend import AotiBackend
from executorch.backends.cuda.cuda_weight_collector import (
    AOTI_DEVICE_TYPE_CPU,
    AOTI_DEVICE_TYPE_CUDA,
    CudaWeightCollector,
)
from executorch.exir._warnings import experimental
from executorch.exir.backend.backend_details import BackendDetails, PreprocessResult
from executorch.exir.backend.compile_spec_schema import CompileSpec


def _aoti_device_type_for_weight(tensor: torch.Tensor) -> int:
    if tensor.device.type in ("cuda", "musa"):
        return AOTI_DEVICE_TYPE_CUDA
    if tensor.device.type == "cpu":
        return AOTI_DEVICE_TYPE_CPU
    raise RuntimeError(f"Unsupported AOTI constant device for MUSA export: {tensor.device}")


@experimental(
    "MusaBackend is an experimental AOTInductor integration for ExecuTorch."
)
class MusaBackend(AotiBackend, BackendDetails):
    """AOTInductor backend for the MUSA PrivateUse1 device.

    The first implementation uses the existing CUDA AOTI weight metadata wire
    format. The runtime can therefore validate the export without introducing a
    new serialized device enum; native MUSA metadata is a follow-up task.
    """

    @classmethod
    def get_device_name(cls) -> str:
        return "musa"

    @classmethod
    def get_supported_fallback_kernels(cls) -> Dict[str, Any]:
        return {}

    @classmethod
    def get_decomposition_table(cls) -> Dict[Any, Any]:
        return {}

    @classmethod
    def get_custom_passes(cls, compile_specs: List[CompileSpec]) -> List[typing.Any]:
        return []

    @classmethod
    def save_data_externally(cls) -> bool:
        return True

    @classmethod
    def get_aoti_compile_options(
        cls, compile_specs: List[CompileSpec]
    ) -> Dict[str, typing.Any]:
        options: Dict[str, typing.Any] = {
            "loop_ordering_after_fusion": False,
            "emulate_precision_casts": True,
            "aot_inductor.embed_kernel_binary": True,
            "aot_inductor.package": True,
            "aot_inductor.package_constants_in_so": False,
            "aot_inductor.package_constants_on_disk": True,
            "max_autotune": False,
        }
        config = getattr(torch._inductor.config, "_config", {})
        return {key: value for key, value in options.items() if key in config}

    @classmethod
    def get_extra_aoti_compile_context_manager(
        cls, compile_specs: List[CompileSpec] | None = None
    ):
        return contextlib.nullcontext()

    @classmethod
    def materialize_weights_blob(
        cls, paths: Any, compile_specs: List[CompileSpec]
    ) -> Any:
        if not isinstance(paths, list):
            return paths

        from torch.export.pt2_archive._package_weights import Weights

        weights = [path for path in paths if isinstance(path, Weights)]
        if not weights:
            return paths
        if len(weights) != 1:
            raise RuntimeError(f"Expected one MUSA AOTI weights output, got {len(weights)}")

        so_path = next(
            (
                path
                for path in paths
                if isinstance(path, str) and path.endswith(".wrapper.so")
            ),
            None,
        )
        if so_path is None:
            raise RuntimeError(f"Expected a MUSA AOTI .wrapper.so output, got {paths}")

        blob_path = os.path.splitext(so_path)[0] + "_weights.blob"
        capture = CudaWeightCollector.current_capture()
        capture.artifact = capture.collector.materialize(
            weights[0], os.path.dirname(blob_path), capture.device_type_for_weight
        )
        with open(blob_path, "wb"):
            pass

        materialized = [path for path in paths if not isinstance(path, Weights)]
        materialized.append(blob_path)
        return materialized

    @classmethod
    def preprocess(
        cls,
        edge_program: Any,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        collector = CudaWeightCollector()
        with collector.capture(_aoti_device_type_for_weight) as capture:
            result = AotiBackend.preprocess.__func__(cls, edge_program, compile_specs)
        if capture.artifact is None:
            raise RuntimeError("MUSA AOTI did not return a structured Weights output")
        collector.add_preprocess_result(
            result,
            capture.artifact,
            cls.get_device_name(),
        )
        collector.finish()
        return result

