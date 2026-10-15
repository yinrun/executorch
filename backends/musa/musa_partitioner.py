# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

from executorch.backends.aoti.aoti_partitioner import AotiPartitioner
from executorch.backends.musa.musa_backend import MusaBackend
from executorch.exir._warnings import experimental
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.passes.propagate_device_pass import TARGET_DEVICE_COMPILE_SPEC_KEY


@experimental(
    "MusaPartitioner is an experimental AOTInductor integration for ExecuTorch."
)
class MusaPartitioner(AotiPartitioner):
    """Partitioner for the initial MUSA AOTI compatibility path.

    ExecuTorch's current serialized device schema only has a CUDA GPU entry.
    Until a native MUSA entry is added, propagation uses ``cuda:0`` while the
    compiler and runtime device remain MUSA.
    """

    def __init__(self, compile_spec: List[CompileSpec]) -> None:
        has_target_device = any(
            spec.key == TARGET_DEVICE_COMPILE_SPEC_KEY for spec in compile_spec
        )
        if not has_target_device:
            compile_spec = list(compile_spec) + [
                CompileSpec(TARGET_DEVICE_COMPILE_SPEC_KEY, b"cuda:0")
            ]
        super().__init__("CudaBackend", compile_spec)
