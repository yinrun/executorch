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
    """Partition AOTI delegates with native MUSA memory-planning metadata."""

    def __init__(self, compile_spec: List[CompileSpec]) -> None:
        has_target_device = any(
            spec.key == TARGET_DEVICE_COMPILE_SPEC_KEY for spec in compile_spec
        )
        if not has_target_device:
            compile_spec = list(compile_spec) + [
                CompileSpec(TARGET_DEVICE_COMPILE_SPEC_KEY, b"musa:0")
            ]
        super().__init__(MusaBackend.__name__, compile_spec)
