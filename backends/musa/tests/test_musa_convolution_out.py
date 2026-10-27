"""Pooled convolution runner regressions; requires EXECUTORCH_MUSA_RUNNER."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch
from torch.export import Dim, export

from executorch.backends.musa import MusaBackend, MusaPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.extension.export_util.utils import save_pte_program

from examples.musa.aoti_model_acceptance import (
    MODEL_FACTORIES,
    TOLERANCES,
    _runner_environment,
    _serialization_schema,
    _snapshot_tensor,
    _write_input,
)


@unittest.skipUnless(
    hasattr(torch, "musa")
    and torch.musa.is_available()
    and os.environ.get("EXECUTORCH_MUSA_RUNNER"),
    "requires MUSA and EXECUTORCH_MUSA_RUNNER",
)
class TestMusaConvolutionOutRunner(unittest.TestCase):
    def _check_model(self, name: str) -> None:
        runner = Path(os.environ["EXECUTORCH_MUSA_RUNNER"])
        factory, shape = MODEL_FACTORIES[name]
        for dtype_name, dtype in (
            ("fp32", torch.float32),
            ("fp16", torch.float16),
            ("bf16", torch.bfloat16),
        ):
            with self.subTest(model=name, dtype=dtype_name):
                torch.manual_seed(123)
                model = factory().eval().to(device="musa", dtype=dtype)
                example = torch.randn(2, *shape[1:], device="musa", dtype=dtype)
                exported = export(
                    model,
                    (example,),
                    dynamic_shapes=({0: Dim("batch", min=1, max=3)},),
                )
                specs = [MusaBackend.generate_method_name_compile_spec("forward")]
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    with _serialization_schema(), torch._inductor.config.patch(
                        memory_planning=True, memory_pool="combined"
                    ):
                        manager = to_edge_transform_and_lower(
                            exported, partitioner=[MusaPartitioner(specs)]
                        ).to_executorch()
                        pte = Path(save_pte_program(manager, name, directory))
                    payload = pte.read_bytes()
                    self.assertIn(b"aoti_torch_musa_convolution_out\0", payload)
                    self.assertIn(b"aoti_torch__alloc_from_pool\0", payload)
                    self.assertNotIn(b"aoti_torch_musa_convolution\0", payload)
                    for batch in (1, 2, 3):
                        with self.subTest(
                            batch=batch
                        ), torch.no_grad(), torch.backends.mudnn.flags(
                            allow_tf32=False
                        ):
                            value = torch.randn(
                                batch, *shape[1:], device="musa", dtype=dtype
                            )
                            expected = _snapshot_tensor(model(value)).flatten()
                            self.assertGreater(expected.abs().max().item(), 0.01)
                            _write_input(value, root / "input.bin", (3, *shape[1:]))
                            prefix = root / f"output-{batch}"
                            result = subprocess.run(
                                [
                                    str(runner),
                                    f"--model_path={pte}",
                                    f"--data_path={root / 'aoti_musa_blob.ptd'}",
                                    f"--inputs={root / 'input.bin'}",
                                    "--input_shapes=" + "x".join(map(str, value.shape)),
                                    f"--output_file={prefix}",
                                    "--print_output=none",
                                    "--num_executions=5",
                                ],
                                env=_runner_environment(),
                                capture_output=True,
                                text=True,
                                timeout=90,
                            )
                            self.assertEqual(
                                result.returncode, 0, result.stdout + result.stderr
                            )
                            actual = torch.frombuffer(
                                bytearray(Path(f"{prefix}-0.bin").read_bytes()),
                                dtype=dtype,
                            )
                            torch.testing.assert_close(
                                actual,
                                expected,
                                atol=TOLERANCES[dtype_name],
                                rtol=TOLERANCES[dtype_name],
                            )

    def test_audio_decoder(self) -> None:
        self._check_model("simple_audio_decoder")

    def test_image_decoder(self) -> None:
        self._check_model("simple_decoder")

    def test_video_decoder(self) -> None:
        self._check_model("simple_video_decoder")


if __name__ == "__main__":
    unittest.main()
