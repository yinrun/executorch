"""Set EXECUTORCH_MUSA_RUNNER and EXECUTORCH_MUSA_PATCHELF to test deployment."""

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch.export import Dim, export

from executorch.backends.musa import MusaBackend, MusaPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.extension.export_util.utils import save_pte_program

from backends.musa.tests.test_musa_activation_out import ActivationChain

from examples.musa.aoti_model_acceptance import (
    _serialization_schema,
    _snapshot_tensor,
    _write_input,
    _write_tensor,
    export_model,
)
from examples.musa.benchmark_aoti import validate_samples
from examples.musa.package_aoti import package


@unittest.skipUnless(
    torch.musa.is_available()
    and os.environ.get("EXECUTORCH_MUSA_RUNNER")
    and os.environ.get("EXECUTORCH_MUSA_PATCHELF"),
    "requires MUSA, EXECUTORCH_MUSA_RUNNER and EXECUTORCH_MUSA_PATCHELF",
)
class TestMusaDeployment(unittest.TestCase):
    def test_relocated_pooled_activation_outputs(self):
        runner = Path(os.environ["EXECUTORCH_MUSA_RUNNER"]).resolve()
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for strided in (False, True):
                with self.subTest(dtype=dtype, strided=strided):
                    values = torch.tensor(
                        [
                            -4,
                            -3,
                            -2,
                            -0.310546875,
                            0,
                            0.85546875,
                            0.99609375,
                            0.66796875,
                            1,
                            2,
                            3,
                            4,
                        ],
                        device="musa",
                        dtype=dtype,
                    ).reshape(1, 3, 4)
                    model = ActivationChain(strided).eval()
                    exported = export(
                        model,
                        (values.repeat(2, 1, 1),),
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
                            pte = Path(
                                save_pte_program(manager, "activations", directory)
                            )
                        for symbol in (
                            "aoti_torch_musa_hardswish_out",
                            "aoti_torch_musa_hardsigmoid_out",
                            "aoti_torch__alloc_from_pool",
                        ):
                            self.assertIn((symbol + "\0").encode(), pte.read_bytes())
                        bundle = root / "package"
                        package(
                            runner.parent,
                            pte,
                            root / "aoti_musa_blob.ptd",
                            bundle,
                            os.environ["EXECUTORCH_MUSA_PATCHELF"],
                            "readelf",
                        )
                        relocated = root / "relocated activations"
                        bundle.rename(relocated)
                        environment = {
                            "PATH": "/usr/bin:/bin",
                            "HOME": str(root),
                            "MUSA_HOME": os.environ.get("MUSA_HOME", "/usr/local/musa"),
                            "LD_BIND_NOW": "1",
                            "LD_DEBUG": "libs",
                        }
                        for batch in (1, 2, 3):
                            with self.subTest(batch=batch), torch.no_grad():
                                value = values.repeat(batch, 1, 1)
                                expected = tuple(
                                    _snapshot_tensor(tensor).flatten()
                                    for tensor in model(value)
                                )
                                _write_input(value, root / "input.bin", (3, 3, 4))
                                result = subprocess.run(
                                    [
                                        str(relocated / "run.sh"),
                                        f"--inputs={root / 'input.bin'}",
                                        f"--input_shapes={batch}x3x4",
                                        f"--output_file={root / 'output'}",
                                        "--num_executions=5",
                                        "--print_output=none",
                                    ],
                                    cwd=root,
                                    env=environment,
                                    capture_output=True,
                                    text=True,
                                    timeout=120,
                                )
                                self.assertEqual(
                                    result.returncode, 0, result.stdout + result.stderr
                                )
                                loaded = re.findall(
                                    r"calling init: (.+)", result.stderr
                                )
                                self.assertTrue(
                                    any(
                                        str(relocated / "lib/libaoti_cuda_shims.so")
                                        in path
                                        for path in loaded
                                    )
                                )
                                for path in loaded:
                                    self.assertFalse(
                                        any(
                                            name in path
                                            for name in (
                                                "libtorch",
                                                "libc10",
                                                "libpython",
                                                "libmusa_python",
                                                "site-packages",
                                                "dist-packages",
                                            )
                                        ),
                                        path,
                                    )
                                for index, reference in enumerate(expected):
                                    actual = torch.frombuffer(
                                        bytearray(
                                            (root / f"output-{index}.bin").read_bytes()
                                        ),
                                        dtype=dtype,
                                    )
                                    torch.testing.assert_close(
                                        actual, reference, atol=0, rtol=0
                                    )

    def test_relocated_runtime_without_torch(self):
        runner = Path(os.environ["EXECUTORCH_MUSA_RUNNER"]).resolve()
        patchelf = os.environ["EXECUTORCH_MUSA_PATCHELF"]
        for name, dtype, memory_planning in (
            ("mlp", torch.float32, False),
            ("simple_cnn", torch.bfloat16, False),
            ("mlp", torch.float32, True),
            ("simple_cnn", torch.bfloat16, True),
        ):
            with self.subTest(
                model=name, memory_planning=memory_planning
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with torch._inductor.config.patch(
                    memory_planning=memory_planning, memory_pool="combined"
                ):
                    pte, ptd, _, _, model, shape = export_model(
                        name,
                        root / "export",
                        dtype,
                        None,
                        None,
                        batch_size=2,
                        dynamic_batch_max=3,
                    )
                if memory_planning:
                    self.assertIn(b"aoti_torch__alloc_from_pool", pte.read_bytes())
                bundle = root / "package"
                manifest = package(runner.parent, pte, ptd, bundle, patchelf, "readelf")
                self.assertTrue(
                    any(key.startswith("model:") for key in manifest["elf"])
                )
                relocated = root / "relocated package"
                bundle.rename(relocated)
                environment = {
                    "PATH": "/usr/bin:/bin",
                    "HOME": str(root),
                    "MUSA_HOME": os.environ.get("MUSA_HOME", "/usr/local/musa"),
                    "LD_BIND_NOW": "1",
                    "LD_DEBUG": "libs",
                }
                for batch in (1, 2, 3):
                    timing = root / f"timing-{batch}.json"
                    benchmark_flags = (
                        [
                            f"--musa_benchmark_output={timing}",
                            "--musa_warmup_executions=2",
                            "--num_executions=3",
                        ]
                        if batch != 1
                        else []
                    )
                    actual_shape = (batch, *shape[1:])
                    value = torch.randn(actual_shape, device="musa", dtype=dtype)
                    _write_input(value, root / "input.bin", (3, *shape[1:]))
                    with torch.no_grad(), torch.backends.mudnn.flags(allow_tf32=False):
                        expected = _write_tensor(model(value), root / "expected.bin")
                    result = subprocess.run(
                        [
                            str(relocated / "run.sh"),
                            f"--inputs={root / 'input.bin'}",
                            f"--input_shapes={'x'.join(map(str, actual_shape))}",
                            f"--output_file={root / 'output'}",
                            "--print_output=summary",
                            *benchmark_flags,
                        ],
                        cwd=root,
                        env=environment,
                        text=True,
                        capture_output=True,
                        timeout=120,
                    )
                    self.assertEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )
                    if benchmark_flags:
                        validate_samples(json.loads(timing.read_text()), 2, 3)
                    else:
                        self.assertFalse(timing.exists())
                    loaded = re.findall(r"calling init: (.+)", result.stderr)
                    self.assertTrue(loaded)
                    self.assertTrue(
                        any(
                            "relocated package/lib/libaoti_cuda_shims.so" in path
                            for path in loaded
                        )
                    )
                    for path in loaded:
                        self.assertFalse(
                            any(
                                value in path
                                for value in (
                                    "libtorch",
                                    "libc10",
                                    "libmusa_python",
                                    "libpython",
                                    "site-packages",
                                    "dist-packages",
                                )
                            ),
                            path,
                        )
                    actual = torch.frombuffer(
                        bytearray((root / "output-0.bin").read_bytes()), dtype=dtype
                    ).reshape(expected.shape)
                    tolerance = 1e-4 if dtype == torch.float32 else 4e-2
                    torch.testing.assert_close(
                        actual, expected, atol=tolerance, rtol=tolerance
                    )
                benchmark = subprocess.run(
                    [
                        sys.executable,
                        "-S",
                        str(
                            Path(__file__).resolve().parents[3]
                            / "examples/musa/benchmark_aoti.py"
                        ),
                        "--package",
                        str(relocated),
                        "--inputs",
                        str(root / "input.bin"),
                        "--input-shapes",
                        "x".join(map(str, actual_shape)),
                        "--reference-output",
                        str(root / "output-0.bin"),
                        "--output-dir",
                        str(root / "benchmark results"),
                        "--warmup",
                        "2",
                        "--iterations",
                        "3",
                        "--repeats",
                        "2",
                    ],
                    env=environment,
                    cwd=root,
                    text=True,
                    capture_output=True,
                    timeout=180,
                )
                self.assertEqual(
                    benchmark.returncode, 0, benchmark.stdout + benchmark.stderr
                )
                report = json.loads(
                    (root / "benchmark results/summary.json").read_text()
                )
                self.assertEqual(report["status"], "passed")
                self.assertEqual(report["execute_ms"]["count"], 6)
                self.assertEqual(len(report["processes"]), 2)
                if name == "mlp":
                    for flags in (
                        ["--num_executions=0"],
                        ["--num_executions=4294967295", "--musa_warmup_executions=1"],
                        ["--server_mode"],
                    ):
                        rejected_report = root / "rejected.json"
                        rejected = subprocess.run(
                            [
                                str(relocated / "run.sh"),
                                f"--musa_benchmark_output={rejected_report}",
                                *flags,
                            ],
                            env=environment,
                            cwd=root,
                            capture_output=True,
                            timeout=30,
                        )
                        self.assertNotEqual(rejected.returncode, 0)
                        self.assertFalse(rejected_report.exists())


if __name__ == "__main__":
    unittest.main()
