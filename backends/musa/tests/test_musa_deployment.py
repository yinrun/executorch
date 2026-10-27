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

from examples.musa.aoti_model_acceptance import (
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
