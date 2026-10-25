"""Set EXECUTORCH_MUSA_RUNNER and EXECUTORCH_MUSA_PATCHELF to test deployment."""

import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch

from examples.musa.aoti_model_acceptance import (
    export_model,
    _write_input,
    _write_tensor,
)
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
        for name, dtype in (("mlp", torch.float32), ("simple_cnn", torch.bfloat16)):
            with self.subTest(model=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                pte, ptd, _, _, model, shape = export_model(
                    name,
                    root / "export",
                    dtype,
                    None,
                    None,
                    batch_size=2,
                    dynamic_batch_max=3,
                )
                bundle = root / "package"
                manifest = package(runner.parent, pte, ptd, bundle, patchelf, "readelf")
                self.assertTrue(any(key.startswith("model:") for key in manifest["elf"]))
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
                        ],
                        cwd=root,
                        env=environment,
                        text=True,
                        capture_output=True,
                    )
                    self.assertEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )
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


if __name__ == "__main__":
    unittest.main()
