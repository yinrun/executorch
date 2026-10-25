import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from examples.musa.package_aoti import inspect_elf, package


class TestMusaPackage(unittest.TestCase):
    def test_accepts_sdk_libraries_and_origin_paths(self):
        dynamic = "(NEEDED) [libmusart.so.5]\n(RUNPATH) [$ORIGIN/../lib]\n"
        with mock.patch("subprocess.check_output", return_value=dynamic):
            self.assertEqual(
                inspect_elf(Path("runner"), "readelf")["needed"], ["libmusart.so.5"]
            )

    def test_rejects_torch_and_python_dependencies(self):
        for name in (
            "libtorch_cpu.so",
            "libtorch_musa.so",
            "libmusa_python.so.2",
            "libc10.so",
            "libpython3.12.so",
        ):
            with self.subTest(name=name), mock.patch(
                "subprocess.check_output", return_value=f"(NEEDED) [{name}]\n"
            ):
                with self.assertRaisesRegex(ValueError, "requires Torch/Python"):
                    inspect_elf(Path("model.so"), "readelf")

    def test_rejects_non_relocatable_search_paths(self):
        for search in ("/tmp/build", "$ORIGIN:/usr/local/lib/python", ".", "$ORIGIN:"):
            with self.subTest(search=search), mock.patch(
                "subprocess.check_output", return_value=f"(RPATH) [{search}]\n"
            ):
                with self.assertRaisesRegex(ValueError, "Non-relocatable"):
                    inspect_elf(Path("model.so"), "readelf")

    def test_unknown_dependencies_are_not_assumed_standalone(self):
        with mock.patch(
            "subprocess.check_output", return_value="(NEEDED) [libcustom_ops.so]\n"
        ):
            with self.assertRaisesRegex(ValueError, "Unbundled dependency"):
                inspect_elf(Path("model.so"), "readelf")

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileExistsError):
                package(
                    root,
                    root / "model.pte",
                    root / "model.ptd",
                    root,
                    "patchelf",
                    "readelf",
                )

    def test_rejects_path_based_dependencies(self):
        with mock.patch(
            "subprocess.check_output", return_value="(NEEDED) [/tmp/lib.so]\n"
        ):
            with self.assertRaisesRegex(ValueError, "Non-relocatable dependency"):
                inspect_elf(Path("model.so"), "readelf")

    def test_failed_patch_does_not_publish_partial_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "executor_runner",
                "backends/cuda/libaoti_cuda_shims.so",
                "extension/cuda/libextension_cuda.so",
                "model.pte",
                "model.ptd",
            ):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"test")
            with mock.patch(
                "subprocess.run",
                side_effect=subprocess.CalledProcessError(1, "patchelf"),
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    package(
                        root,
                        root / "model.pte",
                        root / "model.ptd",
                        root / "package",
                        "patchelf",
                        "readelf",
                    )
            self.assertFalse((root / "package").exists())
            self.assertFalse(list(root.glob(".musa-package-*")))


if __name__ == "__main__":
    unittest.main()
