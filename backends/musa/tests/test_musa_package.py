import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from examples.musa.package_aoti import inspect_elf, package


class TestMusaPackage(unittest.TestCase):
    def test_invalid_default_cache_is_rejected(self):
        for value in (-1, 2**64, True, 1.5):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with self.assertRaisesRegex(ValueError, "unsigned 64-bit"):
                    package(
                        root,
                        root / "model.pte",
                        root / "model.ptd",
                        root / "package",
                        "patchelf",
                        "readelf",
                        musa_cache_bytes=value,
                    )
                self.assertFalse((root / "package").exists())

    def test_launcher_cache_default_is_explicit_and_overridable(self):
        for budget in (0, 536870912):
            with self.subTest(
                budget=budget
            ), tempfile.TemporaryDirectory() as directory:
                arguments = self.make_metadata_inputs(Path(directory))
                with mock.patch("subprocess.run"), mock.patch(
                    "examples.musa.package_aoti.inspect_elf", return_value={}
                ), mock.patch(
                    "examples.musa.package_aoti.model_libraries",
                    return_value={"model": b"elf"},
                ):
                    manifest = package(*arguments, musa_cache_bytes=budget)
                launcher = (arguments[3] / "run.sh").read_text()
                self.assertEqual(
                    manifest["runtime_defaults"]["musa_cache_bytes"], budget
                )
                if budget:
                    flag = f"--musa_cache_bytes={budget}"
                    self.assertIn(flag, launcher)
                    self.assertLess(launcher.index(flag), launcher.index('"$@"'))
                else:
                    self.assertNotIn("--musa_cache_bytes", launcher)

    def test_rejects_unresolved_embedded_kernel_symbols(self):
        for suffix in ("start", "end", "size"):
            symbol = f"__triton_poi_fused_mul_8_{suffix}"
            dynamic = "(NEEDED) [libmusart.so.5]\n"
            symbols = f"  10: 0000000000000000 0 NOTYPE GLOBAL DEFAULT UND {symbol}\n"
            with self.subTest(suffix=suffix), mock.patch(
                "subprocess.check_output", side_effect=[dynamic, symbols]
            ):
                with self.assertRaisesRegex(ValueError, "Unresolved embedded Triton"):
                    inspect_elf(Path("model.so"), "readelf")

    def make_metadata_inputs(self, root):
        for name in (
            "executor_runner",
            "backends/cuda/libaoti_cuda_shims.so",
            "extension/cuda/libextension_cuda.so",
            "qwen35.pte",
            "aoti_musa_blob.ptd",
        ):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
        metadata = root / "result.json"
        metadata.write_text(
            json.dumps(
                {
                    "status": "exported",
                    "dtype": "bf16",
                    "context_length": 256,
                    "artifacts": {
                        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                        for name in ("qwen35.pte", "aoti_musa_blob.ptd")
                    },
                }
            )
        )
        return (
            root,
            root / "qwen35.pte",
            root / "aoti_musa_blob.ptd",
            root / "package",
            "patchelf",
            "readelf",
            metadata,
        )

    def test_export_metadata_uses_packaged_names_and_is_manifested(self):
        with tempfile.TemporaryDirectory() as directory:
            arguments = self.make_metadata_inputs(Path(directory))
            original = arguments[-1].read_bytes()
            with mock.patch("subprocess.run"), mock.patch(
                "examples.musa.package_aoti.inspect_elf", return_value={}
            ), mock.patch(
                "examples.musa.package_aoti.model_libraries",
                return_value={"model": b"elf"},
            ):
                manifest = package(*arguments)
            metadata_path = arguments[3] / "model/export.json"
            metadata = json.loads(metadata_path.read_text())
            self.assertEqual(metadata["dtype"], "bf16")
            self.assertEqual(metadata["context_length"], 256)
            self.assertEqual(set(metadata["artifacts"]), {"model.pte", "model.ptd"})
            for name, digest in metadata["artifacts"].items():
                self.assertEqual(
                    digest,
                    hashlib.sha256(
                        (arguments[3] / "model" / name).read_bytes()
                    ).hexdigest(),
                )
            self.assertEqual(
                metadata["source_export_metadata_sha256"],
                hashlib.sha256(original).hexdigest(),
            )
            self.assertEqual(
                manifest["files"]["model/export.json"]["sha256"],
                hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(arguments[-1].read_bytes(), original)

    def test_export_metadata_rejects_mismatched_model_hashes(self):
        for name in ("qwen35.pte", "aoti_musa_blob.ptd"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                arguments = self.make_metadata_inputs(Path(directory))
                metadata = json.loads(arguments[-1].read_text())
                metadata["artifacts"][name] = "0" * 64
                arguments[-1].write_text(json.dumps(metadata))
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    package(*arguments)
                self.assertFalse(arguments[3].exists())

    def test_export_metadata_rejects_missing_or_malformed_hash_mapping(self):
        for value in ([], {}, {"artifacts": []}):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                arguments = self.make_metadata_inputs(Path(directory))
                arguments[-1].write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError, "artifact hashes"):
                    package(*arguments)
                self.assertFalse(arguments[3].exists())

    def test_changed_model_copy_does_not_publish_package(self):
        with tempfile.TemporaryDirectory() as directory:
            arguments = self.make_metadata_inputs(Path(directory))
            copy = shutil.copy2

            def corrupt_model(source, destination):
                result = copy(source, destination)
                if source == arguments[1]:
                    destination.write_bytes(b"changed")
                return result

            with mock.patch("subprocess.run"), mock.patch(
                "examples.musa.package_aoti.inspect_elf", return_value={}
            ), mock.patch(
                "examples.musa.package_aoti.model_libraries",
                return_value={"model": b"elf"},
            ), mock.patch(
                "examples.musa.package_aoti.shutil.copy2", side_effect=corrupt_model
            ):
                with self.assertRaisesRegex(ValueError, "changed while copying"):
                    package(*arguments)
            self.assertFalse(arguments[3].exists())
            self.assertFalse(list(Path(directory).glob(".musa-package-*")))

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
