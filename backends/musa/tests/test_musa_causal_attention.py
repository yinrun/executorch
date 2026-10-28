"""Valid-prefix BF16 attention metadata, export, and eager contracts."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import torch

from executorch.backends.musa import MusaBackend, MusaPartitioner
from executorch.backends.musa.causal_attention import CAUSAL_ATTENTION, CAUSAL_ATTENTION_OUT
from executorch.exir import to_edge_transform_and_lower
from executorch.extension.export_util.utils import save_pte_program
from examples.musa import aoti_qwen35_export as exporter
from examples.musa.aoti_model_acceptance import _serialization_schema, _write_tensor
from examples.musa.aoti_qwen35_validate import runtime_libraries


class AttentionModule(torch.nn.Module):
    def forward(self, query, key, value, position, mask):
        return CAUSAL_ATTENTION(query, key, value, position, mask)


def inputs(device, capacity=64):
    return (
        torch.zeros((1, 8, 1, 256), device=device, dtype=torch.bfloat16),
        torch.zeros((1, 8, capacity, 256), device=device, dtype=torch.bfloat16),
        torch.ones((1, 8, capacity, 256), device=device, dtype=torch.bfloat16),
        torch.tensor([capacity - 1], device=device, dtype=torch.int64),
        torch.ones((1, capacity), device=device, dtype=torch.bool),
    )


class TestMusaCausalAttentionMetadata(unittest.TestCase):
    def test_generated_headers_exist_under_direct_and_package_imports(self):
        from executorch.backends.musa import musa_backend
        from executorch.backends.musa.causal_attention import HEADER

        for path in (
            HEADER,
            musa_backend._MUSA_BATCH_NORM_HEADER,
            musa_backend._MUSA_ACTIVATION_HEADER,
            musa_backend._MUSA_CONVOLUTION_HEADER,
        ):
            self.assertTrue(Path(path).is_file(), path)

    def test_meta_contract_and_fresh_output(self):
        for capacity in (64, 128, 256):
            tensors = inputs("meta", capacity)
            output = CAUSAL_ATTENTION(*tensors)
            self.assertEqual(output.shape, (1, 8, 1, 256))
            self.assertEqual(output.dtype, torch.bfloat16)
            self.assertTrue(output.is_contiguous())
            self.assertIsNot(output, tensors[0])
            self.assertIs(CAUSAL_ATTENTION_OUT(*tensors, out=output), output)

    def test_invalid_dtype_geometry_and_control_metadata(self):
        base = inputs("meta")
        for index, replacement in (
            (0, base[0].float()),
            (1, torch.empty((1, 8, 32, 256), device="meta", dtype=torch.bfloat16)),
            (2, torch.empty((1, 8, 128, 256), device="meta", dtype=torch.bfloat16)),
            (3, base[3].int()),
            (4, base[4].float()),
        ):
            with self.subTest(index=index), self.assertRaises(RuntimeError):
                values = list(base)
                values[index] = replacement
                CAUSAL_ATTENTION(*values)

    def test_export_preserves_opaque_runtime_length(self):
        exported = torch.export.export(AttentionModule(), inputs("meta"))
        targets = [node.target for node in exported.graph.nodes if node.op == "call_function"]
        self.assertEqual(targets, [CAUSAL_ATTENTION])

    def test_cli_guards_and_opt_in_metadata(self):
        for extra, accepted in (
            ([], True),
            (["--valid-length-attention"], False),
            (["--valid-length-attention", "--dtype", "bf16"], True),
            (["--valid-length-attention", "--dtype", "bf16", "--context-length", "512"], False),
        ):
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "export"
                argv = ["export", "--checkpoint-dir", directory, "--output-dir", str(output), *extra]
                with mock.patch("sys.argv", argv), mock.patch.object(exporter, "export_model") as export:
                    if accepted:
                        exporter.main()
                        self.assertEqual(export.call_args.args[0].valid_length_attention, bool(extra))
                    else:
                        with self.assertRaises(SystemExit):
                            exporter.main()
                        export.assert_not_called()
                        self.assertFalse(output.exists())


@unittest.skipUnless(hasattr(torch, "musa") and torch.musa.is_available(), "requires MUSA")
class TestMusaCausalAttentionEager(unittest.TestCase):
    def test_valid_prefix_excludes_nonfinite_future_and_preserves_inputs(self):
        with torch.no_grad(), torch.backends.mudnn.flags(allow_tf32=False):
            for capacity in (64, 128, 256):
                query, key, value, position, mask = inputs("musa", capacity)
                for length in (1, 3, capacity - 1, capacity):
                    position.fill_(length - 1)
                    mask.copy_(torch.arange(capacity, device="musa").reshape(1, -1) < length)
                    key.zero_()
                    value.fill_(1)
                    key[..., length:, :] = float("nan")
                    value[..., length:, :] = float("nan")
                    before = [tensor.contiguous().view(torch.uint8).cpu().clone() for tensor in (query, key, value, position, mask)]
                    scores = query @ key[..., :length, :].contiguous().transpose(-2, -1)
                    expected = torch.softmax((scores * 0.0625).float(), dim=-1).to(torch.bfloat16) @ value[..., :length, :].contiguous()
                    output = CAUSAL_ATTENTION(query, key, value, position, mask)
                    self.assertTrue(torch.equal(output, expected))
                    self.assertTrue(torch.isfinite(output).all().item())
                    for actual, saved in zip((query, key, value, position, mask), before):
                        self.assertTrue(torch.equal(actual.contiguous().view(torch.uint8).cpu(), saved))

    def test_invalid_positions_and_noncausal_masks(self):
        query, key, value, position, mask = inputs("musa")
        for index in (-1, 64):
            position.fill_(index)
            with self.assertRaisesRegex(ValueError, "outside"):
                CAUSAL_ATTENTION(query, key, value, position, mask)
        position.fill_(31)
        with self.assertRaisesRegex(ValueError, "causal prefix"):
            CAUSAL_ATTENTION(query, key, value, position, mask)

    def test_output_alias_rejection_and_disjoint_pool_views(self):
        tensors = list(inputs("musa"))
        with self.assertRaisesRegex(ValueError, "overlap"):
            CAUSAL_ATTENTION_OUT(*tensors, out=tensors[0])
        pool = torch.zeros(4096, device="musa", dtype=torch.bfloat16)
        tensors[0] = pool[:2048].reshape(1, 8, 1, 256)
        output = pool[2048:].reshape(1, 8, 1, 256)
        self.assertEqual(CAUSAL_ATTENTION_OUT(*tensors, out=output).data_ptr(), output.data_ptr())
        self.assertTrue(torch.equal(output, torch.ones_like(output)))


@unittest.skipUnless(
    hasattr(torch, "musa") and torch.musa.is_available() and os.environ.get("EXECUTORCH_MUSA_RUNNER"),
    "requires MUSA and EXECUTORCH_MUSA_RUNNER",
)
class TestMusaCausalAttentionRunner(unittest.TestCase):
    def test_exported_operator_accepts_runtime_lengths(self):
        runner = Path(os.environ["EXECUTORCH_MUSA_RUNNER"]).resolve()
        environment = {
            "PATH": "/usr/bin:/bin", "LD_DEBUG": "libs", "LD_BIND_NOW": "1",
            "LD_LIBRARY_PATH": ":".join(str(path) for path in (
                runner.parent / "backends/cuda", runner.parent / "extension/cuda",
                Path(os.environ.get("MUSA_HOME", "/usr/local/musa")) / "lib",
            )),
        }
        with torch.no_grad(), torch.backends.mudnn.flags(allow_tf32=False):
            for capacity in (64, 128, 256):
                for planned in (False, True):
                    tensors = inputs("musa", capacity)
                    exported = torch.export.export(AttentionModule(), tensors)
                    with tempfile.TemporaryDirectory() as directory:
                        root = Path(directory)
                        with _serialization_schema(), torch._inductor.config.patch(memory_planning=planned, memory_pool="combined"):
                            specs = [MusaBackend.generate_method_name_compile_spec("forward")]
                            manager = to_edge_transform_and_lower(exported, partitioner=[MusaPartitioner(specs)]).to_executorch()
                            pte = Path(save_pte_program(manager, "attention", directory))
                        self.assertIn(b"aoti_torch_musa_causal_attention_out\0", pte.read_bytes())
                        for length in (1, min(90, capacity), capacity):
                            tensors[3].fill_(length - 1)
                            tensors[4].copy_(torch.arange(capacity, device="musa").reshape(1, -1) < length)
                            paths = [root / f"input-{index}.bin" for index in range(len(tensors))]
                            for tensor, path in zip(tensors, paths):
                                _write_tensor(tensor, path)
                            _write_tensor(CAUSAL_ATTENTION(*tensors), root / "expected.bin")
                            result = subprocess.run([
                                str(runner), f"--model_path={pte}", f"--data_path={root / 'aoti_musa_blob.ptd'}",
                                f"--inputs={','.join(map(str, paths))}", f"--output_file={root / 'output'}",
                                "--print_output=none", "--num_executions=3", "--musa_cache_bytes=536870912",
                            ], env=environment, capture_output=True, text=True, timeout=90)
                            self.assertEqual(result.returncode, 0, result.stderr)
                            self.assertEqual((root / "output-0.bin").read_bytes(), (root / "expected.bin").read_bytes())
                            libraries = runtime_libraries(result.stderr)
                            self.assertTrue(libraries)
                            self.assertFalse(any("libtorch" in path or "libpython" in path for path in libraries))
                            stats = [json.loads(line.removeprefix("MUSA ALLOCATION CACHE ")) for line in result.stderr.splitlines() if line.startswith("MUSA ALLOCATION CACHE {")]
                            self.assertEqual(len(stats), 1)
                            for name in ("cached_bytes", "cached_blocks", "live_blocks"):
                                self.assertEqual(stats[0][name], 0)


if __name__ == "__main__":
    unittest.main()
