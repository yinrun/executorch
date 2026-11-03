"""Shared host prefix contracts and native mixed-device AOTI execution."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import torch

from executorch.backends.musa import MusaBackend, MusaPartitioner
from executorch.backends.musa.causal_attention import CAUSAL_ATTENTION
from executorch.backends.musa.causal_prefix_attention import (
    CAUSAL_PREFIX,
    PREFIX_ATTENTION,
)
from executorch.backends.musa.tests.test_musa_causal_attention import inputs
from executorch.exir import to_edge_transform_and_lower
from executorch.extension.export_util.utils import save_pte_program
from examples.musa.aoti_model_acceptance import _serialization_schema, _write_tensor


class SharedPrefixModule(torch.nn.Module):
    def forward(self, query, key, value, position, mask):
        control = CAUSAL_PREFIX(position, mask)
        first = PREFIX_ATTENTION(query, key, value, control)
        second_query = query * 0.5
        second = PREFIX_ATTENTION(second_query, key, value, control)
        return (
            first,
            second,
            CAUSAL_ATTENTION(query, key, value, position, mask),
            CAUSAL_ATTENTION(second_query, key, value, position, mask),
        )


class TestCausalPrefixMetadata(unittest.TestCase):
    def test_loader_requires_valid_length_mode_before_reading_weights(self):
        from examples.musa.aoti_qwen35_gguf import load_gguf_text

        with self.assertRaisesRegex(ValueError, "valid-length"):
            load_gguf_text("/does-not-exist.gguf", share_attention_prefix=True)

    def test_export_retains_host_snapshot_and_gpu_math(self):
        exported = torch.export.export(SharedPrefixModule(), inputs("meta", 64))
        prefixes = [
            node for node in exported.graph.nodes if node.target == CAUSAL_PREFIX
        ]
        attentions = [
            node for node in exported.graph.nodes if node.target == PREFIX_ATTENTION
        ]
        self.assertEqual(len(prefixes), 1)
        self.assertEqual(len(attentions), 2)
        for node in prefixes:
            self.assertEqual(node.meta["val"].device.type, "cpu")
            self.assertEqual(tuple(node.meta["val"].shape), (2,))
        for control in (
            torch.empty(2, dtype=torch.float32),
            torch.empty(3, dtype=torch.int64),
            torch.empty(2, dtype=torch.int64, device="meta"),
        ):
            with self.assertRaises(RuntimeError):
                PREFIX_ATTENTION(*inputs("meta", 64)[:3], control)


@unittest.skipUnless(
    hasattr(torch, "musa") and torch.musa.is_available(), "requires MUSA"
)
class TestCausalPrefixEager(unittest.TestCase):
    def test_qwen_attention_projection_gate_and_cache_match_legacy(self):
        from executorch.backends.musa.tests.test_musa_qwen35_gguf import metadata
        from executorch.examples.models.llama.attention import AttentionMHA
        from executorch.examples.models.llama.rope import Rope
        from examples.musa.aoti_qwen35_gguf import (
            GgufSharedPrefixAttention,
            GgufValidLengthSDPA,
            model_parameters,
        )

        args = model_parameters(metadata(), 64)
        args.dim = 32
        torch.manual_seed(3103)
        rope = Rope(args).to("musa")
        baseline = AttentionMHA(args, 3, rope).to("musa").eval()
        previous = baseline.SDPA
        baseline.SDPA = GgufValidLengthSDPA(
            previous.dim, previous.head_dim, previous.n_rep, previous.max_context_len
        )
        candidate = GgufSharedPrefixAttention(args, 3, rope).to("musa").eval()
        candidate.load_state_dict(baseline.state_dict())
        with torch.no_grad(), torch.backends.mudnn.flags(allow_tf32=False):
            for index in (0, 1, 7, 31):
                position = torch.tensor([index], device="musa")
                hidden = torch.randn(1, 1, args.dim, device="musa")
                cosine, sine = rope.get_freqs(position, 1)
                mask = (torch.arange(64, device="musa") <= index).reshape(1, -1)
                control = CAUSAL_PREFIX(position, mask)
                expected, _ = baseline(hidden, cosine, sine, input_pos=position)
                actual, _ = candidate(
                    hidden, cosine, sine, input_pos=position, causal_prefix=control
                )
                self.assertTrue(torch.equal(actual, expected))
                self.assertTrue(
                    torch.equal(candidate.kv_cache.k_cache, baseline.kv_cache.k_cache)
                )
                self.assertTrue(
                    torch.equal(candidate.kv_cache.v_cache, baseline.kv_cache.v_cache)
                )

    def test_exact_attention_and_independent_snapshots(self):
        torch.manual_seed(20261103)
        with torch.no_grad(), torch.backends.mudnn.flags(allow_tf32=False):
            for capacity, grouped in ((64, False), (65, True), (4096, True)):
                tensors = inputs("musa", capacity, grouped)
                query, key, value, position, mask = tensors
                query.normal_(std=0.25)
                key.normal_(std=0.25)
                value.normal_()
                for length in (1, 31, capacity):
                    position.fill_(length - 1)
                    mask.copy_(
                        (torch.arange(capacity, device="musa") < length).reshape(1, -1)
                    )
                    control = CAUSAL_PREFIX(position, mask)
                    self.assertEqual(control.tolist(), [length, capacity])
                    expected = CAUSAL_ATTENTION(*tensors)
                    self.assertTrue(
                        torch.equal(
                            PREFIX_ATTENTION(query, key, value, control), expected
                        )
                    )
                    position.fill_(-1)
                    self.assertEqual(control.tolist(), [length, capacity])
                    self.assertTrue(
                        torch.equal(
                            PREFIX_ATTENTION(query, key, value, control), expected
                        )
                    )
                    with self.assertRaises(ValueError):
                        CAUSAL_PREFIX(position, mask)

    def test_rejects_masks_lengths_and_aliases(self):
        query, key, value, position, mask = inputs("musa", 64)
        mask[0, 1] = False
        with self.assertRaises(ValueError):
            CAUSAL_PREFIX(position, mask)
        for prefix in ([0, 64], [-1, 64], [65, 64], [1, 65]):
            with self.assertRaises(ValueError):
                PREFIX_ATTENTION(query, key, value, torch.tensor(prefix))
        with self.assertRaises(ValueError):
            torch.ops.executorch_musa.prefix_attention.out(
                query, key, value, torch.tensor([64, 64]), out=query
            )


@unittest.skipUnless(
    os.environ.get("EXECUTORCH_MUSA_RUNNER")
    and hasattr(torch, "musa")
    and torch.musa.is_available(),
    "requires configured native MUSA runner",
)
class TestCausalPrefixNativeExport(unittest.TestCase):
    def test_shared_cpu_snapshot_in_native_aoti(self):
        runner = Path(os.environ["EXECUTORCH_MUSA_RUNNER"]).resolve()
        environment = dict(
            PATH="/usr/bin:/bin",
            LD_LIBRARY_PATH=f"{runner.parent}/backends/cuda:{runner.parent}/extension/cuda:/usr/local/musa/lib",
        )
        with torch.no_grad(), torch.backends.mudnn.flags(allow_tf32=False):
            for capacity, grouped in ((64, False), (4096, True)):
                tensors = inputs("musa", capacity, grouped)
                tensors[0].normal_(std=0.25)
                tensors[1].normal_(std=0.25)
                tensors[2].normal_()
                exported = torch.export.export(SharedPrefixModule(), tensors)
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    with _serialization_schema(), torch._inductor.config.patch(
                        memory_planning=True, memory_pool="combined"
                    ):
                        specs = [
                            MusaBackend.generate_method_name_compile_spec("forward")
                        ]
                        manager = to_edge_transform_and_lower(
                            exported, partitioner=[MusaPartitioner(specs)]
                        ).to_executorch()
                        pte = Path(
                            save_pte_program(manager, "shared-prefix", directory)
                        )
                    self.assertIn(
                        b"aoti_torch_musa_causal_prefix_out\0", pte.read_bytes()
                    )
                    self.assertIn(
                        b"aoti_torch_musa_prefix_attention_out\0", pte.read_bytes()
                    )
                    for length in (1, 33, capacity):
                        tensors[3].fill_(length - 1)
                        tensors[4].copy_(
                            (torch.arange(capacity, device="musa") < length).reshape(
                                1, -1
                            )
                        )
                        paths = [
                            root / f"input-{index}.bin" for index in range(len(tensors))
                        ]
                        for tensor, path in zip(tensors, paths):
                            _write_tensor(tensor, path)
                        result = subprocess.run(
                            [
                                str(runner),
                                f"--model_path={pte}",
                                f"--data_path={root}/aoti_musa_blob.ptd",
                                f"--inputs={','.join(map(str, paths))}",
                                f"--output_file={root}/output",
                                "--print_output=none",
                                "--num_executions=3",
                                "--musa_cache_bytes=536870912",
                            ],
                            env=environment,
                            capture_output=True,
                            text=True,
                            timeout=90,
                        )
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertEqual(
                            (root / "output-0.bin").read_bytes(),
                            (root / "output-2.bin").read_bytes(),
                        )
                        self.assertEqual(
                            (root / "output-1.bin").read_bytes(),
                            (root / "output-3.bin").read_bytes(),
                        )


if __name__ == "__main__":
    unittest.main()
