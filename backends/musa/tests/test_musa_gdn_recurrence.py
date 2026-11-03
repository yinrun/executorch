import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import torch

from executorch.backends.musa.gdn_recurrence import GDN_RECURRENCE


class Recurrence(torch.nn.Module):
    def forward(self, query, key, value, decay, beta, state):
        return GDN_RECURRENCE(query, key, value, decay, beta, state)


def tensors(heads, device):
    return tuple(
        torch.ones(shape, device=device)
        for shape in [(1, heads, 1, 128)] * 3
        + [(1, heads, 1)] * 2
        + [(1, heads, 128, 128)]
    )


class TestGdnMetadata(unittest.TestCase):
    def test_export_preserves_opaque_recurrence(self):
        for heads in (16, 32):
            exported = torch.export.export(Recurrence(), tensors(heads, "meta"))
            self.assertEqual(
                sum(node.target == GDN_RECURRENCE for node in exported.graph.nodes), 1
            )
            self.assertEqual(
                exported.module()(*tensors(heads, "meta")).numel(), heads * 16512
            )

    def test_rejects_unsupported_inputs(self):
        for heads in (1, 4, 64):
            with self.assertRaises(RuntimeError):
                GDN_RECURRENCE(*tensors(heads, "meta"))
        arguments = tensors(32, "meta")
        noncontiguous = torch.empty((1, 32, 1, 256), device="meta")[..., ::2]
        for replacement in (
            arguments[0].half(),
            arguments[0].expand(2, -1, -1, -1),
            noncontiguous,
        ):
            with self.assertRaises(RuntimeError):
                GDN_RECURRENCE(replacement, *arguments[1:])


@unittest.skipUnless(
    hasattr(torch, "musa") and torch.musa.is_available(), "requires MUSA"
)
class TestGdnEager(unittest.TestCase):
    def test_exact_updates_and_input_preservation(self):
        for heads in (16, 32):
            query, key, value, decay, beta, state = tensors(heads, "musa")
            query.mul_(1 / 32)
            key.mul_(1 / 16)
            value.mul_(0.5)
            decay.mul_(0.5)
            beta.mul_(0.75)
            state.mul_(1 / 64)
            packed = GDN_RECURRENCE(query, key, value, decay, beta, state)
            expected = 1 / 128 + (0.5 - 1 / 16) * 0.75 / 16
            torch.testing.assert_close(
                packed[: state.numel()],
                torch.full_like(state.reshape(-1), expected),
                atol=0,
                rtol=0,
            )
            torch.testing.assert_close(
                packed[state.numel() :],
                torch.full_like(value.reshape(-1), expected * 4),
                atol=0,
                rtol=0,
            )
            torch.testing.assert_close(
                state, torch.full_like(state, 1 / 64), atol=0, rtol=0
            )


@unittest.skipUnless(
    hasattr(torch, "musa")
    and torch.musa.is_available()
    and os.environ.get("EXECUTORCH_MUSA_GDN_RUNNER"),
    "requires a native GDN runner",
)
class TestGdnNative(unittest.TestCase):
    def test_exported_state_updates_match_eager(self):
        from executorch.backends.musa import MusaBackend, MusaPartitioner
        from executorch.exir import to_edge_transform_and_lower
        from executorch.extension.export_util.utils import save_pte_program
        from examples.musa.aoti_model_acceptance import (
            _serialization_schema,
            _write_tensor,
        )

        class Stateful(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "state", torch.zeros((1, 32, 128, 128), device="musa")
                )

            def forward(self, query, key, value, decay, beta):
                packed = GDN_RECURRENCE(query, key, value, decay, beta, self.state)
                self.state.copy_(packed[: self.state.numel()].reshape_as(self.state))
                return packed

        torch.manual_seed(1729)
        arguments = list(tensors(32, "musa")[:-1])
        arguments[0].normal_().mul_(0.01)
        arguments[1].normal_().mul_(0.05)
        arguments[2].normal_()
        arguments[3].fill_(0.9)
        arguments[4].fill_(0.7)
        model = Stateful().eval()
        with torch.no_grad(), tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for _ in range(3):
                expected = model(*arguments)
            expected = expected.cpu().numpy()
            model.state.zero_()
            with _serialization_schema(), torch._inductor.config.patch(
                memory_planning=True, memory_pool="combined"
            ):
                exported = torch.export.export(model, tuple(arguments))
                exported = exported.run_decompositions({})
                self.assertEqual(
                    set(exported.graph_signature.buffers_to_mutate.values()), {"state"}
                )
                specs = [MusaBackend.generate_method_name_compile_spec("forward")]
                program = to_edge_transform_and_lower(
                    exported, partitioner=[MusaPartitioner(specs)]
                ).to_executorch()
                pte = save_pte_program(program, "gdn_recurrence", directory)
            paths = [root / f"input-{index}.bin" for index in range(len(arguments))]
            for value, path in zip(arguments, paths):
                _write_tensor(value, path)
            process = subprocess.run(
                [
                    os.environ["EXECUTORCH_MUSA_GDN_RUNNER"],
                    f"--model_path={pte}",
                    f"--data_path={root}/aoti_musa_blob.ptd",
                    f'--inputs={",".join(map(str, paths))}',
                    f"--output_file={root}/output",
                    "--print_output=none",
                    "--num_executions=3",
                ],
                env=dict(
                    PATH="/usr/bin:/bin", LD_LIBRARY_PATH=os.environ["LD_LIBRARY_PATH"]
                ),
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            actual = np.fromfile(root / "output-0.bin", dtype=np.float32)
            self.assertEqual(actual.shape, expected.shape)
            self.assertTrue(np.isfinite(actual).all())
            np.testing.assert_allclose(actual, expected, atol=0.001, rtol=0.001)


if __name__ == "__main__":
    unittest.main()
