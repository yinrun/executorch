"""Input and stopping contracts for independent BF16 reference generation."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from examples.musa import aoti_qwen35_generate_reference as generator

from examples.musa.aoti_qwen35_generate_reference import (
    generation_stop_reason,
    validate_prompt_tokens,
    validate_prompts,
)


class TestQwen35GenerateReference(unittest.TestCase):
    def test_cli_exposes_valid_length_attention_and_preserves_default(self):
        for flags in ([], ["--valid-length-attention", "--fuse-projections"]):
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                output = root / "references"

                def generate(args, report):
                    self.assertEqual(args.valid_length_attention, bool(flags))
                    self.assertEqual(args.fuse_projections, bool(flags))
                    report.update(status="complete", all_eager_byte_equal=True)

                command = [
                    "reference",
                    "--checkpoint-dir",
                    directory,
                    "--prompts",
                    str(root / "prompts.json"),
                    "--context-length",
                    "256",
                    "--output-dir",
                    str(output),
                    *flags,
                ]
                with mock.patch("sys.argv", command), mock.patch.object(
                    generator, "generate", side_effect=generate
                ):
                    generator.main()
                self.assertEqual(
                    json.loads((output / "manifest.json").read_text())["status"],
                    "complete",
                )

    def test_accepts_nonempty_unique_prompts(self):
        entries = [{"name": "zh_reasoning", "prompt": "如何区分事实与推断？"}]
        self.assertEqual(validate_prompts(entries), entries)

    def test_rejects_unsafe_or_duplicate_names(self):
        for name in (
            "",
            ".",
            "..",
            "../escape",
            "/absolute",
            "dir/name",
            "dir\\name",
            "a\0b",
            None,
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_prompts([{"name": name, "prompt": "Why?"}])
        with self.assertRaises(ValueError):
            validate_prompts([{"name": "same", "prompt": "Why?"}] * 2)

    def test_rejects_missing_text_and_malformed_lists(self):
        for entries in (
            [],
            {},
            None,
            [None],
            [{"name": "ok"}],
            [{"name": "ok", "prompt": " "}],
        ):
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                validate_prompts(entries)

    def test_context_and_vocabulary_boundaries(self):
        validate_prompt_tokens([0, 248319], 64)
        validate_prompt_tokens([1] * 255, 256)
        for tokens in ([], [1] * 256, [True], [-1], [248320], [1.0], (1,)):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                validate_prompt_tokens(tokens, 256)

    def test_eos_takes_precedence_at_context_boundary(self):
        self.assertEqual(generation_stop_reason(9, 9, 255, 256), "eos")
        self.assertEqual(generation_stop_reason(9, 9, 17, 256), "eos")

    def test_context_stop_does_not_extend_trajectory(self):
        self.assertEqual(generation_stop_reason(4, 9, 255, 256), "context_limit")
        self.assertIsNone(generation_stop_reason(4, 9, 254, 256))


if __name__ == "__main__":
    unittest.main()
