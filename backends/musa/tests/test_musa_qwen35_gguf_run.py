import argparse
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

from PIL import Image
import torch

from examples.musa import aoti_qwen35_gguf_run as runner


class TestQwen35RawInputHost(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.checkpoint = self.root / "checkpoint"
        self.checkpoint.mkdir()
        self.config = {
            "text_config": {
                "hidden_size": 2560,
                "num_hidden_layers": 32,
                "vocab_size": 248320,
                "eos_token_id": 248044,
            }
        }
        (self.checkpoint / "config.json").write_text(json.dumps(self.config))
        self.tokenizer = mock.Mock(eos_token_id=248044)
        self.tokenizer.apply_chat_template.return_value = "rendered prompt"
        self.tokenizer.decode.return_value = "decoded answer"
        self.processor = mock.Mock(return_value={"input_ids": torch.tensor([[42, 43]])})
        self.output = self.root / "output"

    def prepare(
        self, images=None, prompt="Explain heterogeneous computing", maximum=160
    ):
        return runner.prepare_request(
            self.checkpoint,
            prompt,
            images or [],
            self.output,
            maximum,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )

    def image_inputs(self, grids):
        path = self.root / "image.png"
        Image.new("RGB", (32, 32), "red").save(path)
        tokens = [42]
        count = 0
        for _, height, width in grids:
            tokens.extend([248056] * (height * width // 4))
            tokens.append(43)
            count += height * width
        self.processor.return_value = dict(
            input_ids=torch.tensor([tokens]),
            image_grid_thw=torch.tensor(grids),
            pixel_values=torch.zeros(count, 1536, dtype=torch.float32),
        )
        return [path] * len(grids)

    def test_text_fixture_and_template(self):
        metadata, tokenizer = self.prepare()
        self.assertIs(tokenizer, self.tokenizer)
        self.assertEqual(metadata["prompt_token_ids"], [42, 43])
        self.assertEqual(
            (self.output / "case.tokens.bin").read_bytes(), struct.pack("<2q", 42, 43)
        )
        self.assertEqual(
            (self.output / "case.positions.bin").read_bytes(),
            struct.pack("<6q", 0, 1, 0, 1, 0, 1),
        )
        self.assertEqual((self.output / "case.expected.bin").read_bytes(), b"")
        self.assertEqual((self.output / "case.embeddings.bin").read_bytes(), b"")
        self.assertFalse(
            self.tokenizer.apply_chat_template.call_args.kwargs["enable_thinking"]
        )
        self.assertIsNone(self.processor.call_args.kwargs["images"])

    def test_two_images_preserve_order_and_unique_patch_files(self):
        images = self.image_inputs([(1, 30, 40), (1, 20, 40)])
        metadata, _ = self.prepare(images)
        self.assertEqual(
            [image["component"] for image in metadata["images"]],
            ["vision1200", "vision800"],
        )
        self.assertEqual(
            [image["feature_rows"] for image in metadata["images"]], [300, 200]
        )
        self.assertEqual(
            [row for row in metadata["image_rows"] if row >= 0], list(range(500))
        )
        self.assertEqual(
            (self.output / "image-0-patches.bin").stat().st_size, 1200 * 1536 * 4
        )
        self.assertEqual(
            (self.output / "image-1-patches.bin").stat().st_size, 800 * 1536 * 4
        )

    def test_rejects_unsupported_grid_before_writing(self):
        images = self.image_inputs([(1, 40, 30)])
        with self.assertRaisesRegex(ValueError, "grids"):
            self.prepare(images)
        self.assertFalse(self.output.exists())

    def test_rejects_nonfinite_or_wrong_patch_dtype(self):
        images = self.image_inputs([(1, 20, 40)])
        self.processor.return_value["pixel_values"][0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            self.prepare(images)
        self.processor.return_value["pixel_values"] = torch.zeros(
            800, 1536, dtype=torch.float64
        )
        with self.assertRaisesRegex(ValueError, "FP32"):
            self.prepare(images)
        self.assertFalse(self.output.exists())

    def test_refuses_existing_output_without_overwriting(self):
        self.output.mkdir()
        marker = self.output / "keep"
        marker.write_text("unchanged")
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(marker.read_text(), "unchanged")

    def test_rejects_invalid_prompt_count_or_model(self):
        with self.assertRaises(ValueError):
            self.prepare(prompt=" ")
        with self.assertRaises(ValueError):
            self.prepare(maximum=True)
        self.processor.return_value["input_ids"] = torch.full((1, 4097), 42)
        with self.assertRaisesRegex(ValueError, "4096"):
            self.prepare()
        self.config["text_config"]["hidden_size"] = 1024
        (self.checkpoint / "config.json").write_text(json.dumps(self.config))
        with self.assertRaisesRegex(ValueError, "4B"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def args(self, prepare_only=False):
        return argparse.Namespace(
            checkpoint=self.checkpoint,
            prompt="A question",
            image=[],
            output=self.output,
            max_new_tokens=160,
            bundle=self.root / "bundle",
            timeout=10,
            prepare_only=prepare_only,
            dump_logits=False,
        )

    def test_prepare_only_never_starts_native_process(self):
        original = runner.prepare_request
        with mock.patch.object(
            runner,
            "prepare_request",
            side_effect=lambda *args: original(
                *args, tokenizer=self.tokenizer, processor=self.processor
            ),
        ), mock.patch.object(runner.subprocess, "run") as native:
            result = runner.run_request(self.args(prepare_only=True))
        self.assertEqual(result["status"], "prepared_host_inputs_no_model_inference")
        native.assert_not_called()

    def test_native_failure_is_preserved(self):
        original = runner.prepare_request
        with mock.patch.object(
            runner,
            "prepare_request",
            side_effect=lambda *args: original(
                *args, tokenizer=self.tokenizer, processor=self.processor
            ),
        ), mock.patch.object(
            runner.subprocess, "run", return_value=mock.Mock(returncode=7)
        ):
            with self.assertRaisesRegex(RuntimeError, "exited 7"):
                runner.run_request(self.args())
        result = json.loads((self.output / "result.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertFalse((self.output / "answer.txt").exists())

    def test_native_ids_are_decoded_without_expected_token_feedback(self):
        original = runner.prepare_request

        def native(*args, **kwargs):
            result = dict(
                status="completed",
                expected_tokens_provided=False,
                model="Qwen3.5-4B-GGUF",
                controller="native_cpp",
                context_capacity=4096,
                runs=[
                    dict(
                        prompt_tokens=2,
                        generated_token_ids=[44, 248044],
                        stop_reason="eos",
                    )
                ],
            )
            (self.output / "native.json").write_text(json.dumps(result))
            return mock.Mock(returncode=0)

        with mock.patch.object(
            runner,
            "prepare_request",
            side_effect=lambda *args: original(
                *args, tokenizer=self.tokenizer, processor=self.processor
            ),
        ), mock.patch.object(runner.subprocess, "run", side_effect=native):
            result = runner.run_request(self.args())
        self.assertEqual(result["answer"], "decoded answer")
        self.assertEqual((self.output / "case.expected.bin").read_bytes(), b"")
        self.tokenizer.decode.assert_called_once_with(
            [44, 248044], skip_special_tokens=True
        )


if __name__ == "__main__":
    unittest.main()
