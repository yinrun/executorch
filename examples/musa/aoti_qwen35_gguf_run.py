"""Host preprocessing and native inference for the packaged Qwen3.5-4B model."""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import time

from examples.musa.aoti_qwen35_gguf_multimodal import image_token_positions


SUPPORTED_GRIDS = {(1, 30, 40): "vision1200", (1, 20, 40): "vision800"}


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_request(
    checkpoint,
    prompt,
    images,
    output,
    maximum_generated,
    tokenizer=None,
    processor=None,
):
    from PIL import Image
    import torch

    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    images = [Path(path).resolve() for path in images]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Expected a nonempty prompt")
    if type(maximum_generated) is not int or not 1 <= maximum_generated <= 4096:
        raise ValueError("Maximum generated tokens must be in [1, 4096]")
    config = json.loads((checkpoint / "config.json").read_text())
    text_config = config["text_config"]
    if (
        text_config["hidden_size"],
        text_config["num_hidden_layers"],
        text_config["vocab_size"],
    ) != (2560, 32, 248320):
        raise ValueError("Expected Qwen3.5-4B tokenizer/processor configuration")
    if any(not path.is_file() for path in images):
        raise FileNotFoundError("An input image does not exist")
    if output.exists():
        raise FileExistsError(output)
    if tokenizer is None or processor is None:
        from transformers import AutoProcessor, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    loaded_images = []
    for path in images:
        with Image.open(path) as image:
            loaded_images.append(image.convert("RGB"))
    try:
        inputs = processor(
            text=[rendered],
            images=loaded_images or None,
            return_tensors="pt",
            add_special_tokens=False,
        )
    finally:
        for image in loaded_images:
            image.close()
    tokens = inputs["input_ids"][0].tolist()
    grids = (
        [tuple(grid) for grid in inputs["image_grid_thw"].tolist()] if images else []
    )
    if len(grids) != len(images) or any(grid not in SUPPORTED_GRIDS for grid in grids):
        raise ValueError(
            "This package supports only still-image grids 1x30x40 and 1x20x40"
        )
    positions = image_token_positions(tokens, grids)
    eos = sorted({tokenizer.eos_token_id, text_config["eos_token_id"]})
    if any(type(token) is not int or not 0 <= token < 248320 for token in eos):
        raise ValueError("Invalid EOS token configuration")
    patches = inputs.get("pixel_values")
    if images:
        count = sum(grid[1] * grid[2] for grid in grids)
        if (
            not isinstance(patches, torch.Tensor)
            or patches.device.type != "cpu"
            or patches.dtype != torch.float32
            or tuple(patches.shape) != (count, 1536)
        ):
            raise ValueError("Expected CPU FP32 patches with shape [patch_count, 1536]")
        if not torch.isfinite(patches).all().item():
            raise ValueError("Nonfinite image patches")
    output.mkdir()
    prefix = output / "case"
    arrays = dict(
        tokens=tokens,
        positions=positions["position_ids"].flatten().tolist(),
        image_rows=positions["image_rows"],
        eos=eos,
        expected=[],
    )
    for suffix, values in arrays.items():
        Path(f"{prefix}.{suffix}.bin").write_bytes(
            struct.pack(f"<{len(values)}q", *values)
        )
    Path(f"{prefix}.embeddings.bin").write_bytes(b"")
    metadata = dict(
        status="prepared_host_inputs_no_model_inference",
        prompt=prompt,
        rendered_prompt=rendered,
        prompt_token_ids=tokens,
        prompt_token_count=len(tokens),
        position_ids=positions["position_ids"].tolist(),
        image_rows=positions["image_rows"],
        next_position=positions["next_position"],
        rope_delta=positions["rope_delta"],
        eos_token_ids=eos,
        context_capacity=4096,
        thinking=False,
        greedy=True,
        maximum_generated_tokens=maximum_generated,
        images=[],
        source_hashes={},
    )
    offset = 0
    for index, (path, grid) in enumerate(zip(images, grids, strict=True)):
        count = grid[1] * grid[2]
        patch_path = output / f"image-{index}-patches.bin"
        patch_path.write_bytes(
            patches[offset : offset + count]
            .contiguous()
            .numpy()
            .astype("<f4", copy=False)
            .tobytes()
        )
        offset += count
        metadata["images"].append(
            dict(
                path=str(path),
                sha256=file_hash(path),
                grid=list(grid),
                component=SUPPORTED_GRIDS[grid],
                patches=str(patch_path),
                patches_sha256=file_hash(patch_path),
                feature_rows=count // 4,
            )
        )
    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "vocab.json",
        "merges.txt",
        "video_preprocessor_config.json",
    ):
        path = checkpoint / name
        if path.is_file():
            metadata["source_hashes"][str(path)] = file_hash(path)
    (output / "prepared.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )
    return metadata, tokenizer


def run_request(args):
    started = time.monotonic()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    prepared, tokenizer = prepare_request(
        args.checkpoint, args.prompt, args.image, output, args.max_new_tokens
    )
    preparation_seconds = time.monotonic() - started
    if args.prepare_only:
        return prepared
    report = dict(
        status="running",
        model_inference="native_musa_subprocesses",
        host_preprocessing_only=True,
        preparation_seconds=preparation_seconds,
        vision=[],
        full_logit_accuracy_validated=False,
    )

    def execute(command, log):
        phase_start = time.monotonic()
        with log.open("x") as stream:
            result = subprocess.run(
                command, stdout=stream, stderr=stream, cwd=output, timeout=args.timeout
            )
        if result.returncode != 0:
            raise RuntimeError(f"Native runner exited {result.returncode}; see {log}")
        return time.monotonic() - phase_start

    try:
        bundle = args.bundle.resolve()
        features = []
        for index, image in enumerate(prepared["images"]):
            command = [
                str(bundle / image["component"] / "run.sh"),
                f"--inputs={image['patches']}",
                f"--output_file={output}/image-{index}",
                "--print_output=none",
            ]
            elapsed = execute(command, output / f"image-{index}.log")
            feature = (output / f"image-{index}-0.bin").read_bytes()
            if len(feature) != image["feature_rows"] * 2560 * 4:
                raise ValueError("Native vision produced an invalid feature size")
            features.append(feature)
            report["vision"].append(dict(command=command, process_seconds=elapsed))
        (output / "case.embeddings.bin").write_bytes(b"".join(features))
        command = [
            str(bundle / "text/run.sh"),
            f"--musa_qwen35_fixture={output}/case",
            f"--musa_qwen35_output={output}/native.json",
            f"--musa_qwen35_max_new_tokens={args.max_new_tokens}",
            "--musa_qwen35_runs=1",
            "--musa_qwen35_warmup_runs=0",
        ]
        if args.dump_logits:
            command.append(f"--musa_qwen35_dump_prefix={output}/logits")
        report["generation_command"] = command
        report["generation_process_seconds"] = execute(command, output / "text.log")
        native = json.loads((output / "native.json").read_text())
        if (
            native["status"] != "completed"
            or native["expected_tokens_provided"]
            or len(native["runs"]) != 1
            or native["model"] != "Qwen3.5-4B-GGUF"
            or native["controller"] != "native_cpp"
            or native["context_capacity"] != 4096
        ):
            raise ValueError("Invalid native generation report")
        run = native["runs"][0]
        if (
            run["prompt_tokens"] != prepared["prompt_token_count"]
            or not 1 <= len(run["generated_token_ids"]) <= args.max_new_tokens
        ):
            raise ValueError("Native generation does not match the prepared request")
        answer = tokenizer.decode(run["generated_token_ids"], skip_special_tokens=True)
        (output / "answer.txt").write_text(answer + "\n")
        report.update(
            status="completed_raw_input_native_request",
            generated_token_ids=run["generated_token_ids"],
            stop_reason=run["stop_reason"],
            answer=answer,
            request_seconds_after_module_import=time.monotonic() - started,
            timing_scope="host_preprocessing_native_loading_vision_generation_and_text_decoding_excludes_module_import",
        )
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        (output / "result.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dump-logits", action="store_true")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    result = run_request(args)
    print(result.get("answer", f"Prepared inputs: {args.output}"))


if __name__ == "__main__":
    main()
