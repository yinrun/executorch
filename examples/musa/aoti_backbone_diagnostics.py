"""Measure MobileNet stage errors against MUSA eager outputs.

Export every backbone stage as an output, or use --stage to isolate one stage
fed by its eager predecessor. Retaining intermediate outputs can change fusion
and allocation, so these measurements supplement full-model acceptance.
Use --include-classifier to observe the complete model pipeline, or
--classifier-operators to isolate the head with an eager backbone input.
Use --module-path to observe children of a Sequential fed by its eager input.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.export import export

from executorch.backends.musa import MusaBackend, MusaPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.extension.export_util.utils import save_pte_program

from examples.musa.aoti_model_acceptance import (
    DeepLabV3MobileNet,
    DTYPES,
    _run_runner,
    _serialization_schema,
    _torchvision_mobilenet_v2,
    _write_tensor,
)


class BackboneStages(nn.Module):
    def __init__(self, stages: list[nn.Module]) -> None:
        super().__init__()
        self.stages = nn.ModuleList(stages)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs = []
        for stage in self.stages:
            image = stage(image)
            outputs.append(image.clone())
        return tuple(outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("deeplabv3_mobilenet_v3_large", "mobilenet_v2"),
        default="deeplabv3_mobilenet_v3_large",
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input", type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--stage", type=int)
    selection.add_argument("--block-operators", type=int)
    selection.add_argument("--stem-operators", action="store_true")
    selection.add_argument("--stem-operator", type=int, choices=(0, 1, 2))
    selection.add_argument("--include-classifier", action="store_true")
    selection.add_argument("--classifier-operators", action="store_true")
    selection.add_argument("--module-path")
    args = parser.parse_args()
    if args.batch_size < 1 or min(args.height, args.width) < 32:
        parser.error("batch size must be positive and spatial sizes at least 32")
    (args.output_dir / "result.json").unlink(missing_ok=True)
    if not torch.musa.is_available():
        raise RuntimeError("MUSA is not available")

    if args.model == "mobilenet_v2":
        model = _torchvision_mobilenet_v2()
        checkpoint_model = model
        backbone = model.features
    else:
        model = DeepLabV3MobileNet()
        checkpoint_model = model.model
        backbone = model.model.backbone
    checkpoint_model.load_state_dict(
        torch.load(args.weights, map_location="cpu", weights_only=True)
    )
    model = model.eval().to(device="musa", dtype=DTYPES[args.dtype])
    stages = list(backbone.children())
    if args.block_operators is not None and (
        not 0 <= args.block_operators < len(stages)
        or not (
            hasattr(stages[args.block_operators], "conv")
            if args.model == "mobilenet_v2"
            else hasattr(stages[args.block_operators], "block")
        )
    ):
        parser.error("--block-operators requires an inverted residual stage")
    stem_mode = args.stem_operators or args.stem_operator is not None
    if stem_mode:
        stages = list(stages[0].children())
    selected_stage = args.stem_operator if stem_mode else args.stage
    if selected_stage is not None and not 0 <= selected_stage < len(stages):
        parser.error(f"stage must be between 0 and {len(stages) - 1}")
    torch.manual_seed(args.seed)
    image_shape = (args.batch_size, 3, args.height, args.width)
    if args.input is None:
        image = torch.randn(image_shape, device="musa").to(DTYPES[args.dtype])
    else:
        payload = bytearray(args.input.read_bytes())
        element_size = torch.empty((), dtype=DTYPES[args.dtype]).element_size()
        required_bytes = math.prod(image_shape) * element_size
        if len(payload) < required_bytes or any(payload[required_bytes:]):
            parser.error("input must contain the requested shape and only zero padding")
        image = (
            torch.frombuffer(
                payload, dtype=DTYPES[args.dtype], count=math.prod(image_shape)
            )
            .reshape(image_shape)
            .to("musa")
        )
    torch.musa.synchronize()
    with torch.no_grad(), torch.backends.mudnn.flags(allow_tf32=False):
        if args.module_path is not None:
            selected_module = model.get_submodule(args.module_path)
            if not isinstance(selected_module, nn.Sequential):
                parser.error("--module-path requires a Sequential module")
            captured = []
            hook = selected_module.register_forward_pre_hook(
                lambda module, inputs: captured.append(inputs[0].detach().clone())
            )
            try:
                model(image)
            finally:
                hook.remove()
            if len(captured) != 1:
                parser.error("selected module must execute exactly once")
            image = captured[0]
            stages = list(selected_module.children())
        if args.include_classifier or args.classifier_operators:
            if args.model == "mobilenet_v2":
                classifier = [
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(1),
                    *model.classifier.children(),
                ]
            else:
                classifier = list(model.model.classifier.children()) + [
                    nn.Upsample(
                        size=(args.height, args.width),
                        mode="bilinear",
                        align_corners=False,
                    )
                ]
            if args.classifier_operators:
                for stage in stages:
                    image = stage(image)
                stages = classifier
            else:
                stages.extend(classifier)
        if args.block_operators is not None:
            for stage in stages[: args.block_operators]:
                image = stage(image)
            block = (
                stages[args.block_operators].conv
                if args.model == "mobilenet_v2"
                else stages[args.block_operators].block
            )
            stages = list(block.children())
        if selected_stage is not None:
            for stage in stages[:selected_stage]:
                image = stage(image)
            stage_indices = [selected_stage]
            module = BackboneStages([stages[selected_stage]])
        else:
            stage_indices = list(range(len(stages)))
            module = BackboneStages(stages)
        module.eval()
        original_input = image.detach().cpu().clone()
        expected = tuple(output.detach().cpu() for output in module(image))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_path = args.output_dir / "input.bin"
    _write_tensor(original_input, input_path)
    for stage_index, tensor in zip(stage_indices, expected):
        _write_tensor(tensor, args.output_dir / f"expected-stage-{stage_index}.bin")
    exported = export(module, (image,))
    compile_spec = [MusaBackend.generate_method_name_compile_spec("forward")]
    with _serialization_schema():
        manager = to_edge_transform_and_lower(
            exported, partitioner=[MusaPartitioner(compile_spec)]
        ).to_executorch()
        pte_path = Path(
            save_pte_program(manager, "backbone_diagnostics", str(args.output_dir))
        )
    completed = _run_runner(
        args.runner,
        pte_path,
        args.output_dir / "aoti_musa_blob.ptd",
        input_path,
        args.output_dir / "output",
    )
    (args.output_dir / "runner.log").write_text(completed.stdout + completed.stderr)
    measurements = []
    for output_index, (stage_index, reference) in enumerate(
        zip(stage_indices, expected)
    ):
        output_path = args.output_dir / f"output-{output_index}.bin"
        actual = torch.frombuffer(
            bytearray(output_path.read_bytes()), dtype=DTYPES[args.dtype]
        ).float()
        reference_flat = reference.flatten().float()
        if actual.numel() != reference_flat.numel():
            raise RuntimeError(f"Output size mismatch for stage {stage_index}")
        finite = torch.isfinite(actual) & torch.isfinite(reference_flat)
        difference = (actual[finite] - reference_flat[finite]).abs()
        measurements.append(
            {
                "stage": stage_index,
                "module_type": type(stages[stage_index]).__name__,
                "shape": list(reference.shape),
                "max_abs_error": (
                    difference.max().item() if difference.numel() else None
                ),
                "mean_abs_error": (
                    difference.mean().item() if difference.numel() else None
                ),
                "p99_abs_error": (
                    torch.quantile(difference, 0.99).item()
                    if difference.numel()
                    else None
                ),
                "actual_nan": torch.isnan(actual).sum().item(),
                "actual_inf": torch.isinf(actual).sum().item(),
                "reference_nan": torch.isnan(reference_flat).sum().item(),
                "reference_inf": torch.isinf(reference_flat).sum().item(),
            }
        )
    if args.module_path is not None:
        mode = "module_children"
    elif stem_mode:
        mode = "stem_operators"
    elif args.block_operators is not None:
        mode = "block_operators"
    elif args.include_classifier:
        mode = "full_model_stages"
    elif args.classifier_operators:
        mode = "classifier_operators"
    else:
        mode = "backbone_stages"
    report = {
        "model": args.model,
        "module_path": args.module_path,
        "dtype": args.dtype,
        "weights": str(args.weights),
        "seed": args.seed,
        "input": str(args.input) if args.input is not None else None,
        "image_shape": [args.batch_size, 3, args.height, args.width],
        "mode": mode,
        "block": args.block_operators,
        "isolated_stage": selected_stage,
        "measurements": measurements,
    }
    report_json = json.dumps(report, indent=2, allow_nan=False)
    (args.output_dir / "result.json").write_text(report_json + "\n")
    print(report_json)
    if any(
        measurement[key]
        for measurement in measurements
        for key in ("actual_nan", "actual_inf", "reference_nan", "reference_inf")
    ):
        raise RuntimeError("Non-finite values found; see result.json")


if __name__ == "__main__":
    main()
