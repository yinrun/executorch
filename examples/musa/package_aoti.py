"""Package a MUSA AOTI model for a Linux host with a compatible MUSA SDK/driver."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


LAUNCHER = """#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
MUSA_HOME=${MUSA_HOME:-/usr/local/musa}
export LD_LIBRARY_PATH="$ROOT/lib:$MUSA_HOME/lib"
unset LD_PRELOAD PYTHONPATH PYTHONHOME
exec "$ROOT/bin/executor_runner" --model_path="$ROOT/model/model.pte" --data_path="$ROOT/model/model.ptd" "$@"
"""


RUNTIME_LIBRARIES = (
    "libaoti_cuda_shims.so",
    "libextension_cuda.so",
    "libmusa.so",
    "libmusart.so",
    "libmublas.so",
    "libmudnn.so",
    "libmudnncxx.so",
    "libstdc++.so",
    "libgcc_s.so",
    "libc.so",
    "libm.so",
    "libdl.so",
    "libpthread.so",
    "librt.so",
    "libgomp.so",
    "libomp.so",
    "ld-linux-aarch64.so",
    "ld-linux-x86-64.so",
)


def inspect_elf(path: Path, readelf: str) -> dict:
    dynamic = subprocess.check_output([readelf, "--wide", "-d", str(path)], text=True)
    needed = re.findall(r"\(NEEDED\).*\[(.*?)\]", dynamic)
    if any("/" in name for name in needed):
        raise ValueError(f"Non-relocatable dependency path in {path.name}: {needed}")
    forbidden = [
        name
        for name in needed
        if name.startswith(("libtorch", "libc10", "libmusa_python", "libpython"))
    ]
    if forbidden:
        raise ValueError(
            f"{path.name} requires Torch/Python libraries {forbidden}; re-export "
            "with the runtime-only MUSA compiler and rebuild the runner"
        )
    unknown = [
        name
        for name in needed
        if not any(
            name == library or name.startswith(library + ".")
            for library in RUNTIME_LIBRARIES
        )
    ]
    if unknown:
        raise ValueError(f"Unbundled dependency in {path.name}: {unknown}")
    paths = re.findall(r"\((?:RPATH|RUNPATH)\).*\[(.*?)\]", dynamic)
    for search in paths:
        if any(
            entry not in ("$ORIGIN", "$ORIGIN/../lib") for entry in search.split(":")
        ):
            raise ValueError(
                f"Non-relocatable runtime search path in {path.name}: {search}"
            )
    return {"needed": needed, "rpath": paths}


def model_libraries(pte: Path, ptd: Path) -> dict[str, bytes]:
    from executorch.backends.cuda.cuda_weight_collector import (
        CUDA_AOTI_METADATA_MAGIC,
        decode_cuda_aoti_metadata,
    )
    from executorch.exir._serialize._cord import Cord
    from executorch.exir._serialize._program import deserialize_pte_binary
    from executorch.extension.flat_tensor.serialize.serialize import (
        FlatTensorSerializer,
    )

    pte_file = deserialize_pte_binary(pte.read_bytes())
    program = pte_file.program
    payload = FlatTensorSerializer().deserialize(Cord(ptd.read_bytes()))
    keys = set()
    for plan in program.execution_plan:
        for delegate in plan.delegates:
            if delegate.id != "MusaBackend":
                raise ValueError(f"Unsupported packaged delegate: {delegate.id}")
            data = program.backend_delegate_data[delegate.processed.index].data
            if data.startswith(CUDA_AOTI_METADATA_MAGIC):
                keys.update(
                    variant.so_blob_key
                    for variant in decode_cuda_aoti_metadata(data).variants
                )
            else:
                keys.add(data.decode("utf-8").splitlines()[0])
    if not keys:
        raise ValueError("The program has no MusaBackend libraries")
    libraries = {}
    inline = pte_file.named_data
    for key in sorted(keys):
        if inline is not None and key in inline.pte_data:
            if key in payload.named_data:
                raise ValueError(f"Ambiguous inline/external library: {key}")
            libraries[key] = bytes(inline.buffers[inline.pte_data[key].buffer_index])
        else:
            libraries[key] = bytes(
                payload.buffers[payload.named_data[key].buffer_index]
            )
    return libraries


def package(
    build_dir: Path, pte: Path, ptd: Path, output: Path, patchelf: str, readelf: str
) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = {
        "bin/executor_runner": build_dir / "executor_runner",
        "lib/libaoti_cuda_shims.so": build_dir / "backends/cuda/libaoti_cuda_shims.so",
        "lib/libextension_cuda.so": build_dir / "extension/cuda/libextension_cuda.so",
        "model/model.pte": pte,
        "model/model.ptd": ptd,
    }
    for source in sources.values():
        if not source.is_file():
            raise FileNotFoundError(source)
    manifest = {
        "version": 1,
        "platform": "linux",
        "requirements": "Matching host architecture, MUSA SDK and GPU driver; no Python/Torch runtime",
        "elf": {},
        "files": {},
    }
    with tempfile.TemporaryDirectory(
        prefix=".musa-package-", dir=output.parent
    ) as temporary:
        root = Path(temporary) / "bundle"
        root.mkdir()
        for name, source in sources.items():
            destination = root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if name.startswith(("bin/", "lib/")):
                rpath = "$ORIGIN/../lib" if name.startswith("bin/") else "$ORIGIN"
                subprocess.run(
                    [patchelf, "--set-rpath", rpath, str(destination)], check=True
                )
                manifest["elf"][name] = inspect_elf(destination, readelf)
        for index, (key, data) in enumerate(
            model_libraries(root / "model/model.pte", root / "model/model.ptd").items()
        ):
            library = Path(temporary) / f"model-{index}.so"
            library.write_bytes(data)
            manifest["elf"][f"model:{key}"] = inspect_elf(library, readelf)
        launcher = root / "run.sh"
        launcher.write_text(LAUNCHER)
        launcher.chmod(0o755)
        for path in sorted(root.rglob("*")):
            if path.is_file():
                data = path.read_bytes()
                manifest["files"][str(path.relative_to(root))] = {
                    "nbytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        root.rename(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--pte", type=Path, required=True)
    parser.add_argument("--ptd", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--patchelf", default="patchelf")
    parser.add_argument("--readelf", default="readelf")
    args = parser.parse_args()
    package(
        args.build_dir, args.pte, args.ptd, args.output_dir, args.patchelf, args.readelf
    )
    print(f"Packaged MUSA runtime: {args.output_dir}")


if __name__ == "__main__":
    main()
