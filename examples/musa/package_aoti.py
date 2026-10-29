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
    symbols = subprocess.check_output([readelf, "--wide", "-Ws", str(path)], text=True)
    unresolved = sorted(
        set(re.findall(r"\bUND\s+(__triton_\w+_(?:start|end|size))\b", symbols))
    )
    if unresolved:
        raise ValueError(
            f"Unresolved embedded Triton binary symbols in {path.name}: {unresolved}; "
            "re-export and check compiler cache paths"
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
    build_dir: Path,
    pte: Path,
    ptd: Path,
    output: Path,
    patchelf: str,
    readelf: str,
    export_metadata: Path | None = None,
    musa_cache_bytes: int = 0,
) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if type(musa_cache_bytes) is not int or not 0 <= musa_cache_bytes < 2**64:
        raise ValueError("MUSA cache bytes must be an unsigned 64-bit integer")
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
    metadata = None
    if export_metadata is not None:
        payload = export_metadata.read_bytes()
        metadata = json.loads(payload)
        if not isinstance(metadata, dict) or not isinstance(
            metadata.get("artifacts"), dict
        ):
            raise ValueError("Export metadata must contain artifact hashes")
        packaged_artifacts = {}
        for source, name in ((pte, "model.pte"), (ptd, "model.ptd")):
            with source.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if metadata["artifacts"].get(source.name) != digest:
                raise ValueError(
                    f"Export metadata package hash mismatch: {source.name}"
                )
            packaged_artifacts[name] = digest
        metadata = {
            **metadata,
            "artifacts": packaged_artifacts,
            "source_export_metadata_sha256": hashlib.sha256(payload).hexdigest(),
        }
    manifest = {
        "version": 1,
        "platform": "linux",
        "requirements": "Matching host architecture and embedded-kernel GPU ISA, compatible MUSA SDK/driver; no Python/Torch runtime",
        "elf": {},
        "files": {},
        "runtime_defaults": {"musa_cache_bytes": musa_cache_bytes},
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
        launcher_text = LAUNCHER
        if musa_cache_bytes:
            launcher_text = launcher_text.replace(
                'exec "$ROOT/bin/executor_runner"',
                f'exec "$ROOT/bin/executor_runner" --musa_cache_bytes={musa_cache_bytes}',
            )
        launcher.write_text(launcher_text)
        launcher.chmod(0o755)
        if metadata is not None:
            for name, expected in metadata["artifacts"].items():
                with (root / "model" / name).open("rb") as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                if actual != expected:
                    raise ValueError("Model changed while copying into the package")
            (root / "model/export.json").write_text(
                json.dumps(metadata, indent=2) + "\n"
            )
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
    parser.add_argument("--export-metadata", type=Path)
    parser.add_argument(
        "--musa-cache-bytes",
        type=int,
        default=0,
        help="Opt into a bounded allocation cache in run.sh; requires a supporting runner",
    )
    parser.add_argument("--patchelf", default="patchelf")
    parser.add_argument("--readelf", default="readelf")
    args = parser.parse_args()
    package(
        args.build_dir,
        args.pte,
        args.ptd,
        args.output_dir,
        args.patchelf,
        args.readelf,
        args.export_metadata,
        args.musa_cache_bytes,
    )
    print(f"Packaged MUSA runtime: {args.output_dir}")


if __name__ == "__main__":
    main()
