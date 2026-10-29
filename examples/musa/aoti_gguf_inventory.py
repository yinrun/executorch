"""Audit actual GGUF tensor encodings without changing or requantizing weights."""

import argparse
from collections import Counter
import hashlib
import inspect
import json
import math
from pathlib import Path


ENCODINGS = {
    "F32": (1, 4),
    "F16": (1, 2),
    "BF16": (1, 2),
    "Q4_K": (256, 144),
    "Q5_K": (256, 176),
    "Q6_K": (256, 210),
}


def tensor_inventory(tensors):
    result = []
    names = set()
    for tensor in tensors:
        name = tensor.name
        if not name or name in names:
            raise ValueError(f"Invalid or duplicate tensor name: {name!r}")
        names.add(name)
        encoding = tensor.tensor_type.name
        if encoding not in ENCODINGS:
            raise ValueError(f"Unsupported GGUF encoding: {encoding}")
        shape = tuple(int(size) for size in tensor.shape)
        if not shape or any(size <= 0 for size in shape):
            raise ValueError(f"Invalid GGUF shape for {name}: {shape}")
        block_size, block_bytes = ENCODINGS[encoding]
        if shape[0] % block_size:
            raise ValueError(f"Incomplete quantization row for {name}")
        elements = math.prod(shape)
        if (
            tensor.n_elements != elements
            or tensor.n_bytes != elements // block_size * block_bytes
        ):
            raise ValueError(f"Inconsistent GGUF storage size for {name}")
        if tensor.data.nbytes != tensor.n_bytes:
            raise ValueError(f"Truncated GGUF tensor: {name}")
        result.append(
            dict(
                name=name,
                encoding=encoding,
                gguf_shape=list(shape),
                torch_shape=list(reversed(shape)),
                elements=elements,
                nbytes=int(tensor.n_bytes),
                sha256=hashlib.sha256(memoryview(tensor.data).cast("B")).hexdigest(),
            )
        )
    if not result:
        raise ValueError("GGUF file contains no tensors")
    return result


def inspect_file(path):
    import gguf

    path = path.resolve()
    before = path.stat()
    reader = gguf.GGUFReader(path)
    tensors = tensor_inventory(reader.tensors)
    metadata = {
        name: field.contents()
        for name, field in reader.fields.items()
        if not name.startswith(("tokenizer.", "GGUF."))
    }
    counts = Counter(tensor["encoding"] for tensor in tensors)
    storage = Counter()
    for tensor in tensors:
        storage[tensor["encoding"]] += tensor["nbytes"]
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"GGUF file changed during inspection: {path}")
    return dict(
        path=str(path),
        sha256=digest,
        nbytes=after.st_size,
        modified_time_ns=after.st_mtime_ns,
        metadata=metadata,
        tensor_counts=dict(counts),
        storage_bytes=dict(storage),
        elements=sum(tensor["elements"] for tensor in tensors),
        tensors=tensors,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing inventory")
    paths = [path.resolve() for path in args.models]
    if len(set(paths)) != len(paths):
        parser.error("Model paths must be unique")
    models = [inspect_file(path) for path in paths]
    import gguf

    sources = [Path(__file__).resolve(), Path(inspect.getfile(gguf.GGUFReader))]
    source_hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
    }
    with args.output.open("x") as target:
        json.dump(
            dict(
                status="inventoried",
                inference_validated=False,
                requantized=False,
                source_hashes=source_hashes,
                models=models,
            ),
            target,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
