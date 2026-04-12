import json
from pathlib import Path

import numpy as np


def create_memmap(dat_path: Path):
    return open(dat_path, "wb")


def write_chunk(fp, array: np.ndarray, byte_offset: int) -> int:
    chunk = array.astype(array.dtype).ravel().view(np.uint8)
    fp.write(chunk.tobytes())
    return byte_offset + len(chunk)


def flush(fp):
    fp.flush()


def read_section(dat_path: Path, meta: dict) -> np.memmap:
    return np.memmap(
        dat_path,
        dtype=meta["dtype"],
        mode="r",
        offset=meta["offset"],
        shape=tuple(meta["shape"]),
    )


def load_metadata(json_path: Path) -> dict:
    with open(json_path) as f:
        return json.load(f)


def save_metadata(json_path: Path, metadata: dict):
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)
