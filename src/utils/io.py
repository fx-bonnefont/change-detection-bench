import json
import numpy as np
import torch
from pathlib import Path


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


def load_features(dat_path: Path, metadata: dict, part: str, period: str) -> torch.Tensor:
    key = f"features_{part}_{period}"
    section = metadata["sections"][key]
    arr = read_section(dat_path, section)
    return torch.from_numpy(np.array(arr))


def load_masks(output_dir: Path, part: str) -> torch.Tensor:
    arr = np.load(output_dir / f"masks_{part}.npy")
    return torch.from_numpy(arr)
