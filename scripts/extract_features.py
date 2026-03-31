import csv
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm
from PIL import Image
from src.data.raw_dataset import ImageDataset
from src.data.paths import get_paths
from src.utils.device import get_device
from src.utils.io import create_memmap, write_chunk, flush, save_metadata
from src.config import PRETRAINED_MODEL_NAMES, RAW_DATA_DIR
from src.config import model_dir, masks_dir, dat_path, metadata_path, mask_path, mask_stats_path, label_path


def load_model_and_processor(model_name: str, img_size: int):
    processor = AutoImageProcessor.from_pretrained(model_name)
    processor.size = {"height": img_size, "width": img_size}
    device = get_device()
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    dim = model.config.hidden_size
    n_tokens = 1 + (img_size // model.config.patch_size) ** 2
    return processor, model, device, dim, n_tokens


def extract_features_batch(loader, device, model, fp, byte_offset: int) -> int:
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Extraction..."):
            pixel_values = batch["pixel_values"].to(device) if isinstance(batch, dict) else batch.to(device)
            outputs = model(pixel_values)
            features = torch.cat([
                outputs.last_hidden_state[:, 0:1, :],
                outputs.last_hidden_state[:, 5:, :]
            ], dim=1).cpu().numpy().astype(np.float32)
            byte_offset = write_chunk(fp, features, byte_offset)
    flush(fp)
    return byte_offset


CHANNEL_VALUES = {0: range(10), 1: range(10), 2: range(3)}


def compute_mask_stats(masks: np.ndarray, part_name: str) -> list[dict]:
    rows = []
    for ch, values in CHANNEL_VALUES.items():
        channel_data = masks[:, :, :, ch]
        for v in values:
            rows.append({
                "split": part_name,
                "channel": ch,
                "value": v,
                "count": int(np.sum(channel_data == v)),
            })
    return rows


def save_masks(paths: dict):
    masks_dir().mkdir(parents=True, exist_ok=True)
    all_stats = []
    for part_name, part_data in paths.items():
        file_paths = part_data.get("mask", [])
        if not file_paths:
            continue
        out_path = mask_path(part_name)
        if out_path.exists():
            print(f"Masks {part_name} already exist, skipping.")
        else:
            masks = []
            for path in tqdm(file_paths, desc=f"Masks {part_name}"):
                with Image.open(path) as im:
                    masks.append(np.array(im, dtype=np.uint8))
            masks = np.stack(masks)
            np.save(out_path, masks)
            all_stats.extend(compute_mask_stats(masks, part_name))

    if all_stats:
        out_csv = mask_stats_path()
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["split", "channel", "value", "count"])
            writer.writeheader()
            writer.writerows(all_stats)
        print(f"Mask stats saved: {out_csv}")


def save_labels(paths: dict):
    for part_name, part_data in paths.items():
        if "mask" not in part_data:
            continue
        lbl_path = label_path(part_name)
        if lbl_path.exists():
            print(f"Labels {part_name} already exist, skipping.")
            continue
        masks = np.load(mask_path(part_name), mmap_mode="r")
        changed_pixels = (masks[:, :, :, 2] == 2)
        n_pixels = masks.shape[1] * masks.shape[2]
        binary = changed_pixels.any(axis=(1, 2)).astype(np.float32)
        ratio = changed_pixels.sum(axis=(1, 2)).astype(np.float32) / n_pixels
        labels = np.stack([binary, ratio], axis=1)
        np.save(lbl_path, labels)
        print(f"Labels {part_name} saved: {lbl_path}")


def extract_all(paths: dict, processor, model, device, fp, n_tokens: int, dim: int,
                batch_size: int, num_workers: int) -> dict:
    sections = {}
    byte_offset = 0

    for part_name, part_data in paths.items():
        for period, file_paths in part_data.items():
            if not file_paths or period not in ["2018", "2019"]:
                continue
            print(f"Processing {part_name} {period}")
            key = f"features_{part_name}_{period}"
            sections[key] = {"offset": byte_offset, "shape": [len(file_paths), n_tokens, dim], "dtype": "float32"}
            dataset = ImageDataset(file_paths, processor)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            byte_offset = extract_features_batch(loader, device, model, fp, byte_offset)

    return sections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=str(RAW_DATA_DIR))
    parser.add_argument("--model_size", type=str, default="small", choices=["small", "small-plus", "base", "large", "large-sat", "huge-plus", "7b", "7b-sat"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    paths = get_paths(args.data_path, args.limit)
    model_name = PRETRAINED_MODEL_NAMES[args.model_size]

    with Image.open(paths["train"]["2018"][0]) as img:
        img_size = img.size[0]

    save_masks(paths)
    save_labels(paths)

    if dat_path(args.model_size).exists():
        print(f"Features {args.model_size} already exist, skipping.")
    else:
        processor, model, device, dim, n_tokens = load_model_and_processor(model_name, img_size)
        out_dir = model_dir(args.model_size)
        out_dir.mkdir(parents=True, exist_ok=True)
        fp = create_memmap(dat_path(args.model_size))

        sections = extract_all(paths, processor, model, device, fp, n_tokens, dim, args.batch_size, args.num_workers)

        metadata = {
            "model_size": args.model_size,
            "dim": dim,
            "n_tokens": n_tokens,
            "sections": sections,
        }
        save_metadata(metadata_path(args.model_size), metadata)
        print(f"Completed. Data: {dat_path(args.model_size)}")


if __name__ == "__main__":
    main()
