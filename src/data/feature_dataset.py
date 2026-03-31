import numpy as np
import torch
from torch.utils.data import Dataset
from src.utils.io import load_metadata, read_section
from src.config import dat_path, metadata_path, mask_path, label_path


class FeatureDataset(Dataset):
    def __init__(self, model_size: str, part: str, load_masks: bool = False):
        metadata = load_metadata(metadata_path(model_size))

        self.features_2018 = read_section(dat_path(model_size), metadata["sections"][f"features_{part}_2018"])
        self.features_2019 = read_section(dat_path(model_size), metadata["sections"][f"features_{part}_2019"])
        self.masks = np.load(mask_path(part), mmap_mode="r") if load_masks else None
        labels = np.load(label_path(part))
        self.binary_labels = torch.from_numpy(labels[:, 0]).long()
        self.change_ratios = torch.from_numpy(labels[:, 1])

    def __len__(self):
        return len(self.features_2018)

    def __getitem__(self, idx):
        item = (
            torch.from_numpy(np.array(self.features_2018[idx])),
            torch.from_numpy(np.array(self.features_2019[idx])),
            self.binary_labels[idx],
            self.change_ratios[idx],
        )
        if self.masks is not None:
            return item + (torch.from_numpy(np.array(self.masks[idx])),)
        return item
