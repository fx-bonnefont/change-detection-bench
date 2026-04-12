import numpy as np
import torch
from torch.utils.data import Dataset

from cd_bench.config import RAW_DATA_DIR, dat_path, metadata_path
from cd_bench.data.mask_mapping import load_scd_targets
from cd_bench.utils.io import load_metadata, read_section


class FeatureDataset(Dataset):
    """Features encoder pré-extraites + cibles sémantiques HI-UCD.

    Le ``.dat`` (format v2) contient un **pool unique** de tiles HI-UCD ;
    le découpage train/val/test est passé via ``indices`` (typiquement
    obtenu depuis :func:`cd_bench.data.splits.stratified_split`).

    ``__getitem__`` renvoie :
    - ``load_masks=True``  : ``(t1, t2, sem_t1, sem_t2, valid)``
    - ``load_masks=False`` : ``(t1, t2)``
    """

    def __init__(
        self,
        encoder_name: str,
        indices: list[int] | None = None,
        load_masks: bool = True,
        limit: int | None = None,
    ):
        metadata = load_metadata(metadata_path(encoder_name, limit))
        if metadata.get("format_version") != 2:
            raise ValueError(
                f"metadata format_version={metadata.get('format_version')} non supporté. "
                "Régénère les features via `cdbench extract` (format v2)."
            )

        self._dat_path = dat_path(encoder_name, limit)
        self._meta_2018 = metadata["sections"]["features_2018"]
        self._meta_2019 = metadata["sections"]["features_2019"]
        self._items = metadata["items"]

        full_length = int(self._meta_2018["shape"][0])
        if full_length != len(self._items):
            raise ValueError(
                f"Incohérence metadata: sections={full_length} vs items={len(self._items)}"
            )

        self._indices = list(range(full_length)) if indices is None else list(indices)
        self._features_2018: np.memmap | None = None
        self._features_2019: np.memmap | None = None
        self._load_masks = load_masks

    def _ensure_open(self) -> None:
        if self._features_2018 is None:
            self._features_2018 = read_section(self._dat_path, self._meta_2018)
            self._features_2019 = read_section(self._dat_path, self._meta_2019)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx):
        real_idx = self._indices[idx]
        self._ensure_open()
        t1 = torch.from_numpy(np.array(self._features_2018[real_idx]))
        t2 = torch.from_numpy(np.array(self._features_2019[real_idx]))
        if not self._load_masks:
            return t1, t2
        mask_path = RAW_DATA_DIR / self._items[real_idx]["mask_path"]
        sem_t1, sem_t2, valid = load_scd_targets(mask_path)
        return t1, t2, sem_t1, sem_t2, valid
