from pathlib import Path

# Chemins de base
RAW_DATA_DIR = Path("/Volumes/X9Pro/HI-UCD")
EXTRACTED_FEATURES_DIR = Path("/Volumes/X9Pro/HI-UCD/extracted-features")

# Sous-dossiers
def model_dir(model_size: str) -> Path:
    return EXTRACTED_FEATURES_DIR / model_size

def masks_dir() -> Path:
    return EXTRACTED_FEATURES_DIR / "masks"

# Fichiers
def dat_path(model_size: str) -> Path:
    return model_dir(model_size) / f"data_{model_size}.dat"

def metadata_path(model_size: str) -> Path:
    return model_dir(model_size) / f"metadata_{model_size}.json"

def mask_path(part: str) -> Path:
    return masks_dir() / f"masks_{part}.npy"

def mask_stats_path() -> Path:
    return masks_dir() / "mask_stats.csv"

def label_path(part: str) -> Path:
    return masks_dir() / f"labels_{part}.npy"


PRETRAINED_MODEL_NAMES = {
    "small":        "facebook/dinov3-vits16-pretrain-lvd1689m",
    "small-plus":   "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "base":         "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "large":        "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "large-sat":    "facebook/dinov3-vitl16-pretrain-sat493m",
    "huge-plus":    "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "7b":           "facebook/dinov3-vit7b16-pretrain-lvd1689m",
    "7b-sat":       "facebook/dinov3-vit7b16-pretrain-sat493m"
}
