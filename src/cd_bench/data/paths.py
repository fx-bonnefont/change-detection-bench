from pathlib import Path

def get_paths(root_path, limit):
    root_path = Path(root_path)    
    paths_dict = {
        "train": {
            "2018": [p for p in sorted((root_path / "train" / "image" / "2018").glob("*.png")) if not p.name.startswith("._")][:limit],
            "2019": [p for p in sorted((root_path / "train" / "image" / "2019").glob("*.png")) if not p.name.startswith("._")][:limit],
            "mask": [p for p in sorted((root_path / "train" / "mask" / "2018_2019").glob("*.png")) if not p.name.startswith("._")][:limit],
        },
        "valid": {
            "2018": [p for p in sorted((root_path / "val" / "image" / "2018").glob("*.png")) if not p.name.startswith("._")][:limit],
            "2019": [p for p in sorted((root_path / "val" / "image" / "2019").glob("*.png")) if not p.name.startswith("._")][:limit],
            "mask": [p for p in sorted((root_path / "val" / "mask" / "2018_2019").glob("*.png")) if not p.name.startswith("._")][:limit],
        },
        "test": {
            "2018": [p for p in sorted((root_path / "test" / "image" / "2018").glob("*.png")) if not p.name.startswith("._")][:limit],
            "2019": [p for p in sorted((root_path / "test" / "image" / "2019").glob("*.png")) if not p.name.startswith("._")][:limit],
        }
    }
    return paths_dict