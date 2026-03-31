from torch.utils.data import Dataset
from PIL import Image
import torch

class ImageDataset(Dataset):
    def __init__(self, paths, processor):
        self.paths = list(paths)
        self.processor = processor
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img_path = self.paths[idx]
        
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Erreur : {e}")
            expected_size = self.processor.size.get("height", 224) if hasattr(self.processor, "size") else 224
            return torch.zeros(3, expected_size, expected_size)
            
        inputs = self.processor(img, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0)