import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class MultiLabelDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.samples = []

        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                image_name = parts[0]

                labels = []
                mask = []

                for val in parts[1:]:
                    if val == "NA":
                        labels.append(0)
                        mask.append(0)
                    else:
                        labels.append(int(val))
                        mask.append(1)

                img_path = os.path.join(image_dir, image_name)
                if os.path.exists(img_path):
                    self.samples.append((img_path, labels, mask))

        print(f"Loaded {len(self.samples)} valid samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, labels, mask = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(labels, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        return image, labels, mask
