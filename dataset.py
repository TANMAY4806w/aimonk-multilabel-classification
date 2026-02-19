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

    def calculate_pos_weights(self):
        """
        Calculates positive weights for each class based on the ratio of negative to positive samples.
        pos_weight = (num_neg / num_pos)
        """
        all_labels = []
        for _, labels, _ in self.samples:
            all_labels.append(labels)
        
        all_labels = torch.tensor(all_labels, dtype=torch.float32)
        
        # Calculate positive and negative counts for each class
        num_pos = all_labels.sum(dim=0)
        num_neg = len(all_labels) - num_pos
        
        # Calculate weights: avoid division by zero
        pos_weights = num_neg / (num_pos + 1e-6)
        
        return pos_weights

    def __getitem__(self, idx):
        img_path, labels, mask = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(labels, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        return image, labels, mask
