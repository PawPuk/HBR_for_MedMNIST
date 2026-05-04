"""Wrappers for datasets that allowing mapping between"""

import csv
import os
from typing import Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        data, label = self.dataset[idx]
        if self.transform:
            data = self.transform(data)
        label = label.squeeze(0)
        return data, label, idx

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]


class SyntheticDataset(Dataset):
    """Dataset for synthetic images stored as .jpg files, supporting train/val/test splits.

    Expected folder structure:
        root/
            mask{masking_percentage}_drop0.00/
                train0_0.jpg
                val0_0.jpg
                test0_0.jpg
                test1_5.jpg
                ...

    Filename format: '{split}{sample_idx}_{label}.jpg' → split prefix and label extracted.
    """

    def __init__(self, root: str, masking_percentage: float,
                 split: str, transform=None, size: Optional[int] = None):
        """
        Args:
            root: Root directory containing the mask folders.
            masking_percentage: One of 0.25, 0.50, 0.75, 1.00.
            split: Which split to load ('train', 'val', 'test'). Default 'test'.
            transform: Torchvision transform to apply to images.
            size: Target size (int) to resize images to; if None, no resizing.
                  If provided, will check the first image's dimensions and resize
                  only if they do not already match the target size.
        """
        self.root = root
        self.masking_percentage = masking_percentage
        self.split = split
        self.transform = transform
        self.size = size

        # Build folder name (e.g., mask0.25_drop0.00)
        folder_name = f"mask{masking_percentage:.2f}_drop0.00"
        if masking_percentage == 1.00:
            folder_name = "mask1.00_drop0.00"
        self.image_dir = os.path.join(root, folder_name)

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        # Gather all .jpg files and extract labels from filenames
        self.samples = []
        for filename in os.listdir(self.image_dir):
            if not filename.lower().endswith('.jpg'):
                continue

            # Check that filename starts with split prefix (e.g., 'test', 'train', 'val')
            if not filename.startswith(split):
                continue

            # Extract label from filename: {split}{sample_idx}_{label}.jpg
            name_without_ext = os.path.splitext(filename)[0]  # e.g., "test123_5"
            try:
                # Split by '_' and take the last part as label
                label = int(name_without_ext.split('_')[-1])
            except (ValueError, IndexError):
                continue  # skip files that don't match the pattern

            self.samples.append((filename, label))

        if not self.samples:
            raise ValueError(f"No valid .jpg files found for split '{split}' in {self.image_dir}")

        # Sort to have deterministic order (required even if we later shuffle to ensure reproducibility)
        self.samples.sort(key=lambda x: x[0])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        img_path = os.path.join(self.image_dir, filename)
        img = Image.open(img_path).convert('RGB')  # ensure RGB

        if self.size is not None:
            resize = transforms.Resize((self.size, self.size))
            img = resize(img)

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)
