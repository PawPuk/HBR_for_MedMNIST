"""Wrappers for datasets that allowing mapping between"""

import os
from typing import Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        data, label = self.dataset[idx]
        if self.transform is not None:
            data = self.transform(data)
        label = label.squeeze(0)
        return data, label, idx

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]


class LocalDataset(Dataset):
    """Dataset for images stored as .jpg files, supporting train/val/test splits.
    The raw data was downloaded as .npz from https://zenodo.org/records/10519652 (and saved in appropriate folder)

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

    def __init__(self, root: str, masking_percentage: float, split: str, transform=None, as_rgb=False):
        """
        Args:
            root: Root directory containing the mask folders.
            masking_percentage: One of 0.25, 0.50, 0.75, 1.00.
            split: Which split to load ('train', 'val', 'test'). Default 'test'.
            transform: Torchvision transform to apply to images.
            as_rgb (bool, optional): If true, convert grayscale images to 3-channel images. Default: False.
        """
        self.root = root
        self.masking_percentage = masking_percentage
        self.split = split
        self.transform = transform
        self.as_rgb = as_rgb

        # Build folder name (e.g., mask0.25_drop0.00)
        folder_name = f"mask{masking_percentage:.2f}_drop0.00"
        self.image_dir = os.path.join(root, folder_name)
        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        # Gather all .jpg files and extract labels from filenames
        self.samples = []
        for filename in os.listdir(self.image_dir):
            # Only consider .jpg files matching the requested split
            if not filename.startswith(self.split) or not filename.endswith('.jpg'):
                continue

            name_without_ext = os.path.splitext(filename)[0]
            label = int(name_without_ext.split('_')[-1])
            self.samples.append((filename, label))

        # Sort to have deterministic order (required even if we later shuffle to ensure reproducibility)
        self.samples.sort(key=lambda x: x[0])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        img_path = os.path.join(self.image_dir, filename)
        img = Image.open(img_path)

        if self.as_rgb:
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)
