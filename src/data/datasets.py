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
    """Dataset for synthetic images stored as .jpg files, with labels from a CSV file.

    Expected folder structure:
        root/
            mask{masking_percentage}_drop0.00/
            {dataset_name}.csv

    The CSV must have columns: split, filename, label
    (e.g., "TEST,test_1234.jpg,0")
    """
    def __init__(self, root: str, dataset_name: str, masking_percentage: float,
                 transform=None, size: Optional[int] = None):
        """
        Args:
            root: Root directory containing the mask folders and the CSV.
            dataset_name: Name of the dataset (e.g., 'pathmnist').
            masking_percentage: One of 0.25, 0.50, 0.75, 1.00.
            transform: Torchvision transform to apply to images.
            size: Target size (int) to resize images to; if None, no resizing.
        """
        self.root = root
        self.masking_percentage = masking_percentage
        self.transform = transform
        self.size = size

        # Build folder name (e.g., mask0.25_drop0.00)
        folder_name = f"mask{masking_percentage:.2f}_drop0.00"
        if masking_percentage == 1.00:
            folder_name = "mask1.00_drop0.00"
        self.image_dir = os.path.join(root, folder_name)

        # Load CSV and filter rows for this split
        csv_path = os.path.join(root, f"{dataset_name}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        self.samples = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # assume header: split,filename,label
            filename_col = header.index('filename') if 'filename' in header else 1
            label_col = header.index('label') if 'label' in header else 2

            for row in reader:
                self.samples.append((row[filename_col], int(row[label_col])))

        if not self.samples:
            raise ValueError(f"No samples found in {csv_path}")

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
