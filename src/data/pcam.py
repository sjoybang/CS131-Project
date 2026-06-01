import h5py
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"

# PCam HDF5 file names
_FILES = {
    "train": ("camelyonpatch_level_2_split_train_x.h5", "camelyonpatch_level_2_split_train_y.h5"),
    "valid": ("camelyonpatch_level_2_split_valid_x.h5", "camelyonpatch_level_2_split_valid_y.h5"),
    "test":  ("camelyonpatch_level_2_split_test_x.h5",  "camelyonpatch_level_2_split_test_y.h5"),
}


def _resolve_split_root(root, x_file, y_file):
    """Find manually placed files or torchvision's root/pcam download folder."""
    if _split_exists(root, x_file, y_file):
        return root
    return root / "pcam"


def _split_exists(root, x_file, y_file):
    return (root / x_file).exists() and (root / y_file).exists()


def _get_torchvision_dataset(split, data_dir):
    """Download/load a PCam split through torchvision."""
    from torchvision.datasets import PCAM

    torchvision_split = "val" if split == "valid" else split
    return PCAM(root=str(data_dir), split=torchvision_split, download=True)


def _load_split_from_torchvision(split, data_dir, max_samples):
    """Download/load PCam through torchvision and return numpy arrays."""
    dataset = _get_torchvision_dataset(split, data_dir)
    n_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    X = []
    y = []
    for idx in range(n_samples):
        image, label = dataset[idx]
        X.append(np.asarray(image))
        y.append(label)

    return np.asarray(X, dtype=np.uint8), np.asarray(y, dtype=int)


def load_split_numpy(split="train", data_dir=None, max_samples=None, download=False):
    """Load a PCam split as numpy arrays (uint8 images, int labels).

    Returns:
        X: (N, 96, 96, 3) uint8
        y: (N,) int  — 0 = normal, 1 = tumor
    """
    root = Path(data_dir) if data_dir else DATA_DIR
    x_file, y_file = _FILES[split]
    split_root = _resolve_split_root(root, x_file, y_file)

    if download and not _split_exists(split_root, x_file, y_file):
        return _load_split_from_torchvision(split, root, max_samples)

    with h5py.File(split_root / x_file, "r") as fx, h5py.File(split_root / y_file, "r") as fy:
        X = fx["x"][:max_samples]          # (N, 96, 96, 3) uint8
        y = fy["y"][:max_samples, 0, 0, 0] # stored as (N, 1, 1, 1)
    return X, y.astype(int)


class PCamDataset(Dataset):
    """PyTorch Dataset for the PCam CNN pipeline.

    Args:
        split: "train", "valid", or "test"
        transform: torchvision transforms applied to each PIL/tensor image
        max_samples: cap the dataset size (useful for quick experiments)
        download: download the split through torchvision if it is missing
    """

    def __init__(self, split="train", transform=None, data_dir=None, max_samples=None, download=False):
        root = Path(data_dir) if data_dir else DATA_DIR
        x_file, y_file = _FILES[split]
        split_root = _resolve_split_root(root, x_file, y_file)

        if download and not _split_exists(split_root, x_file, y_file):
            _get_torchvision_dataset(split, root)
            split_root = _resolve_split_root(root, x_file, y_file)

        self.x_path = split_root / x_file
        self.y_path = split_root / y_file
        with h5py.File(self.x_path, "r") as fx:
            total_samples = fx["x"].shape[0]
        self.num_samples = total_samples if max_samples is None else min(max_samples, total_samples)
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        from PIL import Image
        import torch

        with h5py.File(self.x_path, "r") as fx:
            img = Image.fromarray(fx["x"][idx])
        with h5py.File(self.y_path, "r") as fy:
            label = int(fy["y"][idx, 0, 0, 0])
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


def get_default_transforms(split="train"):
    """Standard torchvision transforms for ResNet18 input."""
    from torchvision import transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if split == "train":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([transforms.ToTensor(), normalize])
