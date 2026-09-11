"""
Dataset + dataloaders for the ASL Alphabet dataset.

The Kaggle "asl_alphabet_train" split (87,000 images) is the only one large
enough to be useful, so we build our own stratified train/val/test split
from it rather than relying on the tiny official test set (28 images).
"""

import json
import os
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image

# pin_memory only works on CUDA; silently ignored (with a warning) on MPS
_PIN_MEMORY = torch.cuda.is_available()
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .explore_dataset import find_train_dir, EXPECTED_CLASSES


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(img_size: int = 224):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.0),  # ASL signs are handedness-sensitive; keep off by default
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


class ASLAlphabetDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as im:
            im = im.convert("RGB")
            if self.transform:
                im = self.transform(im)
        return im, label


def collect_samples(data_root: Path) -> Tuple[List[Tuple[str, int]], dict]:
    train_dir = find_train_dir(data_root)
    class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])

    # Sanity check against expected classes, but don't hard-fail if extra/missing
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    samples = []
    for cls in class_names:
        cls_dir = train_dir / cls
        for f in cls_dir.iterdir():
            if f.is_file():
                samples.append((str(f), class_to_idx[cls]))

    return samples, class_to_idx


def stratified_split(samples: List[Tuple[str, int]], val_frac: float, test_frac: float, seed: int):
    labels = [s[1] for s in samples]

    train_samples, holdout_samples = train_test_split(
        samples, test_size=(val_frac + test_frac), stratify=labels, random_state=seed,
    )

    holdout_labels = [s[1] for s in holdout_samples]
    rel_test_frac = test_frac / (val_frac + test_frac)
    val_samples, test_samples = train_test_split(
        holdout_samples, test_size=rel_test_frac, stratify=holdout_labels, random_state=seed,
    )

    return train_samples, val_samples, test_samples


def stratified_subsample(samples: List[Tuple[str, int]], fraction: float, seed: int):
    """Select a deterministic, class-stratified fraction of training samples."""
    if not 0 < fraction <= 1:
        raise ValueError(f"train_fraction must be in (0, 1], got {fraction}")
    if fraction == 1:
        return samples

    labels = [sample[1] for sample in samples]
    selected, _ = train_test_split(
        samples,
        train_size=fraction,
        stratify=labels,
        random_state=seed,
    )
    return selected


def get_dataloaders(data_root: Path, img_size: int = 224, batch_size: int = 64,
                     val_frac: float = 0.1, test_frac: float = 0.1, seed: int = 42,
                     num_workers: int = 4, output_dir: Path = None,
                     train_fraction: float = 1.0, subsample_seed: int = None):
    samples, class_to_idx = collect_samples(data_root)
    train_samples, val_samples, test_samples = stratified_split(samples, val_frac, test_frac, seed)
    full_train_size = len(train_samples)
    train_samples = stratified_subsample(
        train_samples,
        fraction=train_fraction,
        seed=seed if subsample_seed is None else subsample_seed,
    )

    train_tf, eval_tf = build_transforms(img_size)

    train_ds = ASLAlphabetDataset(train_samples, transform=train_tf)
    val_ds = ASLAlphabetDataset(val_samples, transform=eval_tf)
    test_ds = ASLAlphabetDataset(test_samples, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=_PIN_MEMORY)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=_PIN_MEMORY)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=_PIN_MEMORY)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "class_to_idx.json", "w") as f:
            json.dump(class_to_idx, f, indent=2)
        print(f"Saved class_to_idx mapping to {output_dir / 'class_to_idx.json'}")

    print(f"Total samples: {len(samples)}")
    print(f"  train (full split): {full_train_size}")
    print(f"  train (fraction={train_fraction:g}): {len(train_samples)}")
    print(f"  val:   {len(val_samples)}")
    print(f"  test:  {len(test_samples)}")
    print(f"Classes ({len(class_to_idx)}): {list(class_to_idx.keys())}")

    return train_loader, val_loader, test_loader, class_to_idx
