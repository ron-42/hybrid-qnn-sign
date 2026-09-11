"""Signer-disjoint dataloaders for the MU HandImages ASL dataset."""

import json
import re
import string
from pathlib import Path
from typing import List, Tuple

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .dataset import ASLAlphabetDataset, _PIN_MEMORY, build_transforms


MUSample = Tuple[str, int, int]
MU_FILENAME = re.compile(
    r"^hand(?P<signer>\d+)_(?P<label>[a-z0-9])_"
    r"(?P<illumination>bot|dif|diff|left|right|top)_seg_"
    r"(?P<repetition>\d+)_cropped\.(?:jpe?g|png)$",
    re.IGNORECASE,
)


def collect_mu_samples(data_root: Path) -> Tuple[List[MUSample], dict]:
    """Collect A-Z images and parse signer IDs from the official filename format."""
    data_root = Path(data_root)
    image_root = data_root / "asl_dataset"
    if not image_root.is_dir():
        image_root = data_root

    class_to_idx = {
        letter.upper(): index for index, letter in enumerate(string.ascii_lowercase)
    }
    samples: List[MUSample] = []
    unparsed = []

    for path in sorted(image_root.rglob("*")):
        if not path.is_file():
            continue
        match = MU_FILENAME.match(path.name)
        if match is None:
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                unparsed.append(str(path))
            continue

        label = match.group("label").upper()
        if label not in class_to_idx:
            continue
        samples.append(
            (str(path), class_to_idx[label], int(match.group("signer")))
        )

    if not samples:
        raise FileNotFoundError(
            f"No MU HandImages files matching {MU_FILENAME.pattern!r} under {image_root}"
        )
    if unparsed:
        raise ValueError(
            f"Found {len(unparsed)} image files with unrecognized names; "
            f"first example: {unparsed[0]}"
        )

    return samples, class_to_idx


def split_mu_loso(
    samples: List[MUSample],
    held_out_signer: int,
    val_frac: float,
    seed: int,
):
    """Hold out one signer for testing; stratify validation within other signers."""
    if not 0 < val_frac < 1:
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")

    development = [sample for sample in samples if sample[2] != held_out_signer]
    test_samples = [sample for sample in samples if sample[2] == held_out_signer]
    if not test_samples:
        available = sorted({sample[2] for sample in samples})
        raise ValueError(
            f"held_out_signer={held_out_signer} not found; available={available}"
        )

    labels = [sample[1] for sample in development]
    train_samples, val_samples = train_test_split(
        development,
        test_size=val_frac,
        stratify=labels,
        random_state=seed,
    )
    return train_samples, val_samples, test_samples


def get_mu_loso_dataloaders(
    data_root: Path,
    held_out_signer: int,
    img_size: int = 224,
    batch_size: int = 32,
    val_frac: float = 0.15,
    seed: int = 42,
    num_workers: int = 2,
    output_dir: Path = None,
):
    samples, class_to_idx = collect_mu_samples(data_root)
    train_samples, val_samples, test_samples = split_mu_loso(
        samples, held_out_signer=held_out_signer, val_frac=val_frac, seed=seed
    )
    train_tf, eval_tf = build_transforms(img_size)

    # The shared dataset consumes (path, label) pairs; retain signer metadata here.
    train_ds = ASLAlphabetDataset(
        [(path, label) for path, label, _ in train_samples], transform=train_tf
    )
    val_ds = ASLAlphabetDataset(
        [(path, label) for path, label, _ in val_samples], transform=eval_tf
    )
    test_ds = ASLAlphabetDataset(
        [(path, label) for path, label, _ in test_samples], transform=eval_tf
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=_PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=_PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=_PIN_MEMORY,
    )

    signers = sorted({sample[2] for sample in samples})
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "class_to_idx.json", "w") as file:
            json.dump(class_to_idx, file, indent=2)
        with open(output_dir / "split_manifest.json", "w") as file:
            json.dump(
                {
                    "dataset": "MU HandImages ASL",
                    "held_out_signer": held_out_signer,
                    "available_signers": signers,
                    "seed": seed,
                    "val_frac": val_frac,
                    "train_samples": len(train_samples),
                    "val_samples": len(val_samples),
                    "test_samples": len(test_samples),
                    "classes": list(class_to_idx),
                },
                file,
                indent=2,
            )

    print(f"MU HandImages A-Z samples: {len(samples)}")
    print(f"  train: {len(train_samples)}")
    print(f"  val:   {len(val_samples)}")
    print(f"  test signer {held_out_signer}: {len(test_samples)}")
    print(f"Signers: {signers}; classes: {len(class_to_idx)}")

    return train_loader, val_loader, test_loader, class_to_idx
