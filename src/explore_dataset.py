"""
Quick sanity check on the ASL Alphabet dataset (Kaggle: grassknoted/asl-alphabet).

Run this first after unzipping the dataset, e.g.:

    python explore_dataset.py --data-root /path/to/asl-alphabet

It will:
  - locate the training directory (29 class folders: A-Z, del, nothing, space)
  - print per-class image counts
  - check image sizes / modes for a sample of files
  - save a small sample grid image to ./sample_grid.png
"""

import argparse
import os
import random
from collections import OrderedDict
from pathlib import Path

from PIL import Image

EXPECTED_CLASSES = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + ["del", "nothing", "space"]


def find_train_dir(data_root: Path) -> Path:
    """
    The Kaggle download usually unzips to a structure like:
        asl-alphabet/
          asl_alphabet_train/
            asl_alphabet_train/
              A/ B/ C/ ... del/ nothing/ space/
          asl_alphabet_test/
            asl_alphabet_test/
              ...

    This walks the tree looking for the first directory that contains
    (most of) the expected 29 class folders.
    """
    candidates = []
    for dirpath, dirnames, _ in os.walk(data_root):
        names = set(dirnames)
        overlap = names.intersection(EXPECTED_CLASSES)
        if len(overlap) >= 25:  # allow a little slack
            candidates.append(Path(dirpath))

    if not candidates:
        raise FileNotFoundError(
            f"Could not find a directory under {data_root} containing the 29 "
            f"ASL class folders (A-Z, del, nothing, space). "
            f"Check that the dataset was unzipped correctly."
        )

    # Prefer the shallowest match (closest to data_root)
    candidates.sort(key=lambda p: len(p.parts))
    return candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path,
                         help="Path to the unzipped asl-alphabet folder")
    parser.add_argument("--sample-per-class", type=int, default=1,
                         help="How many sample images per class to put in the grid")
    args = parser.parse_args()

    train_dir = find_train_dir(args.data_root)
    print(f"Found training directory: {train_dir}\n")

    counts = OrderedDict()
    sizes_seen = set()
    modes_seen = set()
    sample_paths = []

    class_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    for class_dir in class_dirs:
        files = [f for f in class_dir.iterdir() if f.is_file()]
        counts[class_dir.name] = len(files)

        # Inspect a few images per class for size/mode consistency
        check_n = min(5, len(files))
        for f in random.sample(files, check_n) if files else []:
            with Image.open(f) as im:
                sizes_seen.add(im.size)
                modes_seen.add(im.mode)

        if files:
            sample_paths.append(random.choice(files))

    total = sum(counts.values())
    print(f"Classes found: {len(counts)} (expected 29)")
    print(f"Total images: {total}\n")

    print("Per-class counts:")
    for cls, n in counts.items():
        print(f"  {cls:>8s}: {n}")

    print(f"\nImage sizes seen (sample): {sizes_seen}")
    print(f"Image modes seen (sample): {modes_seen}")

    if set(counts.keys()) != set(EXPECTED_CLASSES):
        missing = set(EXPECTED_CLASSES) - set(counts.keys())
        extra = set(counts.keys()) - set(EXPECTED_CLASSES)
        if missing:
            print(f"\nWARNING: missing expected classes: {sorted(missing)}")
        if extra:
            print(f"\nWARNING: unexpected extra folders: {sorted(extra)}")

    # Save a quick sample grid for a visual sanity check
    try:
        import matplotlib.pyplot as plt

        n = len(sample_paths)
        cols = 6
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        axes = axes.flatten()
        for ax, p in zip(axes, sample_paths):
            with Image.open(p) as im:
                ax.imshow(im)
            ax.set_title(p.parent.name, fontsize=8)
            ax.axis("off")
        for ax in axes[n:]:
            ax.axis("off")
        plt.tight_layout()
        out_path = "sample_grid.png"
        plt.savefig(out_path, dpi=120)
        print(f"\nSaved sample grid to {out_path}")
    except Exception as e:  # pragma: no cover
        print(f"\nSkipped sample grid (matplotlib issue: {e})")


if __name__ == "__main__":
    main()
