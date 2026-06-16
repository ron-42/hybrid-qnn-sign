"""
Phase 0: classical DenseNet121 baseline on the ASL Alphabet dataset.

Example (CLI):
    python -m src.train_baseline \\
        --data-root asl-dataset \\
        --output-dir runs/densenet_baseline \\
        --epochs 10 --batch-size 64 --lr 3e-4

Example (config file):
    python -m src.train_baseline --config configs/baseline.yaml

This establishes the accuracy/parameter-count reference point that the
quantum-attention variants (Phase 2+) will be compared against.
"""

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from tqdm import tqdm

from .dataset import get_dataloaders
from .models.densenet_baseline import build_model
from .utils import merge_config_and_args, get_device


# ── Training loop ─────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_n += images.size(0)
    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def evaluate_test_set(model, loader, device, class_names):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in tqdm(loader, leave=False, desc="test"):
        images = images.to(device)
        preds = model(images).argmax(1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
    return classification_report(all_labels, all_preds, target_names=class_names, digits=4)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 0: DenseNet121 baseline training"
    )
    # Data
    parser.add_argument("--data-root", type=Path,
                        help="Path to the asl-alphabet dataset root")
    parser.add_argument("--output-dir", type=Path,
                        help="Directory to write checkpoints / logs")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    # Model
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Freeze DenseNet feature extractor, train classifier head only")

    args = merge_config_and_args(parser)

    if args.data_root is None or args.output_dir is None:
        parser.error("--data-root and --output-dir are required (or set them in the YAML config)")

    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, class_to_idx = get_dataloaders(
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
    )
    class_names = sorted(class_to_idx, key=class_to_idx.get)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(
        num_classes=len(class_to_idx),
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_trainable:,} / Total: {n_total:,}")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training loop ─────────────────────────────────────────────────────────
    history_path = args.output_dir / "history.csv"
    best_ckpt_path = args.output_dir / "best_model.pt"
    best_val_acc = 0.0

    with open(history_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc",
                         "lr", "time_sec"])

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss, train_acc = run_epoch(
                model, train_loader, criterion, optimizer, device, train=True
            )
            val_loss, val_acc = run_epoch(
                model, val_loader, criterion, optimizer, device, train=False
            )
            scheduler.step()
            dt = time.time() - t0

            print(f"[{epoch:02d}/{args.epochs}] "
                  f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  ({dt:.1f}s)")

            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc,
                              optimizer.param_groups[0]["lr"], dt])
            f.flush()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    "model_state": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "args": vars(args),
                }, best_ckpt_path)
                print(f"  -> new best checkpoint  val_acc={val_acc:.4f}")

    # ── Final test evaluation ─────────────────────────────────────────────────
    print("\nLoading best checkpoint for held-out test evaluation ...")
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    report = evaluate_test_set(model, test_loader, device, class_names)
    print("\nTest set classification report:")
    print(report)

    report_path = args.output_dir / "test_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
