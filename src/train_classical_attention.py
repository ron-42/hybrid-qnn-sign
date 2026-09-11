"""
Train matched DenseNet121 classical controls for the quantum-attention study.

Supports SE, parameter-matched MLP attention, and a no-attention classifier
head under the same training and checkpoint conventions as train_quantum.py.
"""

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn

from .dataset import get_dataloaders
from .train_baseline import evaluate_test_set, run_epoch
from .utils import get_device, merge_config_and_args, save_resolved_config


def main():
    parser = argparse.ArgumentParser(
        description="DenseNet121 classical-attention control training"
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
    parser.add_argument("--split-seed", type=int, default=42,
                        help="Fixed train/val/test split seed")
    parser.add_argument("--train-fraction", type=float, default=1.0,
                        help="Stratified fraction of the training split to use")
    # Training
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    # Model
    parser.add_argument("--attention-type", choices=["se", "mlp", "none"])
    parser.add_argument("--reduction", type=int, default=16,
                        help="SE bottleneck reduction ratio")
    parser.add_argument("--bottleneck-dim", type=int, default=8,
                        help="MLP attention bottleneck width")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Freeze DenseNet features; train only attention/head")

    args = merge_config_and_args(parser)

    if args.data_root is None or args.output_dir is None or args.attention_type is None:
        parser.error(
            "--data-root, --output-dir, and --attention-type are required "
            "(or set them in YAML)"
        )

    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(args, args.output_dir)

    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, class_to_idx = get_dataloaders(
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.split_seed,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        train_fraction=args.train_fraction,
        subsample_seed=args.seed,
    )
    class_names = sorted(class_to_idx, key=class_to_idx.get)

    if args.attention_type == "se":
        from .models.classical_attention import build_model

        model = build_model(
            num_classes=len(class_to_idx),
            reduction=args.reduction,
            pretrained=args.pretrained,
            freeze_backbone=args.freeze_backbone,
        )
        attention_params = sum(p.numel() for p in model.se.parameters())
    elif args.attention_type == "mlp":
        from .models.classical_attention import build_mlp_model

        model = build_mlp_model(
            num_classes=len(class_to_idx),
            bottleneck_dim=args.bottleneck_dim,
            pretrained=args.pretrained,
            freeze_backbone=args.freeze_backbone,
        )
        attention_params = sum(p.numel() for p in model.mlp_attn.parameters())
    else:
        from .models.densenet_baseline import build_model

        model = build_model(
            num_classes=len(class_to_idx),
            pretrained=args.pretrained,
            freeze_backbone=args.freeze_backbone,
        )
        attention_params = 0

    model = model.to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nattention_type={args.attention_type}")
    print(f"Attention params : {attention_params:>10,}")
    print(f"Trainable params : {trainable_params:>10,}")
    print(f"Total params     : {total_params:>10,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    history_path = args.output_dir / "history.csv"
    best_ckpt_path = args.output_dir / "best_model.pt"
    best_val_acc = 0.0

    with open(history_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "train_acc", "val_loss", "val_acc",
            "lr", "time_sec",
        ])

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss, train_acc = run_epoch(
                model, train_loader, criterion, optimizer, device, train=True
            )
            val_loss, val_acc = run_epoch(
                model, val_loader, criterion, optimizer, device, train=False
            )
            scheduler.step()
            elapsed = time.time() - t0

            print(
                f"[{epoch:02d}/{args.epochs}] "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
                f"({elapsed:.1f}s)"
            )
            writer.writerow([
                epoch, train_loss, train_acc, val_loss, val_acc,
                optimizer.param_groups[0]["lr"], elapsed,
            ])
            f.flush()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    "model_state": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "args": vars(args),
                    "model_family": "classical_attention",
                    "backbone": "densenet",
                    "attention_type": args.attention_type,
                    "reduction": args.reduction,
                    "bottleneck_dim": args.bottleneck_dim,
                    "attention_params": attention_params,
                    "trainable_params": trainable_params,
                    "total_params": total_params,
                    "train_fraction": args.train_fraction,
                    "seed": args.seed,
                    "split_seed": args.split_seed,
                }, best_ckpt_path)
                print(f"  -> new best checkpoint  val_acc={val_acc:.4f}")

    print("\nLoading best checkpoint for held-out test evaluation ...")
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    report = evaluate_test_set(model, test_loader, device, class_names)
    print("\nTest set classification report:")
    print(report)

    report_path = args.output_dir / "test_report.txt"
    with open(report_path, "w") as f:
        f.write(
            f"attention_type={args.attention_type}  "
            f"attention_params={attention_params}  "
            f"trainable_params={trainable_params}  total_params={total_params}\n\n"
        )
        f.write(report)
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
