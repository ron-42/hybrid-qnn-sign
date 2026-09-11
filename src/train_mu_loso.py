"""Train unfrozen attention models with a signer-held-out MU HandImages split."""

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from tqdm import tqdm

from .mu_dataset import get_mu_loso_dataloaders
from .train_quantum import run_epoch
from .utils import get_device, merge_config_and_args, save_resolved_config


def build_mu_model(args, num_classes: int):
    if args.method == "qca":
        from .models.densenet_quantum_attention import build_model

        model = build_model(
            num_classes=num_classes,
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            pretrained=args.pretrained,
            freeze_backbone=False,
            device_name=args.pennylane_device,
            quantum_architecture=args.quantum_architecture,
        )
        attention = model.q_attn
    elif args.method == "se":
        from .models.classical_attention import build_model

        model = build_model(
            num_classes=num_classes,
            reduction=args.reduction,
            pretrained=args.pretrained,
            freeze_backbone=False,
        )
        attention = model.se
    elif args.method == "mlp":
        from .models.classical_attention import build_mlp_model

        model = build_mlp_model(
            num_classes=num_classes,
            bottleneck_dim=args.bottleneck_dim,
            pretrained=args.pretrained,
            freeze_backbone=False,
        )
        attention = model.mlp_attn
    else:
        from .models.densenet_baseline import build_model

        model = build_model(
            num_classes=num_classes,
            pretrained=args.pretrained,
            freeze_backbone=False,
        )
        attention = None

    attention_params = (
        sum(parameter.numel() for parameter in attention.parameters())
        if attention is not None
        else 0
    )
    return model, attention_params


def build_optimizer(model, backbone_lr: float, head_lr: float, weight_decay: float):
    """Apply the professor-requested full fine-tuning with a conservative CNN LR."""
    backbone_parameters = [
        parameter for parameter in model.features.parameters() if parameter.requires_grad
    ]
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    head_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in backbone_ids
    ]
    if not backbone_parameters or not head_parameters:
        raise ValueError("Expected trainable parameters in both backbone and head")

    return torch.optim.AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": backbone_lr,
                "name": "backbone",
            },
            {"params": head_parameters, "lr": head_lr, "name": "head"},
        ],
        weight_decay=weight_decay,
    )


@torch.no_grad()
def evaluate_test(model, loader, device, class_names):
    model.eval()
    predictions, labels = [], []
    for images, batch_labels in tqdm(loader, leave=False, desc="test"):
        predicted = model(images.to(device)).argmax(1).cpu()
        predictions.extend(predicted.tolist())
        labels.extend(batch_labels.tolist())

    text_report = classification_report(
        labels,
        predictions,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    dict_report = classification_report(
        labels,
        predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return text_report, dict_report, predictions, labels


def main():
    parser = argparse.ArgumentParser(
        description="Unfrozen DenseNet121 attention comparison on MU HandImages"
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--method", choices=["none", "se", "mlp", "qca"])
    parser.add_argument("--held-out-signer", type=int, default=1)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument(
        "--pretrained", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--reduction", type=int, default=16)
    parser.add_argument("--bottleneck-dim", type=int, default=8)
    parser.add_argument("--n-qubits", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--pennylane-device", default="default.qubit")
    parser.add_argument("--quantum-architecture", choices=["v1", "v2"], default="v1")

    args = merge_config_and_args(parser)
    if args.data_root is None or args.output_dir is None or args.method is None:
        parser.error(
            "--data-root, --output-dir, and --method are required "
            "(or set them in YAML)"
        )

    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(args, args.output_dir)
    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, class_to_idx = (
        get_mu_loso_dataloaders(
            data_root=args.data_root,
            held_out_signer=args.held_out_signer,
            img_size=args.img_size,
            batch_size=args.batch_size,
            val_frac=args.val_frac,
            seed=args.seed,
            num_workers=args.num_workers,
            output_dir=args.output_dir,
        )
    )
    class_names = sorted(class_to_idx, key=class_to_idx.get)

    model, attention_params = build_mu_model(args, len(class_to_idx))
    model = model.to(device)
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_params = sum(parameter.numel() for parameter in model.parameters())
    optimizer = build_optimizer(
        model,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    criterion = nn.CrossEntropyLoss()

    print(f"Method: {args.method}; held-out signer: {args.held_out_signer}")
    print(f"Attention params: {attention_params:,}")
    print(f"Trainable params: {trainable_params:,} / {total_params:,}")
    print(
        f"Initial learning rates: backbone={args.backbone_lr:g}, "
        f"head={args.head_lr:g}"
    )

    history_path = args.output_dir / "history.csv"
    best_checkpoint_path = args.output_dir / "best_model.pt"
    best_val_accuracy = -1.0

    with open(history_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
                "backbone_lr",
                "head_lr",
                "time_sec",
            ]
        )

        for epoch in range(1, args.epochs + 1):
            started = time.time()
            train_loss, train_accuracy = run_epoch(
                model, train_loader, criterion, optimizer, device, train=True
            )
            val_loss, val_accuracy = run_epoch(
                model, val_loader, criterion, optimizer, device, train=False
            )
            elapsed = time.time() - started
            scheduler.step()
            backbone_lr, head_lr = (
                optimizer.param_groups[0]["lr"],
                optimizer.param_groups[1]["lr"],
            )

            print(
                f"[{epoch:02d}/{args.epochs}] "
                f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f} "
                f"({elapsed:.1f}s)"
            )
            writer.writerow(
                [
                    epoch,
                    train_loss,
                    train_accuracy,
                    val_loss,
                    val_accuracy,
                    backbone_lr,
                    head_lr,
                    elapsed,
                ]
            )
            file.flush()

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "class_to_idx": class_to_idx,
                        "epoch": epoch,
                        "val_acc": val_accuracy,
                        "args": vars(args),
                        "model_family": "mu_loso",
                        "method": args.method,
                        "backbone": "densenet",
                        "freeze_backbone": False,
                        "held_out_signer": args.held_out_signer,
                        "attention_params": attention_params,
                        "trainable_params": trainable_params,
                        "total_params": total_params,
                        "n_qubits": args.n_qubits,
                        "n_layers": args.n_layers,
                        "pennylane_device": args.pennylane_device,
                        "quantum_architecture": args.quantum_architecture,
                        "reduction": args.reduction,
                        "bottleneck_dim": args.bottleneck_dim,
                    },
                    best_checkpoint_path,
                )
                print(f"  -> new best checkpoint val_acc={val_accuracy:.4f}")

    checkpoint = torch.load(
        best_checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state"])
    text_report, dict_report, predictions, labels = evaluate_test(
        model, test_loader, device, class_names
    )
    test_accuracy = dict_report["accuracy"]
    macro_f1 = dict_report["macro avg"]["f1-score"]

    with open(args.output_dir / "test_report.txt", "w") as file:
        file.write(
            f"method={args.method} held_out_signer={args.held_out_signer} "
            f"attention_params={attention_params} "
            f"trainable_params={trainable_params}\n\n"
        )
        file.write(text_report)

    with open(args.output_dir / "test_metrics.json", "w") as file:
        json.dump(
            {
                "method": args.method,
                "held_out_signer": args.held_out_signer,
                "seed": args.seed,
                "best_epoch": checkpoint["epoch"],
                "best_val_accuracy": checkpoint["val_acc"],
                "test_accuracy": test_accuracy,
                "macro_f1": macro_f1,
                "attention_params": attention_params,
                "trainable_params": trainable_params,
                "n_test": len(labels),
            },
            file,
            indent=2,
        )

    with open(args.output_dir / "test_predictions.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["true_index", "true_class", "pred_index", "pred_class"])
        for true_index, pred_index in zip(labels, predictions):
            writer.writerow(
                [
                    true_index,
                    class_names[true_index],
                    pred_index,
                    class_names[pred_index],
                ]
            )

    print(text_report)
    print(f"Test accuracy: {test_accuracy:.4f}; macro-F1: {macro_f1:.4f}")


if __name__ == "__main__":
    main()
