"""
Standalone evaluation script: loads a saved checkpoint and runs the full
held-out test split, then writes a classification report and a confusion
matrix PNG.

Works for ANY checkpoint produced by train_baseline.py or train_quantum.py —
the checkpoint stores the model type in args.

Usage:
    python -m src.evaluate \\
        --checkpoint runs/densenet_baseline/best_model.pt \\
        --data-root asl-dataset \\
        --output-dir runs/densenet_baseline

    python -m src.evaluate \\
        --checkpoint runs/densenet_quantum/best_model.pt \\
        --data-root asl-dataset \\
        --output-dir runs/densenet_quantum
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from .dataset import get_dataloaders
from .utils import get_device


def load_model_from_checkpoint(ckpt: dict, device: torch.device):
    """
    Reconstruct the model from the checkpoint's saved args dict.
    Supports DenseNetASL (baseline) and DenseNetQuantumAttention (quantum).
    """
    saved_args = ckpt.get("args", {})
    class_to_idx = ckpt["class_to_idx"]
    num_classes = len(class_to_idx)

    is_quantum = "n_qubits" in ckpt

    if is_quantum:
        from .models.densenet_quantum_attention import build_model
        model = build_model(
            num_classes=num_classes,
            n_qubits=ckpt["n_qubits"],
            n_layers=ckpt["n_layers"],
            pretrained=False,  # weights come from the checkpoint
            freeze_backbone=saved_args.get("freeze_backbone", True),
            device_name=ckpt.get("pennylane_device", "default.qubit"),
        )
        model_tag = f"quantum_q{ckpt['n_qubits']}_l{ckpt['n_layers']}"
    else:
        from .models.densenet_baseline import build_model
        model = build_model(num_classes=num_classes, pretrained=False)
        model_tag = "baseline"

    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, model_tag


@torch.no_grad()
def run_evaluation(model, loader, device, class_names):
    all_preds, all_labels, all_probs = [], [], []
    for images, labels in tqdm(loader, desc="evaluating"):
        images = images.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=1).cpu()
        preds = logits.argmax(1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
        all_probs.append(probs)

    return all_labels, all_preds, torch.cat(all_probs, dim=0)


def save_confusion_matrix(labels, preds, class_names, out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        cm = confusion_matrix(labels, preds)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        n = len(class_names)
        fig, ax = plt.subplots(figsize=(max(10, n * 0.5), max(8, n * 0.45)))
        im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(class_names, rotation=90, fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Normalised confusion matrix")

        # Annotate cells with counts for small n
        if n <= 30:
            for i in range(n):
                for j in range(n):
                    color = "white" if cm_norm[i, j] > 0.6 else "black"
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            fontsize=6, color=color)

        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved confusion matrix to {out_path}")
    except Exception as e:
        print(f"Could not save confusion matrix ({e})")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to best_model.pt produced by a training script")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Path to the asl-alphabet dataset root")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where to write eval outputs (defaults to checkpoint dir)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["test", "val", "train"], default="test",
                        help="Which split to evaluate (default: test)")
    args = parser.parse_args()

    output_dir = args.output_dir or args.checkpoint.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")

    # ── Load checkpoint ────────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    class_to_idx = ckpt["class_to_idx"]
    class_names = sorted(class_to_idx, key=class_to_idx.get)

    print(f"Checkpoint metadata: epoch={ckpt.get('epoch')}  "
          f"val_acc={ckpt.get('val_acc', 'n/a'):.4f}")
    if "n_qubits" in ckpt:
        print(f"  quantum: n_qubits={ckpt['n_qubits']}  n_layers={ckpt['n_layers']}  "
              f"q_params={ckpt.get('n_q_params', '?'):,}")

    model, model_tag = load_model_from_checkpoint(ckpt, device)

    # ── Build dataloader ───────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, _ = get_dataloaders(
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    loader_map = {"train": train_loader, "val": val_loader, "test": test_loader}
    loader = loader_map[args.split]
    print(f"Evaluating on {args.split} split ...")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    labels, preds, probs = run_evaluation(model, loader, device, class_names)

    report = classification_report(labels, preds, target_names=class_names, digits=4)
    print(f"\n{args.split.capitalize()} set classification report:")
    print(report)

    # Save report
    report_path = output_dir / f"eval_{args.split}_{model_tag}.txt"
    with open(report_path, "w") as f:
        f.write(f"checkpoint: {args.checkpoint}\n")
        f.write(f"split: {args.split}\n\n")
        f.write(report)
    print(f"Saved report to {report_path}")

    # Save confusion matrix
    cm_path = output_dir / f"confusion_{args.split}_{model_tag}.png"
    save_confusion_matrix(labels, preds, class_names, cm_path)

    # Save per-class probabilities summary
    prob_path = output_dir / f"probs_{args.split}_{model_tag}.pt"
    torch.save({"probs": probs, "labels": torch.tensor(labels),
                "preds": torch.tensor(preds), "class_names": class_names},
               prob_path)
    print(f"Saved probability tensors to {prob_path}")


if __name__ == "__main__":
    main()
