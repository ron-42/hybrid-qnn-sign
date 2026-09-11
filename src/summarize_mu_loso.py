"""Aggregate five signer-held-out MU HandImages runs for each model."""

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


METHODS = ("none", "se", "mlp", "qca")
SIGNERS = (1, 2, 3, 4, 5)


def read_metrics(runs_root: Path, signer: int, method: str, seed: int):
    run_dir = runs_root / f"mu_loso_signer{signer}_{method}_seed{seed}"
    with open(run_dir / "test_metrics.json") as file:
        metrics = json.load(file)
    return run_dir, metrics


def read_predictions(run_dir: Path):
    true_classes, pred_classes = [], []
    with open(run_dir / "test_predictions.csv", newline="") as file:
        for row in csv.DictReader(file):
            true_classes.append(row["true_class"])
            pred_classes.append(row["pred_class"])
    return true_classes, pred_classes


def write_csv(path: Path, fieldnames, rows):
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_signer_accuracy(result_rows, output_path: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    display_names = {
        "none": "No attention",
        "se": "SE",
        "mlp": "Matched MLP",
        "qca": "QCA",
    }
    for method in METHODS:
        rows = sorted(
            (row for row in result_rows if row["method"] == method),
            key=lambda row: row["held_out_signer"],
        )
        ax.plot(
            [row["held_out_signer"] for row in rows],
            [100 * row["test_accuracy"] for row in rows],
            marker="o",
            linewidth=2,
            label=display_names[method],
        )
    ax.set_xticks(SIGNERS)
    ax.set_xlabel("Held-out signer")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Unfrozen MU HandImages signer-held-out accuracy")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_confusion_matrices(predictions_by_method, output_path: Path):
    class_names = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    display_names = {
        "none": "No attention",
        "se": "SE",
        "mlp": "Matched MLP",
        "qca": "QCA",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
    image = None
    for ax, method in zip(axes.flat, METHODS):
        true_classes, pred_classes = predictions_by_method[method]
        matrix = confusion_matrix(
            true_classes,
            pred_classes,
            labels=class_names,
            normalize="true",
        )
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
        ax.set_title(display_names[method])
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, fontsize=5)
        ax.set_yticklabels(class_names, fontsize=5)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.colorbar(image, ax=axes, shrink=0.75, label="Recall within class")
    fig.suptitle("Pooled confusion matrices across five held-out signers")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    result_rows = []
    summary_rows = []
    predictions_by_method = {}
    for method in METHODS:
        method_metrics = []
        all_true, all_pred = [], []
        for signer in SIGNERS:
            run_dir, metrics = read_metrics(
                args.runs_root, signer=signer, method=method, seed=args.seed
            )
            true_classes, pred_classes = read_predictions(run_dir)
            all_true.extend(true_classes)
            all_pred.extend(pred_classes)
            method_metrics.append(metrics)
            result_rows.append(
                {
                    "method": method,
                    "held_out_signer": signer,
                    "seed": args.seed,
                    "n_test": metrics["n_test"],
                    "best_epoch": metrics["best_epoch"],
                    "best_val_accuracy": metrics["best_val_accuracy"],
                    "test_accuracy": metrics["test_accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "attention_params": metrics["attention_params"],
                    "trainable_params": metrics["trainable_params"],
                }
            )

        accuracies = [row["test_accuracy"] for row in method_metrics]
        macro_f1s = [row["macro_f1"] for row in method_metrics]
        summary_rows.append(
            {
                "method": method,
                "attention_params": method_metrics[0]["attention_params"],
                "trainable_params": method_metrics[0]["trainable_params"],
                "signer_mean_accuracy": statistics.mean(accuracies),
                "signer_sd_accuracy": statistics.stdev(accuracies),
                "signer_mean_macro_f1": statistics.mean(macro_f1s),
                "signer_sd_macro_f1": statistics.stdev(macro_f1s),
                "pooled_accuracy": accuracy_score(all_true, all_pred),
                "pooled_macro_f1": f1_score(
                    all_true, all_pred, average="macro", zero_division=0
                ),
                "n_test": len(all_true),
            }
        )
        predictions_by_method[method] = (all_true, all_pred)

    result_fields = [
        "method",
        "held_out_signer",
        "seed",
        "n_test",
        "best_epoch",
        "best_val_accuracy",
        "test_accuracy",
        "macro_f1",
        "attention_params",
        "trainable_params",
    ]
    summary_fields = [
        "method",
        "attention_params",
        "trainable_params",
        "signer_mean_accuracy",
        "signer_sd_accuracy",
        "signer_mean_macro_f1",
        "signer_sd_macro_f1",
        "pooled_accuracy",
        "pooled_macro_f1",
        "n_test",
    ]
    write_csv(args.output_dir / "mu_loso_results.csv", result_fields, result_rows)
    write_csv(args.output_dir / "mu_loso_summary.csv", summary_fields, summary_rows)
    plot_signer_accuracy(
        result_rows, args.output_dir / "mu_loso_signer_accuracy.png"
    )
    plot_confusion_matrices(
        predictions_by_method,
        args.output_dir / "mu_loso_confusion_matrices.png",
    )

    for row in summary_rows:
        print(
            f"{row['method']:>4}: signer accuracy "
            f"{row['signer_mean_accuracy']:.4f} ± "
            f"{row['signer_sd_accuracy']:.4f}; pooled "
            f"{row['pooled_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
