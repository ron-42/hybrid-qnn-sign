"""Aggregate independently evaluated low-data runs and plot learning curves."""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import torch


MODELS = ("none", "se", "qca", "mlp")
PARAM_FIELDS = ("attention_params", "trainable_params", "total_params")


def parse_report(path: Path) -> tuple[float, float]:
    text = path.read_text()
    accuracy = re.search(r"accuracy\s+([0-9.]+)\s+\d+", text)
    macro_f1 = re.search(
        r"macro avg\s+[0-9.]+\s+[0-9.]+\s+([0-9.]+)\s+\d+",
        text,
    )
    if accuracy is None or macro_f1 is None:
        raise ValueError(f"Could not parse accuracy and macro-F1 from {path}")
    return float(accuracy.group(1)), float(macro_f1.group(1))


def quantum_param_counts(checkpoint: dict) -> dict[str, int]:
    from .models.densenet_quantum_attention import build_model

    args = checkpoint["args"]
    model = build_model(
        num_classes=len(checkpoint["class_to_idx"]),
        n_qubits=checkpoint["n_qubits"],
        n_layers=checkpoint["n_layers"],
        pretrained=False,
        freeze_backbone=args["freeze_backbone"],
        device_name=checkpoint["pennylane_device"],
        quantum_architecture=checkpoint.get("quantum_architecture", "v1"),
    )
    return {
        "attention_params": sum(p.numel() for p in model.q_attn.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_params": sum(p.numel() for p in model.parameters()),
    }


def read_run(runs_dir: Path, fraction: float, model: str, seed: int) -> dict:
    pct = int(fraction * 100)
    run_dir = runs_dir / f"low_data_{pct}pct_{model}_seed{seed}"
    eval_reports = sorted(run_dir.glob("eval_test_*.txt"))
    if len(eval_reports) != 1:
        raise FileNotFoundError(
            f"Expected one independent eval report in {run_dir}, found {len(eval_reports)}"
        )

    with open(run_dir / "history.csv", newline="") as f:
        history = list(csv.DictReader(f))
    best = max(history, key=lambda row: float(row["val_acc"]))
    test_acc, macro_f1 = parse_report(eval_reports[0])

    checkpoint = torch.load(
        run_dir / "best_model.pt",
        map_location="cpu",
        weights_only=False,
    )
    if model == "qca":
        params = quantum_param_counts(checkpoint)
    else:
        params = {field: int(checkpoint[field]) for field in PARAM_FIELDS}

    return {
        "model": model,
        "backbone": "densenet",
        "train_fraction": fraction,
        "seed": seed,
        "best_val_acc": float(best["val_acc"]),
        "test_acc": test_acc,
        "macro_f1": macro_f1,
        **params,
        "best_epoch": int(best["epoch"]),
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["train_fraction"], row["model"])].append(row)

    summary = []
    for (fraction, model), group in sorted(grouped.items()):
        summary.append({
            "model": model,
            "backbone": "densenet",
            "train_fraction": fraction,
            "n_seeds": len(group),
            "best_val_acc_mean": mean(row["best_val_acc"] for row in group),
            "best_val_acc_sample_std": stdev(row["best_val_acc"] for row in group),
            "test_acc_mean": mean(row["test_acc"] for row in group),
            "test_acc_sample_std": stdev(row["test_acc"] for row in group),
            "macro_f1_mean": mean(row["macro_f1"] for row in group),
            "macro_f1_sample_std": stdev(row["macro_f1"] for row in group),
            **{field: group[0][field] for field in PARAM_FIELDS},
        })
    return summary


def plot_learning_curve(summary: list[dict], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "none": "No attention",
        "se": "SE",
        "qca": "QCA",
        "mlp": "Matched MLP",
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in MODELS:
        rows = sorted(
            (row for row in summary if row["model"] == model),
            key=lambda row: row["train_fraction"],
        )
        ax.errorbar(
            [row["train_fraction"] * 100 for row in rows],
            [row["test_acc_mean"] * 100 for row in rows],
            yerr=[row["test_acc_sample_std"] * 100 for row in rows],
            marker="o",
            capsize=3,
            label=labels[model],
        )

    ax.set_xlabel("Training split used (%)")
    ax.set_ylabel("Held-out test accuracy (%)")
    ax.set_title("Low-data ASL accuracy (mean ± sample SD, seeds 42–44)")
    ax.set_xticks([1, 5, 10])
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()

    fractions = (0.01, 0.05, 0.10)
    seeds = (42, 43, 44)
    rows = [
        read_run(args.runs_dir, fraction, model, seed)
        for fraction in fractions
        for model in MODELS
        for seed in seeds
    ]
    summary = summarize(rows)

    raw_path = args.runs_dir / "low_data_results.csv"
    summary_path = args.runs_dir / "low_data_summary.csv"
    plot_path = args.runs_dir / "low_data_learning_curve.png"
    write_csv(raw_path, rows, list(rows[0]))
    write_csv(summary_path, summary, list(summary[0]))
    plot_learning_curve(summary, plot_path)
    print(f"Saved {raw_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
