.PHONY: install explore baseline quantum evaluate \
        smoke-baseline smoke-quantum smoke-se help

DATA_ROOT ?= asl-dataset

# ── Setup ─────────────────────────────────────────────────────────────────────

install:
	pip install -r requirements.txt

# ── Dataset sanity check ──────────────────────────────────────────────────────

explore:
	python -m src.explore_dataset --data-root $(DATA_ROOT)

# ── Training ──────────────────────────────────────────────────────────────────

# Phase 0: classical DenseNet121 baseline
baseline:
	python -m src.train_baseline --config configs/baseline.yaml \
		$(if $(DATA_ROOT),--data-root $(DATA_ROOT),)

# Phase 2: DenseNet + QuantumChannelAttention
quantum:
	python -m src.train_quantum --config configs/quantum.yaml \
		$(if $(DATA_ROOT),--data-root $(DATA_ROOT),)

# ── Evaluation ────────────────────────────────────────────────────────────────

# Evaluate a checkpoint — set CKPT= and DATA_ROOT=
# Example: make evaluate CKPT=runs/densenet_baseline/best_model.pt
evaluate:
	python -m src.evaluate \
		--checkpoint $(CKPT) \
		--data-root $(DATA_ROOT)

# ── Smoke tests (no data / GPU needed) ───────────────────────────────────────

smoke-baseline:
	python -m src.models.densenet_baseline

smoke-quantum:
	python -m src.models.quantum_attention

smoke-se:
	python -m src.models.classical_attention

# ── Help ──────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "Usage: make <target> [DATA_ROOT=<path>] [CKPT=<path>]"
	@echo ""
	@echo "Setup"
	@echo "  install        pip install -r requirements.txt"
	@echo ""
	@echo "Dataset"
	@echo "  explore        sanity-check dataset (DATA_ROOT=asl-dataset)"
	@echo ""
	@echo "Training"
	@echo "  baseline       Phase 0 — DenseNet121 (configs/baseline.yaml)"
	@echo "  quantum        Phase 2 — DenseNet + QuantumChannelAttention (configs/quantum.yaml)"
	@echo ""
	@echo "Evaluation"
	@echo "  evaluate       eval checkpoint  CKPT=runs/.../best_model.pt"
	@echo ""
	@echo "Smoke tests  (shape / gradient checks, no dataset needed)"
	@echo "  smoke-baseline   DenseNet121 shape check"
	@echo "  smoke-quantum    PennyLane circuit + gradient flow"
	@echo "  smoke-se         SE / CBAM shape check"
	@echo ""
