.PHONY: install explore baseline quantum quantum-frozen resnet-quantum \
        quantum-v2-pilot se-attention no-attention low-data-10-pilot \
        low-data-10-mlp mu-loso summarize-mu-loso evaluate \
        smoke-baseline smoke-quantum smoke-resnet-quantum smoke-se help

DATA_ROOT ?= asl-dataset
MU_DATA_ROOT ?= external-data/mu-handimages
METHOD ?= qca
SIGNER ?= 1
SEED ?= 42

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

# Reproduce the frozen 15-epoch QCA reference
quantum-frozen:
	python -m src.train_quantum --config configs/quantum_frozen.yaml \
		$(if $(DATA_ROOT),--data-root $(DATA_ROOT),)

quantum-v2-pilot:
	python -m src.train_quantum --config configs/quantum_v2_1pct.yaml \
		$(if $(DATA_ROOT),--data-root $(DATA_ROOT),)

# Phase 2: ResNet50 + QuantumChannelAttention
resnet-quantum:
	python -m src.train_quantum --config configs/resnet_quantum.yaml \
		$(if $(DATA_ROOT),--data-root $(DATA_ROOT),)

# Phase 0b: frozen DenseNet classical controls
se-attention:
	python -m src.train_classical_attention --config configs/se_attention.yaml \
		$(if $(DATA_ROOT),--data-root $(DATA_ROOT),)

no-attention:
	python -m src.train_classical_attention --config configs/no_attention.yaml \
		$(if $(DATA_ROOT),--data-root $(DATA_ROOT),)

# Phase C pilot: fixed holdouts, stratified 10% training subset, seed 42
low-data-10-pilot:
	python -m src.train_quantum --config configs/quantum_10pct.yaml \
		$(if $(DATA_ROOT),--data-root $(DATA_ROOT),)
	python -m src.train_classical_attention --config configs/se_attention_10pct.yaml \
		$(if $(DATA_ROOT),--data-root $(DATA_ROOT),)
	python -m src.train_classical_attention --config configs/no_attention_10pct.yaml \
		$(if $(DATA_ROOT),--data-root $(DATA_ROOT),)

low-data-10-mlp:
	python -m src.train_classical_attention --config configs/mlp_attention_10pct.yaml \
		$(if $(DATA_ROOT),--data-root $(DATA_ROOT),)

# Unfrozen signer-held-out MU HandImages comparison
mu-loso:
	python -m src.train_mu_loso --config configs/mu_loso.yaml \
		--data-root $(MU_DATA_ROOT) --method $(METHOD) \
		--held-out-signer $(SIGNER) --seed $(SEED) \
		--output-dir runs/mu_loso_signer$(SIGNER)_$(METHOD)_seed$(SEED)

summarize-mu-loso:
	python -m src.summarize_mu_loso --seed $(SEED)

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

smoke-resnet-quantum:
	python -m src.models.resnet_quantum_attention

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
	@echo "  quantum-frozen Reproduce frozen 15-epoch DenseNet + QCA reference"
	@echo "  quantum-v2-pilot  Data-reuploading QCA-v2, frozen 1% seed-42 pilot"
	@echo "  resnet-quantum Phase 2 — ResNet50 + QuantumChannelAttention (configs/resnet_quantum.yaml)"
	@echo "  se-attention   Phase 0b — frozen DenseNet + SE control"
	@echo "  no-attention   Phase 0b — frozen DenseNet classifier-head control"
	@echo "  low-data-10-pilot  Phase C — sequential 10% QCA/SE/no-attention pilot"
	@echo "  low-data-10-mlp    Phase D — 10% parameter-matched MLP control"
	@echo "  mu-loso        Unfrozen MU signer fold (METHOD=, SIGNER=, SEED=)"
	@echo "  summarize-mu-loso  Aggregate five completed MU signer folds"
	@echo ""
	@echo "Evaluation"
	@echo "  evaluate       eval checkpoint  CKPT=runs/.../best_model.pt"
	@echo ""
	@echo "Smoke tests  (shape / gradient checks, no dataset needed)"
	@echo "  smoke-baseline   DenseNet121 shape check"
	@echo "  smoke-quantum    PennyLane circuit + gradient flow"
	@echo "  smoke-resnet-quantum  ResNet50 + quantum attention shape check"
	@echo "  smoke-se         SE / CBAM shape check"
	@echo ""
