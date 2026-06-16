# AGENTS.md

Context file for AI coding agents working on this repo. Read this before
making changes, especially before adding new model variants or training
scripts.

## What this project is

Research codebase: quantum-attention layers (PennyLane variational
circuits) bolted onto a DenseNet121 backbone, evaluated on the Kaggle ASL
Alphabet dataset (87k images, 29 classes: A-Z, del, nothing, space).

**The headline metric is NOT accuracy.** DenseNet121 already reaches
~99.9% on this dataset, so there's no accuracy headroom. The research
framing is one (and only one, at a time) of:

1. Parameter efficiency of the attention block vs a classical SE/CBAM block
2. Sample efficiency in low-data regimes
3. Robustness (quantum noise simulation, or input perturbation)
4. Interpretability of attention weights

When adding experiments or metrics, check `README.md`'s "Notes on framing"
section and the active roadmap item before assuming accuracy is the thing
to optimize or report.

## Repo layout

```
research-deep-sir/
├── README.md                     # setup + phase-by-phase run instructions
├── AGENTS.md                     # this file
├── requirements.txt
├── Makefile                      # convenience one-liners
├── configs/
│   ├── baseline.yaml             # Phase 0 training defaults
│   ├── quantum.yaml              # Phase 2 training defaults
│   └── se_attention.yaml         # Phase 0b SE-attention defaults
├── runs/                         # training outputs (gitignored, except .gitkeep)
└── src/
    ├── __init__.py
    ├── utils.py                  # YAML config loader + argparse merge helper
    ├── explore_dataset.py        # dataset sanity check, locates 29 class folders
    ├── dataset.py                # ASLAlphabetDataset, stratified split, dataloaders
    ├── train_baseline.py         # Phase 0: DenseNet121 fine-tuning loop
    ├── train_quantum.py          # Phase 2: DenseNet + QuantumChannelAttention
    ├── evaluate.py               # standalone checkpoint evaluation + confusion matrix
    └── models/
        ├── __init__.py                        # re-exports all model builders
        ├── densenet_baseline.py               # Phase 0
        ├── classical_attention.py             # Phase 0b: SEBlock, CBAMChannelBlock
        ├── quantum_attention.py               # Phase 1: QuantumChannelAttention
        │                                      #   + QuantumTokenSelfAttention stub
        └── densenet_quantum_attention.py      # Phase 2: DenseNet + QuantumChannelAttention
```

## Setup / environment

```bash
pip install -r requirements.txt
# or:
make install
```

- `pennylane-lightning` can fail to build on some platforms (needs a C++
  toolchain). It's optional for early phases — `default.qubit` (pure
  Python/NumPy, used everywhere currently) works without it.
- `densenet_baseline.py` and `densenet_quantum_attention.py` download
  ImageNet-pretrained DenseNet121 weights via torchvision on first run —
  requires internet access. If sandboxed/offline, pass `pretrained=False`
  for smoke tests.
- Before launching a real training run, confirm `torch.cuda.is_available()`
  is `True`. CPU training on 87k images at 224x224 is not practical.

## Commands

All commands run from the **project root** (not from `src/`).

```bash
# 1. Verify dataset structure (run once after any dataset change)
python -m src.explore_dataset --data-root asl-dataset
# or: make explore

# 2. Smoke-test model shapes + gradient flow (no dataset / GPU needed)
make smoke-baseline    # DenseNet121 shape check
make smoke-quantum     # PennyLane circuit + autograd check
make smoke-se          # SE / CBAM shape check

# 3. Phase 0 — train DenseNet121 baseline
python -m src.train_baseline --config configs/baseline.yaml
# or: make baseline

# 4. Phase 2 — train DenseNet + QuantumChannelAttention
python -m src.train_quantum --config configs/quantum.yaml
# or: make quantum

# 5. Evaluate a checkpoint (writes classification report + confusion matrix)
python -m src.evaluate \
    --checkpoint runs/densenet_baseline/best_model.pt \
    --data-root asl-dataset
# or: make evaluate CKPT=runs/densenet_baseline/best_model.pt

# 6. Config overrides: CLI flags always win over the YAML file
python -m src.train_baseline --config configs/baseline.yaml --epochs 5 --batch-size 32
```

There is no formal test suite. Every `models/*.py` has an
`if __name__ == "__main__":` smoke-test block. **New model files must
follow this pattern** — it's the fastest way to catch a broken quantum
circuit shape or a non-differentiable op before a multi-hour training run.

## Conventions to follow

**Data pipeline**
- The official Kaggle "test" folder has only 28 images and is unused.
  All splits come from `dataset.stratified_split()` on the 87k training
  images: 80/10/10 train/val/test, `seed=42`, stratified by class.
- `dataset.get_dataloaders(..., output_dir=...)` writes `class_to_idx.json`
  to the run's output dir. Any script that loads a checkpoint should load
  this mapping alongside it rather than recomputing class order.
- No horizontal flip in training transforms (`dataset.build_transforms`).
  This is deliberate — ASL fingerspelling signs are handedness/orientation
  sensitive and flipping can silently relabel a sign. Don't add flip
  augmentation without flagging it.

**Models**
- Every classifier exposes `forward_features(x) -> (B, 1024)` (pooled,
  post-ReLU DenseNet features) separately from `forward(x) -> (B, num_classes)`.
  Keep this split — Phase 2+ models hook in at `forward_features`.
- Hybrid models accept `return_attn: bool = False` and, when `True`,
  return `(logits, attention_weights)` for later visualization/analysis.
- `build_model(...)` is the standard factory function name in every
  `models/*.py` file. New variants should match this signature pattern
  (`num_classes`, `pretrained`, `freeze_backbone`, plus model-specific
  kwargs) so training scripts can swap models with a one-line import change.

**Quantum modules (`models/quantum_attention.py`)**
- Keep `n_qubits` ≤ ~10. Simulation cost scales as 2^n_qubits — don't bump
  this for "more capacity" without discussing the compute tradeoff.
- `default.qubit` + `diff_method="backprop"` is the prototyping default.
  `lightning.qubit` + `diff_method="adjoint"` is the scale-up path; switch
  via the `device_name` constructor arg, don't hardcode a new device.
- Classical features must be rescaled into a valid angle range before
  `AngleEmbedding` (see the `tanh(x) * pi/2` pattern in
  `QuantumChannelAttention.forward`). Any new encoding-based module needs
  an equivalent rescaling step — uncontrolled-range inputs to angle
  embeddings train badly and are a common silent bug.
- `QuantumTokenSelfAttention` is an intentional `NotImplementedError` stub
  (Phase 3). It's expensive — one circuit per spatial token (49 for a 7x7
  DenseNet121 feature map). Don't implement it as a quick patch; it needs
  its own design pass (see the class docstring) and should probably
  downsample the feature map first.

**Training scripts**
- All training scripts use `src.utils.merge_config_and_args` so they accept
  both `--config yaml_file` and direct CLI flags; CLI flags override YAML.
- Mirror `train_baseline.py`'s structure for new training scripts
  (`run_epoch`, CSV history logging, best-checkpoint-by-val-acc, final
  test-set classification report).
- `train_quantum.py` additionally logs `n_qubits`/`n_layers`/`n_q_params`
  prominently — this is the primary research metric (parameter efficiency).
- `evaluate.py` auto-detects model type from the checkpoint's saved args
  and reconstructs the model. Keep this working for any new model variant
  by storing `"n_qubits"` or a `"model_type"` key in the checkpoint dict.

**Config files**
- YAML keys must use underscores matching argparse `dest` names
  (e.g. `data_root`, not `data-root`).
- Path-typed keys (`data_root`, `output_dir`) are coerced to `pathlib.Path`
  by `utils.merge_config_and_args`; no manual conversion needed in scripts.

## Current status

- [x] Phase 0: baseline + data pipeline (`train_baseline.py`)
- [x] Phase 0b: classical attention (`models/classical_attention.py` —
      SEBlock + CBAMChannelBlock; needs `train_se_attention.py` for a full
      training run)
- [x] Phase 1: `QuantumChannelAttention` (validated via smoke test)
- [x] Phase 2: `train_quantum.py` — implemented, awaiting first real run
- [ ] Phase 3: `QuantumTokenSelfAttention` — design stub only
- [ ] Phase 4–6: ablations, noise robustness, low-data regime — blocked on
      Phase 2 results

Full details in `README.md`.
