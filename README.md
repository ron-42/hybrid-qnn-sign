# QNN + DenseNet for ASL Alphabet Classification

Research scaffold: quantum attention layers on top of a DenseNet121
backbone, evaluated on the Kaggle ASL Alphabet dataset (87k images, 29
classes: A-Z, del, nothing, space).

## Setup (Apple Silicon / M-series)

```bash
uv venv                          # creates .venv with system Python
source .venv/bin/activate
uv pip install torch torchvision
uv pip install numpy pandas scikit-learn matplotlib pillow tqdm pyyaml
uv pip install pennylane
uv pip install pennylane-lightning   # needs Xcode CLT; skip if it fails
```

Verify MPS is available (M1/M2/M3/M4):

```bash
python -c "import torch; print(torch.backends.mps.is_available())"  # expect: True
```

All training scripts auto-select the best device: CUDA → MPS → CPU.
On M4 Air you'll see `Device: mps` at startup.

All commands below are run from the **project root** (not from `src/`).

## 1. Verify the dataset

```bash
python -m src.explore_dataset --data-root /path/to/asl-alphabet
# or: make explore DATA_ROOT=/path/to/asl-alphabet
```

This locates the 29 class folders, prints per-class counts, checks image
sizes/modes, and saves `sample_grid.png` for a visual check. Expect 3,000
images per class, 200x200 RGB.

## 2. Phase 0 -- classical DenseNet121 baseline

```bash
python -m src.train_baseline \
    --data-root /path/to/asl-alphabet \
    --output-dir runs/densenet_baseline \
    --epochs 10 --batch-size 64 --lr 3e-4
# or: make baseline DATA_ROOT=/path/to/asl-alphabet
```

This builds an 80/10/10 stratified train/val/test split (the official
"test" folder only has 28 images, too small to be useful), fine-tunes
DenseNet121, and writes:

- `runs/densenet_baseline/history.csv` -- per-epoch train/val loss & acc
- `runs/densenet_baseline/best_model.pt` -- best checkpoint by val acc
- `runs/densenet_baseline/test_report.txt` -- per-class precision/recall/F1
- `runs/densenet_baseline/class_to_idx.json` -- label mapping (reused by
  later phases)

Expect this to land in the high-90s / near-100% range (consistent with
published DenseNet121 results on this dataset) -- this is your accuracy
ceiling, not your target to beat. The interesting comparisons later are
parameter count, sample efficiency, and robustness, not raw accuracy.

## 3. Phase 1 -- validate the quantum attention module in isolation

No dataset or DenseNet needed -- this just checks PennyLane + the
`TorchLayer` integration works and is differentiable:

```bash
python -m src.models.quantum_attention
# or: make smoke-quantum
```

If this runs and prints a gradient/parameter summary, you're ready for
Phase 2.

## 4. Phase 2 -- DenseNet + quantum channel attention

```bash
python -m src.models.densenet_quantum_attention
# or: make smoke-baseline   (for DenseNet shape check)
```

Smoke-tests the hybrid model shape. To actually train it, write a
`train_quantum.py` that mirrors `train_baseline.py` but imports
`models.densenet_quantum_attention.build_model` instead. Recommended
first run: `freeze_backbone=True` so only the quantum attention block +
classifier train -- this is the cheapest way to check whether the quantum
layer is learning anything before paying for full fine-tuning.

## Roadmap (full plan)

- [x] Phase 0: classical DenseNet121 baseline + data pipeline
- [ ] Phase 0b: classical attention baselines (SE-block / CBAM / transformer
      self-attention on the same pooled features) -- matched parameter
      budget comparison point
- [x] Phase 1: `QuantumChannelAttention` module + smoke test
- [ ] Phase 2: `train_quantum.py` -- train DenseNet + QuantumChannelAttention,
      frozen backbone first, then fine-tuned
- [ ] Phase 3: `QuantumTokenSelfAttention` (QSANN-style, spatial tokens) --
      see stub docstring in `models/quantum_attention.py`
- [ ] Phase 4: ablations -- n_qubits (4/6/8/10), n_layers, encoding scheme
      (angle vs amplitude), attention placement
- [ ] Phase 5: noise robustness -- PennyLane `qml.DepolarizingChannel` /
      `default.mixed` device, compare degradation curves vs classical
      attention baselines
- [ ] Phase 6: low-data regime -- retrain everything on 10%/25%/50% of
      training data to test the sample-efficiency framing

## Notes on framing

DenseNet121 (and other CNNs) already reach ~99.9% on this dataset, so
"higher accuracy" is not a winnable headline result. The useful research
questions are:

1. **Parameter efficiency** -- does the quantum attention block reach
   comparable accuracy to a classical SE/CBAM block with fewer trainable
   parameters in the attention module itself?
2. **Sample efficiency** -- does it close the gap faster (fewer
   epochs/samples) in low-data regimes?
3. **Robustness** -- under simulated quantum noise, or under classical
   input perturbations, does the attention mechanism degrade differently?
4. **Interpretability** -- do the quantum attention weights highlight
   semantically meaningful channels/regions (visualize via `return_attn=True`)?

Pick one as the primary story before running large sweeps -- it determines
which ablations are worth the compute.
