# AGENTS.md

Context file for AI coding agents working on this repo. Read this before
making any changes. Contains full project context, completed results,
conventions, and what is left to do.

---

## What this project is

Research codebase: variational quantum circuits (PennyLane) used as a
channel-attention mechanism inside DenseNet121 and ResNet50 backbones,
evaluated on the Kaggle ASL Alphabet dataset (87k images, 29 classes:
A–Z, del, nothing, space) and the MU HandImages ASL dataset (1,815 A–Z
images from 5 signers after filtering).

**The headline metric is NOT raw accuracy.** This random stratified split
is near-saturated: DenseNet121 reaches 100.00% test accuracy and ResNet50
+ QCA reached 99.954% validation accuracy by epoch 3. These values are not
evidence that quantum attention improves accuracy.

**Current research framing:** a controlled evaluation of whether QCA's
attention-parameter reduction relative to SE translates into a reliable
predictive benefit when compared with no attention and a parameter-matched
classical MLP. It does not in the completed experiments. The project is
therefore framed as a negative/inconclusive empirical study rather than a
quantum-advantage claim.

**Protocol priority:** the unfrozen MU signer-held-out study is the primary
external/generalisation experiment, following the supervisor's instruction
not to freeze backbone layers. Frozen Kaggle results remain useful only as
controlled head-capacity and low-data ablations.

Completed secondary study:
- Sample efficiency at 1%, 5%, and 10% training data, three seeds each

Deferred directions (start only after supervisor approval):
- Robustness under quantum noise simulation (`default.mixed` device)
- Interpretability of learned attention weights (`return_attn=True`)
- Qubit/layer sweeps or additional data fractions

---

## Completed training runs — results

Unless a subsection says otherwise, Kaggle runs use the ASL Alphabet
80/10/10 stratified split, split seed 42, AdamW, CosineAnnealingLR,
224×224 images, and the Apple M4 MPS device. Low-data training seeds and MU
signer-held-out details are stated separately below.

### Phase 0 — DenseNet121 Baseline (full fine-tune)
- Config: `configs/baseline.yaml`
- Output: `runs/densenet_baseline/`
- Epochs: 2 (converged)
- LR: 3e-4, batch size: 64
- Trainable params: **6,983,581** (all params)
- Best val acc: **99.99%** (epoch 2)
- Test acc: **100.00%** (8,700 images, all 29 classes at 1.0000 F1)

### Phase 2 — Quantum (frozen backbone, 7 epochs)
- Output: `runs/densenet_quantum_7ep/`
- Epochs: 7, LR: 1e-3, batch size: 32, freeze_backbone=True
- Quantum attention params: **17,464** | Total trainable: **47,189**
- Best val acc: **95.30%** (epoch 7, still converging)
- Test acc: **95.30%**

### Phase 2 — Quantum (frozen backbone, 15 epochs)
- Output: `runs/densenet_quantum_15ep_frozen/`
- Epochs: 15, LR: 1e-3, batch size: 32, freeze_backbone=True
- Quantum attention params: **17,464** | Total trainable: **47,189**
- Best val acc: **96.10%** (epoch 14)
- Test acc: **95.98%** | Macro-F1: **0.9599**
- Note: plateaued at ~96% — frozen backbone is the bottleneck, cosine LR hit 0

### Phase 0b — SE attention (frozen backbone, 15 epochs)
- Config: `configs/se_attention.yaml`
- Output: `runs/densenet_se_15ep_frozen/`
- Epochs: 15, LR: 1e-3, batch size: 32, seed=42
- SE attention params: **131,072** | Total trainable: **160,797**
- Best val acc: **97.72%** (epoch 14)
- Independently evaluated test acc: **97.10%** | Macro-F1: **0.9710**

### Phase 0b — No attention (frozen backbone, 15 epochs)
- Config: `configs/no_attention.yaml`
- Output: `runs/densenet_none_15ep_frozen/`
- Epochs: 15, LR: 1e-3, batch size: 32, seed=42
- Attention params: **0** | Total trainable: **29,725**
- Best val acc: **96.17%** (epoch 11)
- Independently evaluated test acc: **95.89%** | Macro-F1: **0.9588**

### Phase C — Low-data study (frozen backbone, seeds 42/43/44)
- Fixed split seed: 42; only the training split is stratified-subsampled
- Validation and test remain unchanged at 8,700 images each
- Epochs: 15, LR: 1e-3, batch size: 32
- Independently evaluated test accuracy, mean ± sample SD:
  - **1%:** SE 66.17% ± 0.42%; no attention 64.69% ± 0.63%;
    QCA 62.16% ± 1.16%; matched MLP 60.80% ± 1.31%
  - **5%:** SE 87.22% ± 0.88%; no attention 86.43% ± 0.55%;
    matched MLP 86.39% ± 0.47%; QCA 86.13% ± 0.93%
  - **10%:** SE 91.12% ± 0.45%; matched MLP 90.93% ± 0.24%;
    QCA 90.36% ± 0.35%; no attention 90.19% ± 0.60%
- QCA attention params: **17,464**; matched MLP: **17,480** (+16);
  SE: **131,072**
- Raw runs: `runs/low_data_results.csv`
- Summary: `runs/low_data_summary.csv`
- Learning curve: `runs/low_data_learning_curve.png`
- Interpretation: SE leads at every fraction. QCA trails no attention at 1%
  and 5%, and exceeds it by only 0.17 points at 10%. The matched MLP beats QCA
  at 10%. These results do not demonstrate a quantum-circuit advantage.

### QCA-v2 pilot — data re-uploading (1% training data, seed 42)
- Config: `configs/quantum_v2_1pct.yaml`
- Output: `runs/low_data_1pct_qca_v2_seed42/`
- Re-embeds the same 8 projected inputs before each variational layer
- Attention params: **17,464**, identical to QCA-v1
- Independently evaluated test acc: **61.38%** | Macro-F1: **0.6007**
- Matched QCA-v1: **61.83%** test | Macro-F1: **0.6030**
- Interpretation: v2 is 0.45 percentage points worse in this single-seed
  pilot, so broader v2 training is not justified without a new hypothesis

### MU HandImages — unfrozen signer-held-out study (seed 42)
- Supervisor-requested amendment: all DenseNet layers train; frozen results
  remain a controlled ablation rather than the primary external experiment
- Public mirror revision: `1e23d649fab0fbb1d4cac523d0c93b1ab06e2213`
- Used 1,815 A–Z images from 5 signers; one signer held out per fold
- Signer counts are imbalanced: 650, 645, 130, 130, and 260 images
- 15 epochs, batch size 32, backbone LR 1e-5, head LR 1e-3
- Equal-signer test accuracy, mean ± sample SD:
  - No attention: **94.78% ± 5.17%**
  - Matched MLP: **94.55% ± 4.02%**
  - QCA: **94.50% ± 4.95%**
  - SE: **93.72% ± 5.17%**
- Pooled accuracy: QCA 94.99%, MLP 94.88%, no attention 94.82%, SE 94.71%
- QCA paired signer-level accuracy tests: vs no attention p=0.543, vs SE
  p=0.116, vs MLP p=0.943 (two-sided paired t-tests)
- Interpretation: QCA's 0.11-point pooled lead is caused by sample weighting,
  is absent under equal-signer averaging, and is not statistically significant
- Raw results: `runs/mu_loso_results.csv`; summary:
  `runs/mu_loso_summary.csv`

### Phase 2 — Quantum (unfrozen backbone, 4 epochs — stopped early)
- Config: `configs/quantum.yaml` (current state: freeze_backbone=false, lr=3e-4, epochs=15)
- Output: `runs/densenet_quantum/`
- Epochs run: 4 (training killed after convergence confirmed)
- Trainable params: **7,001,045** (all params including backbone)
- Val accuracy progression: 99.89% → **100.00%** → 99.94% → 99.99%
- Best val acc: **100.00%** at epoch 2
- Independently evaluated test acc: **100.00%** (8,700 images)
- Test macro-F1: **1.0000**
- Evaluation output: `runs/densenet_quantum/eval_test_densenet_quantum_q8_l2.txt`

### Phase 2 — ResNet50 + Quantum (unfrozen, early-stopped after 3 epochs)
- Config: `configs/resnet_quantum.yaml`
- Output: `runs/resnet_quantum_7ep/`
- Planned epochs: 7; completed: 3; stopped after repeated saturation to avoid
  spending compute on four more full-data epochs
- LR: 3e-4, batch size: 32, freeze_backbone=False
- Quantum attention params: **34,872** | Total trainable: **23,602,325**
- Validation progression: 99.816% → 97.126% → **99.954%**
- Best val acc: **99.954%** (epoch 3)
- Independently evaluated test acc: **100.00%** (8,700 images)
- Test macro-F1: **1.0000**
- Evaluation output: `runs/resnet_quantum_7ep/eval_test_resnet_quantum_q8_l2.txt`
- Interpretation: the run fine-tunes all ResNet50 parameters, so it cannot
  establish quantum attention's contribution or parameter efficiency by itself.

### Summary table

| Model | Attention Params | Trainable Params | Epochs | Seed | Best Val Acc | Test Acc | Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DenseNet121 Baseline | 0 | 6,983,581 | 2 | 42 | 99.99% | 100.00% | 1.0000 |
| No Attention Frozen | 0 | 29,725 | 15 | 42 | 96.17% | 95.89% | 0.9588 |
| QCA Frozen 7ep | 17,464 | 47,189 | 7 | 42 | 95.30% | 95.30% | 0.9530 |
| QCA Frozen 15ep | 17,464 | 47,189 | 15 | 42 | 96.10% | 95.98% | 0.9599 |
| SE Frozen 15ep | 131,072 | 160,797 | 15 | 42 | 97.72% | 97.10% | 0.9710 |
| QCA Unfrozen 4ep | 17,464 | 7,001,045 | 4 | 42 | 100.00% | 100.00% | 1.0000 |
| ResNet50 + QCA Unfrozen | 34,872 | 23,602,325 | 3 of 7 | 42 | 99.954% | 100.00% | 1.0000 |

**Key finding:** Under matched frozen-backbone conditions, SE is strongest
(97.10% test), while QCA (95.98%) is effectively tied with no attention
(95.89%) in this single-seed experiment. QCA uses **7.5× fewer attention
parameters** than SE, but that efficiency comes with a 1.12 percentage-point
accuracy gap. The saturated unfrozen results do not establish a quantum
advantage.

---

## Repo layout

```
research-deep-sir/
├── AGENTS.md                          # durable agent handoff — read first
├── README.md                          # setup + run instructions
├── requirements.txt
├── Makefile                           # convenience one-liners
├── configs/
│   ├── baseline.yaml                  # DenseNet baseline
│   ├── quantum.yaml                   # unfrozen DenseNet-QCA
│   ├── quantum_frozen.yaml            # frozen 15-epoch QCA reference
│   ├── quantum_v2_1pct.yaml           # data-reuploading pilot
│   ├── no_attention*.yaml             # no-attention controls
│   ├── se_attention*.yaml             # SE controls
│   ├── mlp_attention_10pct.yaml       # matched MLP control
│   ├── mu_loso.yaml                   # unfrozen signer-held-out protocol
│   └── resnet_quantum.yaml            # unfrozen ResNet50-QCA
├── runs/
│   ├── low_data_results.csv           # canonical 36-run low-data results
│   ├── low_data_summary.csv           # three-seed aggregates
│   ├── mu_loso_results.csv            # canonical 20-fold results
│   ├── mu_loso_summary.csv            # signer-balanced/pooled aggregates
│   ├── low_data_learning_curve.png
│   ├── mu_loso_signer_accuracy.png
│   └── mu_loso_confusion_matrices.png
└── src/
    ├── dataset.py                     # Kaggle splits and train subsampling
    ├── mu_dataset.py                  # MU signer parsing and LOSO loaders
    ├── train_baseline.py
    ├── train_classical_attention.py   # none/SE/matched-MLP controls
    ├── train_quantum.py
    ├── train_mu_loso.py               # fully unfrozen MU training
    ├── evaluate.py
    ├── summarize_low_data.py
    ├── summarize_mu_loso.py
    ├── utils.py
    └── models/
        ├── densenet_baseline.py
        ├── classical_attention.py        # SE and matched MLP
        ├── quantum_attention.py          # QCA-v1/v2 + token-attention stub
        ├── densenet_quantum_attention.py
        └── resnet_quantum_attention.py
```

---

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

- Platform: Apple M4 Air, macOS. Device selection: CUDA → MPS → CPU (auto via `get_device()`)
- `pennylane-lightning` optional — `default.qubit` used everywhere currently
- DenseNet121 and ResNet50 download ImageNet weights on their first run
  (requires internet)
- `pin_memory` is auto-disabled on MPS (only enabled on CUDA)

---

## Commands

All commands run from the **project root**.

```bash
# Dataset check
make explore DATA_ROOT=asl-dataset

# Smoke tests (no dataset/GPU needed)
make smoke-baseline
make smoke-quantum
make smoke-resnet-quantum
make smoke-se

# Training
make baseline                          # Phase 0
make quantum                           # Phase 2
make resnet-quantum                    # Phase 2: ResNet50 + QCA
make quantum-frozen                    # reproduce frozen QCA ablation
make se-attention                      # frozen SE control
make no-attention                      # frozen classifier-head control
make quantum-v2-pilot                  # 1% data-reuploading pilot
make mu-loso METHOD=qca SIGNER=1       # one fully unfrozen MU fold
make summarize-mu-loso SEED=42         # aggregate completed MU folds

# Evaluation
make evaluate CKPT=runs/densenet_quantum/best_model.pt DATA_ROOT=asl-dataset
make evaluate CKPT=runs/resnet_quantum/best_model.pt DATA_ROOT=asl-dataset
```

---

## Conventions

**Data pipeline**
- Official Kaggle test folder (28 images) is unused. All splits come from
  `dataset.stratified_split()` on 87k train images: 80/10/10, seed=42.
- `get_dataloaders(..., output_dir=...)` writes `class_to_idx.json` to the
  run dir. Load this alongside any checkpoint — never recompute class order.
- No horizontal flip augmentation. ASL signs are handedness-sensitive;
  flipping silently relabels signs. Do not add it.
- MU uses only A–Z classes from the pinned public mirror revision
  `1e23d649fab0fbb1d4cac523d0c93b1ab06e2213`.
- MU evaluation is leave-one-signer-out. The primary summary is the
  equal-signer mean; pooled metrics are secondary because signer counts are
  imbalanced.
- Do not freeze the backbone in any new primary external experiment. Frozen
  runs may be retained only when the research question explicitly requires an
  attribution ablation.

**Models**
- Every model exposes `forward_features(x)` and `forward(x)`. DenseNet121
  returns `(B, 1024)` features; ResNet50 returns `(B, 2048)`.
- Hybrid models accept `return_attn=True` → returns `(logits, attn_weights)`.
- Factory function is always `build_model(num_classes, pretrained, freeze_backbone, ...)`.
- Checkpoints store `model_state`, `class_to_idx`, `epoch`, `val_acc`, `args`.
  Quantum checkpoints additionally store `n_qubits`, `n_layers`, `pennylane_device`,
  `n_q_params`. Quantum `args` also stores `backbone` (`densenet` or `resnet`);
  `evaluate.py` uses this to auto-reconstruct the correct model.

**Quantum circuit**
- Keep `n_qubits` ≤ 10. Sim cost scales as 2^n_qubits.
- `default.qubit` + `diff_method="backprop"` for prototyping.
  `lightning.qubit` + `diff_method="adjoint"` for scale-up.
- Input features MUST be rescaled before `AngleEmbedding`.
  Pattern: `tanh(x) * (pi/2)` → range (-π/2, π/2). Never skip this.
- PennyLane runs on CPU always. The fix in `quantum_attention.py` moves
  the tensor to CPU before the circuit and back to the original device
  (MPS/CUDA) after: `q_out = self.qlayer(z.cpu()).to(src_device)`.
  Do not remove this bridge.
- `QuantumTokenSelfAttention` is a `NotImplementedError` stub (Phase 3).
  Do not implement it without a full design pass — it runs one circuit per
  spatial token (49 for a 7×7 DenseNet feature map) which is very expensive.

**Training scripts**
- All scripts use `merge_config_and_args(parser)` — accepts `--config yaml`
  with CLI flag overrides. YAML keys use underscores (argparse dest names).
- Path keys (`data_root`, `output_dir`) auto-coerced to `pathlib.Path`.
- Structure: `run_epoch` → CSV history → best-checkpoint-by-val-acc → test report.
- New training scripts must follow this exact structure.
- Do not start another sweep, training grid, or noise experiment before the
  negative/inconclusive result is discussed with the supervisor.

---

## Current status

- [x] Phase 0: DenseNet121 baseline — **100% test accuracy**
- [x] Phase 0b: matched frozen controls — SE **97.10%**, no attention **95.89%** test accuracy
- [x] Phase 1: `QuantumChannelAttention` — implemented and validated
- [x] Phase 2: `train_quantum.py` — trained, best result 100% val acc (unfrozen, 4ep)
- [x] ResNet50 + QCA implementation and smoke test — ResNet50 support in training/evaluation
- [x] Independently evaluate unfrozen DenseNet-QCA — **100.00% test accuracy**
- [x] Early-stop the saturated ResNet50-QCA rerun at epoch 3 and independently evaluate — **100.00% test accuracy**
- [x] Phase 0b shared SE/no-attention trainer, configs, evaluation reconstruction, and Makefile targets
- [x] Train and independently evaluate matched frozen SE and no-attention controls
- [x] Add stratified train-only subsampling and independently evaluate a 10% pilot
- [x] Repeat the 10% condition with seeds 43 and 44 and report mean ± SD
- [x] Add and independently evaluate a parameter-matched MLP control at 10%
- [x] Train the four-way 1% and 5% grids for seeds 42/43/44
- [x] Implement and independently evaluate the QCA-v2 1% seed-42 pilot
- [x] Independently evaluate and aggregate the 1%/5%/10% low-data grids
- [x] Assess and preregister compatible external image validation
- [x] Complete unfrozen five-fold MU signer-held-out comparison
- [x] Compute paired signer-level tests and document metric-weighting reversal
- [ ] **Next: discuss the negative/inconclusive result with the supervisor
  before starting another experiment**
- [ ] Decide whether to repeat MU folds with additional training seeds
- [ ] Confirm MU mirror licensing before releasing data-derived artifacts
- [ ] Phase 3 qubit/layer sweep — deferred because matched controls and QCA-v2
  do not show a quantum advantage
- [ ] Phase 4 25%/50% runs — deferred unless external validation changes the
  research direction
- [ ] Phase 5: noise robustness (`default.mixed` device, depolarising channel)
