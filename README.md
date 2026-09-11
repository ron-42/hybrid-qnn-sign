# Hybrid Quantum Channel Attention for ASL Recognition

Controlled evaluation of a simulated variational quantum circuit used as a
channel-attention block in pretrained image classifiers.

[Full project and reproducibility context](AGENTS.md)

## Research result

This repository does **not** demonstrate a quantum advantage.

- QCA uses 17,464 attention parameters, compared with 131,072 for the tested
  squeeze-and-excitation (SE) block.
- In the three-seed frozen-backbone low-data study, SE has the highest mean
  test accuracy at every tested training fraction.
- In the fully unfrozen MU signer-held-out study, no attention has the highest
  equal-signer mean accuracy (94.78%); QCA reaches 94.50%.
- QCA has the highest pooled MU accuracy (94.99%), but signer imbalance changes
  the apparent ranking and paired signer-level tests are not significant.
- Full-data Kaggle results saturate near 100% and are treated as sanity checks,
  not evidence for the attention mechanism.

The supported conclusion is narrow: QCA is parameter-compact relative to SE,
but it does not reliably outperform no attention or a parameter-matched
classical MLP in the completed experiments.

## Repository contents

- `src/`: datasets, models, training, evaluation, and aggregation code
- `configs/`: reproducible YAML configurations
- `runs/*.csv`: canonical run-level and aggregate metrics
- `runs/*.png`: deterministic result figures
- `AGENTS.md`: complete project history, constraints, and remaining decisions

Raw datasets, checkpoints, per-run output directories, and local agent/editor
configuration are intentionally excluded from Git. Manuscript, report, and
paper-writing process artifacts are also kept local.

## Installation

Python 3.10 or newer is recommended.

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

The code selects CUDA, then Apple MPS, then CPU. PennyLane's
`default.qubit` simulator runs on CPU; the classical CNN can run on an
accelerator.

## Data

### Kaggle ASL Alphabet

Download the
[ASL Alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
and place or link it at `asl-dataset/`.

The code ignores the 28-image official test folder and creates a fixed
stratified 80/10/10 split from the 87,000 training images. Horizontal flips
are intentionally disabled because ASL hand configurations are
handedness-sensitive.

### MU HandImages ASL

The original dataset is described by the
[Massey University record](https://mro.massey.ac.nz/items/d822add8-ffe4-4eec-9b5c-3c0aaa1c72be).
The experiment used the public
[`hajarek24/asl-sign-language-recognition`](https://github.com/hajarek24/asl-sign-language-recognition)
mirror at revision
`1e23d649fab0fbb1d4cac523d0c93b1ab06e2213`, placed at
`external-data/mu-handimages/`.

Only the 1,815 A--Z images are used. All primary MU experiments train every
DenseNet layer and hold out one signer per fold. The mirror does not declare a
separate license, so its images are not redistributed here.

## Quick checks

These commands do not require either dataset:

```bash
make smoke-baseline
make smoke-quantum
make smoke-resnet-quantum
make smoke-se
```

## Training and evaluation

Run all commands from the repository root.

```bash
# Full-data models
make baseline DATA_ROOT=asl-dataset
make quantum DATA_ROOT=asl-dataset
make resnet-quantum DATA_ROOT=asl-dataset

# Frozen attribution controls
make quantum-frozen DATA_ROOT=asl-dataset
make se-attention DATA_ROOT=asl-dataset
make no-attention DATA_ROOT=asl-dataset

# One fully unfrozen MU signer-held-out fold
make mu-loso METHOD=qca SIGNER=1 SEED=42 \
  MU_DATA_ROOT=external-data/mu-handimages

# Aggregate completed MU folds
make summarize-mu-loso SEED=42

# Independently evaluate a saved checkpoint
make evaluate CKPT=runs/densenet_quantum/best_model.pt \
  DATA_ROOT=asl-dataset
```

Training outputs are written under `runs/` and ignored by Git. Each run saves
its resolved configuration, history, best validation checkpoint, class map,
and independently evaluated test metrics.

## Public result artifacts

Canonical aggregate evidence included with the code:

- `runs/low_data_results.csv`
- `runs/low_data_summary.csv`
- `runs/mu_loso_results.csv`
- `runs/mu_loso_summary.csv`

## Research constraints

- Do not describe saturated 99--100% scores as evidence for QCA.
- Do not claim superiority, speedup, hardware efficiency, robustness, or
  state-of-the-art performance.
- Frozen-backbone results are controlled ablations, not the primary external
  experiment.
- The external study has five signers and one training seed; its statistical
  power is limited.
- Additional sweeps are deferred until the result and research direction are
  discussed with the supervisor.

## License

No open-source license has been selected yet. Public availability does not
grant permission to reuse or redistribute the code, paper, or artifacts.
