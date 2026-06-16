"""
Shared utilities: YAML config loading and argparse merge helper.

Usage in a training script:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", ...)
    ...
    args = merge_config_and_args(parser)

If the user passes --config configs/baseline.yaml, YAML values are loaded
as defaults and any explicit CLI flags still override them.
"""

import argparse
from pathlib import Path

import torch
import yaml

def get_device() -> torch.device:
    """
    Pick the best available accelerator in priority order:
      CUDA (NVIDIA) > MPS (Apple Silicon) > CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Arguments whose values should be coerced to Path after loading from YAML
_PATH_ARGS = {"data_root", "output_dir"}


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def merge_config_and_args(
    parser: argparse.ArgumentParser, argv=None
) -> argparse.Namespace:
    """
    1. Add --config to the parser.
    2. Do a first-pass parse to find --config (ignoring unknown args).
    3. If --config is set, load the YAML and push values as argparse defaults.
    4. Parse fully; CLI flags override YAML defaults.
    5. Coerce known path-typed keys to pathlib.Path.
    """
    try:
        parser.add_argument(
            "--config", type=Path, default=None,
            metavar="YAML", help="Optional YAML config file (CLI flags override it)",
        )
    except argparse.ArgumentError:
        pass  # already added by caller

    pre_args, _ = parser.parse_known_args(argv)

    if pre_args.config is not None:
        cfg = load_config(pre_args.config)
        # Filter out comment-only keys (None values) and the config key itself
        cfg.pop("config", None)
        parser.set_defaults(**cfg)

    args = parser.parse_args(argv)

    # Coerce path-typed arguments (set_defaults bypasses type= conversion)
    for key in _PATH_ARGS:
        val = getattr(args, key, None)
        if val is not None and not isinstance(val, Path):
            setattr(args, key, Path(val))

    return args
