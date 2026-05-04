"""Small helpers: yaml loading, TB, bootstrap CI, plotting, seeding."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from loguru import logger


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def make_run_dir(root: str | Path, run_name: str) -> Path:
    run_dir = Path(root) / f"{run_name}_{timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def configure_logger(run_dir: Path) -> None:
    log_path = run_dir / "run.log"
    logger.add(log_path, level="INFO")
    logger.info(f"Logging to {log_path}")


def dtype_from_str(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def bootstrap_mean_ci(
    values: np.ndarray,
    num_resamples: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Return (mean, lo, hi) for a two-sided (1-alpha) CI via percentile bootstrap."""
    if rng is None:
        rng = np.random.default_rng(0)
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, n, size=(num_resamples, n))
    means = values[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(values.mean()), lo, hi


def dump_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def dump_json(path: str | Path, obj: dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def env_threads(num: int = 8) -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, str(num))
    torch.set_num_threads(num)
