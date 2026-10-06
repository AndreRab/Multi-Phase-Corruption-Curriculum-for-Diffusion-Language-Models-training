from __future__ import annotations

import random


MAX_SEED = 2**32 - 1


def validate_seed(seed: int | None) -> None:
    """Validate the optional experiment seed used by all runners."""

    if seed is None:
        return
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be a non-negative integer or null.")
    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be in the interval [0, {MAX_SEED}].")


def seed_everything(seed: int | None) -> None:
    """Seed Python, NumPy (when available), and PyTorch when requested.

    ``seed=None`` deliberately leaves all random number generators untouched so
    that legacy experiments can be represented as explicitly unseeded runs.
    """

    validate_seed(seed)
    if seed is None:
        return

    random.seed(seed)

    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        np.random.seed(seed)

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
