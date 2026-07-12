"""
seeds.py — Reproducibility seed management for FLEX-ID.

Sets identical seeds for Python's random, NumPy, TensorFlow, and the
OS hash seed so that every experiment produces the same result given the
same data and hyperparameters.

Usage
-----
    from utils.seeds import set_global_seeds
    set_global_seeds()          # uses default RANDOM_SEED = 42
    set_global_seeds(seed=0)    # explicit seed
"""

from __future__ import annotations

import logging
import os
import random

logger = logging.getLogger(__name__)

# ── Global default ─────────────────────────────────────────────────────────────
RANDOM_SEED: int = 42


def set_global_seeds(seed: int = RANDOM_SEED) -> None:
    """Set seeds for Python, NumPy, TensorFlow, and the OS hash.

    Parameters
    ----------
    seed : int
        Seed value to apply globally.  Default is ``RANDOM_SEED`` (42).
    """
    # 1. Python built-in random
    random.seed(seed)

    # 2. OS-level hash randomisation (affects dict/set ordering in Python 3.3+)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 3. NumPy
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        logger.warning("NumPy not installed; skipping np.random.seed.")

    # 4. TensorFlow / Keras
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        logger.warning("TensorFlow not installed; skipping tf.random.set_seed.")

    logger.info("Global seeds set to %d.", seed)
