"""
data_utils.py — Shared data-loading helpers for FLEX-ID.

Centralises repeated boilerplate (loading partitions, label encoders, model
weights) so every script uses the same paths and error messages.

Usage
-----
    from utils.data_utils import load_partition, load_label_encoder, load_weights
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Type aliases ───────────────────────────────────────────────────────────────
SplitData = Tuple[
    Tuple[np.ndarray, np.ndarray],  # (X_train, y_train)
    Tuple[np.ndarray, np.ndarray],  # (X_test,  y_test)
]


def load_partition(cid: int, data_dir: str = "data") -> SplitData:
    """Load a client's pre-computed data partition from disk.

    Parameters
    ----------
    cid : int
        Client index (0-based).
    data_dir : str
        Directory that contains the ``client_partition_*.pkl`` files.
        Default is ``"data"``.

    Returns
    -------
    ((X_train, y_train), (X_test, y_test))

    Raises
    ------
    FileNotFoundError
        If the partition file does not exist.
    """
    path = os.path.join(data_dir, f"client_partition_{cid}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Partition file not found: '{path}'. "
            "Run '2_create_partitions.py' first."
        )
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    logger.debug("Loaded partition for client %d from '%s'.", cid, path)
    return data  # type: ignore[return-value]


def load_label_encoder(data_dir: str = "data"):
    """Load the global LabelEncoder saved during partitioning.

    Parameters
    ----------
    data_dir : str
        Directory that contains ``label_encoder.pkl``.

    Returns
    -------
    sklearn.preprocessing.LabelEncoder

    Raises
    ------
    FileNotFoundError
        If the encoder file does not exist.
    """
    from sklearn.preprocessing import LabelEncoder  # lazy import

    path = os.path.join(data_dir, "label_encoder.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Label encoder not found: '{path}'. "
            "Run '2_create_partitions.py' first."
        )
    with open(path, "rb") as fh:
        le: LabelEncoder = pickle.load(fh)
    logger.debug("Loaded label encoder with %d classes.", len(le.classes_))
    return le


def load_weights(weights_path: str) -> list:
    """Load serialised model weights from a pickle file.

    Parameters
    ----------
    weights_path : str
        Path to the ``.pkl`` file produced by the server's ``save_parameters``.

    Returns
    -------
    list of np.ndarray

    Raises
    ------
    FileNotFoundError
        If the weights file does not exist.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: '{weights_path}'.")
    with open(weights_path, "rb") as fh:
        weights = pickle.load(fh)
    logger.debug("Loaded weights from '%s'.", weights_path)
    return weights
