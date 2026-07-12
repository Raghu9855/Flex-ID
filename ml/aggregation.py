"""
aggregation.py — Byzantine-robust aggregation algorithms for FLEX-ID.

Implements four mathematically correct aggregation rules that can be used
as drop-in replacements for FedAvg weight averaging:

  - Krum            : Blanchard et al., NIPS 2017
  - Multi-Krum      : Blanchard et al., NIPS 2017
  - Trimmed Mean    : Yin et al., ICML 2018
  - Coordinate Median : Yin et al., ICML 2018

All functions share the same interface:
    aggregate(weights_list: List[List[np.ndarray]]) -> List[np.ndarray]

References
----------
Blanchard, P., Guerraoui, R., Stainer, J., et al. (2017).
    Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent.
    Advances in Neural Information Processing Systems (NeurIPS), 30.

Yin, D., Chen, Y., Kannan, R., & Bartlett, P. (2018).
    Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates.
    Proceedings of the 35th ICML, 80, 5650–5659.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _flatten(weights: List[np.ndarray]) -> np.ndarray:
    """Concatenate all weight tensors into a single 1-D vector."""
    return np.concatenate([w.ravel() for w in weights])


def _pairwise_l2(flat_list: List[np.ndarray]) -> np.ndarray:
    """Return an (n x n) matrix of squared L2 distances between flat vectors."""
    n = len(flat_list)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sum((flat_list[i] - flat_list[j]) ** 2)
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _weighted_average(
    weights_list: List[List[np.ndarray]],
    selected: List[int],
) -> List[np.ndarray]:
    """Simple mean of weights at *selected* indices."""
    chosen = [weights_list[i] for i in selected]
    return [
        np.mean([w[layer] for w in chosen], axis=0)
        for layer in range(len(chosen[0]))
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Krum
# ──────────────────────────────────────────────────────────────────────────────

def krum_aggregate(
    weights_list: List[List[np.ndarray]],
    num_byzantine: int = 1,
) -> List[np.ndarray]:
    """Krum aggregation (Blanchard et al., 2017).

    Selects the *single* client update whose sum of distances to its
    ``n - num_byzantine - 2`` nearest neighbours is smallest.

    Parameters
    ----------
    weights_list : list of list of np.ndarray
        One weight list per client.
    num_byzantine : int
        Assumed upper bound on the number of Byzantine clients (``f``).

    Returns
    -------
    list of np.ndarray
        The selected client's weight tensors (unchanged — not averaged).

    Notes
    -----
    Requires ``n >= 2f + 3``.  With 4 clients and f=1 this condition is
    NOT satisfied (4 < 5).  Use Multi-Krum (m=2) for 4 clients instead.
    """
    n = len(weights_list)
    f = num_byzantine
    required = 2 * f + 3
    if n < required:
        logger.warning(
            "Krum requires n >= 2f+3 (need %d, have %d). "
            "Falling back to coordinate median.",
            required, n,
        )
        return coordinate_median_aggregate(weights_list)

    flat = [_flatten(w) for w in weights_list]
    dist = _pairwise_l2(flat)
    k = n - f - 2  # number of nearest neighbours to sum

    scores = []
    for i in range(n):
        row = dist[i].copy()
        row[i] = np.inf          # exclude self
        nearest = np.sort(row)[:k]
        scores.append(nearest.sum())

    best = int(np.argmin(scores))
    logger.info("Krum selected client %d (score=%.4f).", best, scores[best])
    return [w.copy() for w in weights_list[best]]


# ──────────────────────────────────────────────────────────────────────────────
# Multi-Krum
# ──────────────────────────────────────────────────────────────────────────────

def multi_krum_aggregate(
    weights_list: List[List[np.ndarray]],
    num_byzantine: int = 1,
    m: int | None = None,
) -> List[np.ndarray]:
    """Multi-Krum aggregation (Blanchard et al., 2017).

    Selects the ``m`` client updates with the lowest Krum scores and
    averages them.  ``m = n - f`` by default.

    Parameters
    ----------
    weights_list : list of list of np.ndarray
        One weight list per client.
    num_byzantine : int
        Assumed number of Byzantine clients (``f``).
    m : int | None
        Number of clients to average.  Defaults to ``n - f``.

    Returns
    -------
    list of np.ndarray
        Layer-wise mean of the ``m`` selected clients.
    """
    n = len(weights_list)
    f = num_byzantine
    if m is None:
        m = max(1, n - f)

    flat = [_flatten(w) for w in weights_list]
    dist = _pairwise_l2(flat)
    k = n - f - 2

    # Ensure k is valid
    k = max(1, min(k, n - 1))

    scores = []
    for i in range(n):
        row = dist[i].copy()
        row[i] = np.inf
        nearest = np.sort(row)[:k]
        scores.append(nearest.sum())

    # Pick m lowest-score indices
    selected = list(np.argsort(scores)[:m])
    logger.info(
        "Multi-Krum selected clients %s (m=%d, f=%d).", selected, m, f
    )
    return _weighted_average(weights_list, selected)


# ──────────────────────────────────────────────────────────────────────────────
# Trimmed Mean
# ──────────────────────────────────────────────────────────────────────────────

def trimmed_mean_aggregate(
    weights_list: List[List[np.ndarray]],
    trim_ratio: float = 0.1,
) -> List[np.ndarray]:
    """Coordinate-wise trimmed mean (Yin et al., 2018).

    For each parameter coordinate, sorts the values across clients, drops
    the top and bottom ``floor(n * trim_ratio)`` values, and returns the
    mean of the remainder.

    Parameters
    ----------
    weights_list : list of list of np.ndarray
        One weight list per client.
    trim_ratio : float
        Fraction of clients to trim from each tail.  Must be in [0, 0.5).
        With 4 clients and trim_ratio=0.1 → k=0 (no trimming); with 8
        clients → k=0 as well; effectively triggers only at n>=10 for
        trim_ratio=0.1.  Use trim_ratio=0.25 for 4 clients (k=1 each side).

    Returns
    -------
    list of np.ndarray
        Layer-wise trimmed-mean tensors.
    """
    n = len(weights_list)
    k = int(n * trim_ratio)   # number of values trimmed from each tail

    if k == 0:
        logger.warning(
            "trim_ratio=%.2f with n=%d clients gives k=0 (no trimming). "
            "Consider trim_ratio >= %.2f.",
            trim_ratio, n, 1.0 / n,
        )

    num_layers = len(weights_list[0])
    aggregated: List[np.ndarray] = []

    for layer_idx in range(num_layers):
        # Stack: shape (n, *layer_shape)
        stack = np.array([weights_list[i][layer_idx] for i in range(n)])
        if k > 0:
            # Sort along client axis, trim tails
            sorted_stack = np.sort(stack, axis=0)
            trimmed = sorted_stack[k : n - k]
        else:
            trimmed = stack
        aggregated.append(trimmed.mean(axis=0))

    logger.info(
        "Trimmed Mean: n=%d, k=%d trimmed per tail, %.1f%% retained.",
        n, k, 100.0 * (n - 2 * k) / n,
    )
    return aggregated


# ──────────────────────────────────────────────────────────────────────────────
# Coordinate-wise Median
# ──────────────────────────────────────────────────────────────────────────────

def coordinate_median_aggregate(
    weights_list: List[List[np.ndarray]],
) -> List[np.ndarray]:
    """Coordinate-wise median (Yin et al., 2018).

    For each parameter coordinate, takes the median value across all
    clients.  More robust than the mean when a minority of clients submit
    adversarial updates.

    Parameters
    ----------
    weights_list : list of list of np.ndarray
        One weight list per client.

    Returns
    -------
    list of np.ndarray
        Layer-wise median tensors.
    """
    num_layers = len(weights_list[0])
    aggregated: List[np.ndarray] = []

    for layer_idx in range(num_layers):
        stack = np.array([weights_list[i][layer_idx] for i in range(len(weights_list))])
        aggregated.append(np.median(stack, axis=0))

    logger.info("Coordinate Median aggregation complete (n=%d).", len(weights_list))
    return aggregated


# ──────────────────────────────────────────────────────────────────────────────
# Registry — maps CLI names to functions
# ──────────────────────────────────────────────────────────────────────────────

AGGREGATION_REGISTRY = {
    "krum": krum_aggregate,
    "multikrum": multi_krum_aggregate,
    "trimmed_mean": trimmed_mean_aggregate,
    "median": coordinate_median_aggregate,
}
