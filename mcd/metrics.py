from __future__ import annotations

from typing import List

import numpy as np


def dtop(true_adjacency: np.ndarray, estimated_order: List[int]) -> float:
    """Normalized topological-order error.

    Counts the fraction of true direct edges (parent -> child) whose direction
    is violated by the estimated source-to-sink order. Denominator is the
    number of edges in the true graph. Returns 0 if there are no edges.

    Matches the definition in Rolland et al. (2022), Section 5.1.
    """
    # true_adjacency[child, parent] = 1
    pairs = np.argwhere(true_adjacency)  # (child, parent) pairs
    if pairs.size == 0:
        return 0.0
    pos = {node: i for i, node in enumerate(estimated_order)}
    violations = 0
    for child, parent in pairs:
        # parent should come before child in source-to-sink order
        if pos[int(parent)] > pos[int(child)]:
            violations += 1
    return violations / pairs.shape[0]
