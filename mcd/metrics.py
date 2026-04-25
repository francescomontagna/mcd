from __future__ import annotations

from typing import List

import numpy as np

from .graphs import descendants_matrix


def dtop(true_adjacency: np.ndarray, estimated_order: List[int]) -> float:
    """Normalized topological-order error.

    Counts the fraction of true ancestor-before-descendant relations violated by
    the estimated source-to-sink order. If the true graph has no comparable
    pairs, returns 0.
    """
    R = descendants_matrix(true_adjacency)  # R[descendant, ancestor]
    pairs = np.argwhere(R)
    if pairs.size == 0:
        return 0.0
    pos = {node: i for i, node in enumerate(estimated_order)}
    violations = 0
    for desc, anc in pairs:
        if pos[int(anc)] > pos[int(desc)]:
            violations += 1
    return violations / pairs.shape[0]
