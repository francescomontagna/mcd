from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class DAGConfig:
    """Configuration for synthetic DAG generation.

    The code uses the convention A[child, parent] = 1 for parent -> child.
    """

    d: int = 5
    density: float = 0.3
    graph_type: str = "erdos_renyi"  # {erdos_renyi, chain, empty}


def topological_order(adjacency: np.ndarray) -> List[int]:
    """Return a topological order for A[child, parent] = 1."""
    A = np.asarray(adjacency)
    d = A.shape[0]
    indeg = A.sum(axis=1).astype(int)
    active = [i for i in range(d) if indeg[i] == 0]
    order: List[int] = []

    while active:
        node = active.pop(0)
        order.append(node)
        children = np.where(A[:, node] != 0)[0]
        for child in children:
            indeg[child] -= 1
            if indeg[child] == 0:
                active.append(int(child))

    if len(order) != d:
        raise ValueError("The adjacency matrix is not acyclic.")
    return order


def sample_erdos_renyi_dag(
    d: int,
    density: float,
    rng: np.random.Generator,
    order: Optional[Iterable[int]] = None,
) -> Tuple[np.ndarray, List[int]]:
    """Sample an Erdos-Renyi DAG with controllable node count and edge density.

    Parameters
    ----------
    d:
        Number of nodes.
    density:
        Probability of each admissible edge in a random topological ordering.
    rng:
        NumPy random generator.
    order:
        Optional topological order. If omitted, a random order is sampled.

    Returns
    -------
    adjacency, order:
        adjacency[child, parent] = 1 means parent -> child.
    """
    if not 0 <= density <= 1:
        raise ValueError("density must be in [0, 1].")
    if order is None:
        topo = [int(x) for x in rng.permutation(d)]
    else:
        topo = [int(x) for x in order]
        if sorted(topo) != list(range(d)):
            raise ValueError("order must be a permutation of range(d).")

    A = np.zeros((d, d), dtype=int)
    for pos_parent in range(d):
        parent = topo[pos_parent]
        for pos_child in range(pos_parent + 1, d):
            child = topo[pos_child]
            if rng.random() < density:
                A[child, parent] = 1
    return A, topo


def sample_chain_dag(d: int) -> Tuple[np.ndarray, List[int]]:
    """Return the chain 0 -> 1 -> ... -> d-1."""
    A = np.zeros((d, d), dtype=int)
    for parent in range(d - 1):
        A[parent + 1, parent] = 1
    return A, list(range(d))


def sample_dag(config: DAGConfig, rng: np.random.Generator) -> Tuple[np.ndarray, List[int]]:
    """Sample a DAG according to a small configuration object."""
    graph_type = config.graph_type.lower()
    if graph_type in {"er", "erdos", "erdos_renyi"}:
        return sample_erdos_renyi_dag(config.d, config.density, rng)
    if graph_type == "chain":
        return sample_chain_dag(config.d)
    if graph_type == "empty":
        return np.zeros((config.d, config.d), dtype=int), list(range(config.d))
    raise ValueError(f"Unknown graph_type: {config.graph_type}")


def descendants_matrix(adjacency: np.ndarray) -> np.ndarray:
    """Transitive closure: R[descendant, ancestor] = True iff ancestor ->* descendant."""
    A = np.asarray(adjacency).astype(bool)
    d = A.shape[0]
    R = A.copy()
    # Floyd-Warshall with child,parent convention.
    for mid in range(d):
        # If mid is an ancestor of row and col is an ancestor of mid, then col is ancestor of row.
        rows = np.where(R[:, mid])[0]
        cols = np.where(R[mid, :])[0]
        if rows.size and cols.size:
            R[np.ix_(rows, cols)] = True
    np.fill_diagonal(R, False)
    return R


def full_dag_from_order(order: List[int]) -> np.ndarray:
    """Dense DAG induced by an order; A[child, parent] = 1."""
    d = len(order)
    A = np.zeros((d, d), dtype=int)
    for i, parent in enumerate(order):
        for child in order[i + 1 :]:
            A[child, parent] = 1
    return A
