from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .graphs import DAGConfig, sample_dag, topological_order
from .mechanisms import NodeMechanism, build_mechanisms


@dataclass
class EnvShiftConfig:
    """Location-scale shifts for source distributions across environments."""

    intervention_frac: float = 1.0
    mu_range: Tuple[float, float] = (-1.0, 1.0)
    sigma_range: Tuple[float, float] = (0.75, 1.35)
    ensure_first_two_strong: bool = True
    first_two_mu: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.7, 1.2), (-1.2, -0.7))
    first_two_sigma: Tuple[float, float] = (0.8, 1.25)
    student_t_df: int = 6


@dataclass
class SCMData:
    X: np.ndarray  # (k, n, d)
    S: np.ndarray  # (k, n, d)
    adjacency: np.ndarray  # A[child, parent] = 1
    order: List[int]
    mechanisms: List[NodeMechanism]
    env_params: Dict[str, Any]


def _standardized_noise(n: int, d: int, dist: str, rng: np.random.Generator, cfg: EnvShiftConfig) -> np.ndarray:
    dist = dist.lower()
    if dist in {"gaussian", "normal"}:
        return rng.normal(0.0, 1.0, size=(n, d))
    if dist == "laplace":
        return rng.laplace(0.0, 1 / np.sqrt(2), size=(n, d))
    if dist in {"student_t", "t"}:
        df = cfg.student_t_df
        u = rng.standard_t(df, size=(n, d))
        if df > 2:
            u = u / np.sqrt(df / (df - 2))
        return u
    if dist == "gamma":
        shape, scale = 2.0, 1.0
        u = rng.gamma(shape, scale, size=(n, d))
        return (u - shape * scale) / np.sqrt(shape * scale**2)
    if dist == "exponential":
        return rng.exponential(1.0, size=(n, d)) - 1.0
    raise ValueError(f"Unknown noise distribution: {dist}")


# NOTE: i am unhappy of how this is handled. We need to make interventions more effective
def sample_sources(
    k: int,
    n: int,
    d: int,
    noise: Union[str, Sequence[str]],
    rng: np.random.Generator,
    shift: Optional[EnvShiftConfig] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Sample independent source variables for k environments."""
    if shift is None:
        shift = EnvShiftConfig()
    noises = [noise] * k if isinstance(noise, str) else list(noise)
    if len(noises) < k:
        noises = noises + [noises[-1]] * (k - len(noises))

    mu = np.zeros((k, d), dtype=float)
    sigma = np.ones((k, d), dtype=float)

    m = max(1, int(round(shift.intervention_frac * d)))
    m = min(m, d)

    def choose_nodes(exclude: Optional[np.ndarray] = None) -> np.ndarray:
        nodes = np.arange(d)
        if exclude is not None and exclude.size > 0:
            nodes = np.setdiff1d(nodes, exclude, assume_unique=False)
        if nodes.size <= m:
            return nodes
        return rng.choice(nodes, size=m, replace=False)

    if k > 1:
        if shift.ensure_first_two_strong:
            idx1 = choose_nodes()
            lo, hi = shift.first_two_mu[0]
            slo, shi = shift.first_two_sigma
            mu[1, idx1] = rng.uniform(lo, hi, size=idx1.size)
            sigma[1, idx1] = rng.uniform(slo, shi, size=idx1.size)

            if k > 2:
                # If intervention_frac=1, env1 and env2 both shift all nodes.
                exclude = idx1 if shift.intervention_frac < 0.999 else None
                idx2 = choose_nodes(exclude=exclude)
                lo, hi = shift.first_two_mu[1]
                mu[2, idx2] = rng.uniform(lo, hi, size=idx2.size)
                sigma[2, idx2] = rng.uniform(slo, shi, size=idx2.size)

            start = 3
        else:
            start = 1

        for e in range(start, k):
            idx = choose_nodes()
            mu[e, idx] = rng.uniform(*shift.mu_range, size=idx.size)
            sigma[e, idx] = rng.uniform(*shift.sigma_range, size=idx.size)

    S_envs = []
    for e in range(k):
        u = _standardized_noise(n, d, noises[e], rng, shift) # NOTE: do I want standardization?
        S_envs.append(mu[e] + sigma[e] * u)
    S = np.stack(S_envs, axis=0)
    return S, {"mu": mu, "sigma": sigma, "noise": noises, "shift_config": shift.__dict__}


def sample_from_scm(
    adjacency: np.ndarray,
    order: List[int],
    mechanisms: Sequence[NodeMechanism],
    sources: np.ndarray,
) -> np.ndarray:
    """Push source samples through the SCM in topological order."""
    k, n, d = sources.shape
    X = np.zeros((k, n, d), dtype=float)
    for e in range(k):
        for node in order:
            parents = np.where(adjacency[node] != 0)[0]
            if parents.size == 0:
                X[e, :, node] = sources[e, :, node]
            else:
                X[e, :, node] = mechanisms[node](X[e][:, parents], sources[e, :, node])
    return X


def sample_multienv_scm(
    k: int,
    n: int,
    adjacency: Optional[np.ndarray] = None,
    order: Optional[List[int]] = None,
    dag_config: Optional[DAGConfig] = None,
    mechanism: Union[str, Sequence[Union[str, NodeMechanism]]] = "mixed",
    noise: Union[str, Sequence[str]] = "gaussian",
    rng: Optional[np.random.Generator] = None,
    shift: Optional[EnvShiftConfig] = None,
) -> SCMData:
    """End-to-end synthetic data generation.

    This is the main entry point. It supports controllable graph size/density,
    modular mechanisms, edge-level mixed mechanisms, custom mechanisms, and an
    arbitrary number of environments.
    """
    if rng is None:
        rng = np.random.default_rng()
    if adjacency is None:
        if dag_config is None:
            dag_config = DAGConfig()
        adjacency, order = sample_dag(dag_config, rng)
    else:
        adjacency = np.asarray(adjacency).astype(int)
        if order is None:
            order = topological_order(adjacency)
    if order is None:
        order = topological_order(adjacency)

    d = adjacency.shape[0]
    mechanisms = build_mechanisms(adjacency, mechanism, rng)
    S, env_params = sample_sources(k, n, d, noise, rng, shift)
    X = sample_from_scm(adjacency, order, mechanisms, S)
    return SCMData(X=X, S=S, adjacency=adjacency, order=order, mechanisms=mechanisms, env_params=env_params)
