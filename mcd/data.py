from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .graphs import DAGConfig, sample_dag, topological_order
from .mechanisms import NodeMechanism, build_mechanisms


# ---------------------------------------------------------------------------
# Exponential family specifications
# ---------------------------------------------------------------------------
# Gaussian N(mu, sigma^2):  natural params eta = [eta1, eta2]
#   eta1 = mu / sigma^2,   eta2 = -1 / (2 * sigma^2)
#   base distribution N(0, 1):  eta_base = [0.0, -0.5]
#   most_meaningful_idx = 1  (eta2 controls variance)
#   std_ranges: sampling intervals in (mu, sigma) space for auxiliary envs
#               sigma range [0.5, 4.0] gives eta2 in [-2.0, -0.0625]

_FAMILY_SPECS: Dict[str, Dict[str, Any]] = {
    "gaussian": {
        "n_params": 2,
        "base_eta": np.array([0.0, -0.5]),
        "most_meaningful_idx": 1,
        "std_ranges": [(-2.0, 2.0), (0.5, 4.0)],  # (mu_range, sigma_range)
    },
    # Exponential Exp(lambda):  p(x|eta) = exp(eta*x - log(-eta))  for x >= 0
    #   eta = -lambda  (eta < 0),  base Exp(1): eta_base = [-1.0]
    #   lambda range [0.5, 4.0] <-> eta range [-4.0, -0.5]
    "exponential": {
        "n_params": 1,
        "base_eta": np.array([-1.0]),
        "most_meaningful_idx": 0,
        "std_ranges": [(0.5, 4.0)],  # lambda_range
    },
}


def _eta_to_std_gaussian(eta: np.ndarray) -> Tuple[float, float]:
    sigma2 = -1.0 / (2.0 * eta[1])
    mu = -eta[0] / (2.0 * eta[1])
    return float(mu), float(np.sqrt(sigma2))


def _std_to_eta_gaussian(mu: float, sigma: float) -> np.ndarray:
    sigma2 = sigma ** 2
    return np.array([mu / sigma2, -1.0 / (2.0 * sigma2)])


def _draw_aux_eta_gaussian(
    base_eta: np.ndarray,
    std_ranges: List[Tuple[float, float]],
    shift_all: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    sigma = rng.uniform(std_ranges[1][0], std_ranges[1][1])
    if shift_all:
        mu = rng.uniform(std_ranges[0][0], std_ranges[0][1])
    else:
        mu, _ = _eta_to_std_gaussian(base_eta)
    return _std_to_eta_gaussian(mu, sigma)


def _draw_aux_eta_exponential(std_ranges: List[Tuple[float, float]], rng: np.random.Generator) -> np.ndarray:
    lam = rng.uniform(std_ranges[0][0], std_ranges[0][1])
    return np.array([-lam])


def _draw_aux_eta(cfg: ExpFamilyConfig, rng: np.random.Generator) -> np.ndarray:
    spec = _FAMILY_SPECS[cfg.family]
    if cfg.family == "gaussian":
        return _draw_aux_eta_gaussian(spec["base_eta"], spec["std_ranges"], cfg.shift_all_eta, rng)
    if cfg.family == "exponential":
        return _draw_aux_eta_exponential(spec["std_ranges"], rng)
    raise ValueError(f"Unknown exponential family: {cfg.family!r}")


def _sample_from_eta(family: str, eta: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if family == "gaussian":
        mu, sigma = _eta_to_std_gaussian(eta)
        return rng.normal(mu, sigma, size=n)
    if family == "exponential":
        lam = -eta[0]
        return rng.exponential(scale=1.0 / lam, size=n)
    raise ValueError(f"Unknown family: {family!r}")


def _node_dist(family: str, eta_a: np.ndarray, eta_b: np.ndarray, shift_all: bool) -> float:
    if family == "gaussian":
        mu_a, sigma_a = _eta_to_std_gaussian(eta_a)
        mu_b, sigma_b = _eta_to_std_gaussian(eta_b)
        if shift_all:
            return float(np.sqrt((mu_a - mu_b) ** 2 + (sigma_a - sigma_b) ** 2))
        return abs(sigma_a - sigma_b)
    if family == "exponential":
        # distance in lambda space; shift_all irrelevant (single parameter)
        return abs(float(-eta_a[0] - (-eta_b[0])))
    raise ValueError(f"Unknown family: {family!r}")


def _pairwise_env_distances(
    eta: np.ndarray,  # (k, d, n_params)
    family: str,
    shift_all: bool,
) -> np.ndarray:
    k, d = eta.shape[:2]
    dist = np.zeros((k, k))
    for e1, e2 in combinations(range(k), 2):
        node_dists = [_node_dist(family, eta[e1, j], eta[e2, j], shift_all) for j in range(d)]
        dist[e1, e2] = dist[e2, e1] = float(np.mean(node_dists))
    return dist


def _is_diverse_enough(
    eta: np.ndarray,
    family: str,
    shift_all: bool,
    min_env_diff: float,
    min_sufficient_envs: int,
) -> bool:
    """True if at least min_sufficient_envs environments are mutually >= min_env_diff apart."""
    k = eta.shape[0]
    if min_sufficient_envs > k:
        return True
    dist = _pairwise_env_distances(eta, family, shift_all)
    sufficient = dist >= min_env_diff
    for subset in combinations(range(k), min_sufficient_envs):
        if all(sufficient[e1, e2] for e1, e2 in combinations(subset, 2)):
            return True
    return False


@dataclass
class ExpFamilyConfig:
    """Exponential family source distribution configuration."""
    family: str = "gaussian"
    shift_all_eta: bool = False
    min_env_diff: float = 1.0
    min_sufficient_envs: int = 3
    max_retries: int = 500


@dataclass
class SCMData:
    X: np.ndarray  # (k, n, d)
    S: np.ndarray  # (k, n, d)
    adjacency: np.ndarray  # A[child, parent] = 1
    order: List[int]
    mechanisms: List[NodeMechanism]
    env_params: Dict[str, Any]


def sample_sources(
    k: int,
    n: int,
    d: int,
    cfg: ExpFamilyConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Sample k environments of independent source variables from the exponential family.

    Auxiliary environments (1..k-1) have per-node natural parameters eta sampled
    uniformly in standard-parameter space (mu, sigma for Gaussian) then converted
    to natural parameters.  A rejection loop ensures at least cfg.min_sufficient_envs
    environments are mutually separated by >= cfg.min_env_diff (in standard-param space).
    """
    spec = _FAMILY_SPECS[cfg.family]
    base_eta: np.ndarray = spec["base_eta"]

    eta_all = np.tile(base_eta, (k, d, 1))  # (k, d, n_params); env 0 always uses base params
    for _ in range(cfg.max_retries):
        eta_all = np.tile(base_eta, (k, d, 1))
        for e in range(1, k):
            for j in range(d):
                eta_all[e, j] = _draw_aux_eta(cfg, rng)
        if _is_diverse_enough(eta_all, cfg.family, cfg.shift_all_eta, cfg.min_env_diff, cfg.min_sufficient_envs):
            break
    else:
        warnings.warn(
            f"Could not satisfy min_env_diff={cfg.min_env_diff} within {cfg.max_retries} retries; "
            "using last sampled configuration."
        )

    S = np.zeros((k, n, d))
    for e in range(k):
        for j in range(d):
            S[e, :, j] = _sample_from_eta(cfg.family, eta_all[e, j], n, rng)

    return S, {"eta": eta_all, "family": cfg.family, "shift_all_eta": cfg.shift_all_eta}


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
    rng: Optional[np.random.Generator] = None,
    exp_family_cfg: Optional[ExpFamilyConfig] = None,
) -> SCMData:
    """End-to-end synthetic data generation for multi-environment causal discovery."""
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
    if exp_family_cfg is None:
        exp_family_cfg = ExpFamilyConfig()

    d = adjacency.shape[0]
    mechanisms = build_mechanisms(adjacency, mechanism, rng)
    S, env_params = sample_sources(k, n, d, exp_family_cfg, rng)
    X = sample_from_scm(adjacency, order, mechanisms, S)
    return SCMData(X=X, S=S, adjacency=adjacency, order=order, mechanisms=mechanisms, env_params=env_params)
