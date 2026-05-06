from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Union

import numpy as np

Array = np.ndarray

_N_RFF = 32
_GP_LENGTH_SCALE = 2


class NodeMechanism:
    """Base class for mechanisms x_i = F_i(x_pa(i), s_i)."""

    name: str = "base"

    def __call__(self, parents: Array, source: Array) -> Array:
        raise NotImplementedError

    def metadata(self) -> Dict[str, object]:
        return {"name": self.name}


@dataclass
class RootMechanism(NodeMechanism):
    name: str = "root"

    def __call__(self, parents: Array, source: Array) -> Array:
        return source


# ---------------------------------------------------------------------------
# Additive noise mechanisms
# ---------------------------------------------------------------------------


@dataclass
class LinearAdditive(NodeMechanism):
    weights: Array
    name: str = "linear"

    def __call__(self, parents: Array, source: Array) -> Array:
        return parents @ self.weights + source

    def metadata(self) -> Dict[str, object]:
        return {"name": self.name, "weights": self.weights.tolist()}


@dataclass
class NonlinearAdditive(NodeMechanism):
    """x = sin(pa @ w) + s"""
    weights: Array
    name: str = "anm_tanh"

    def __call__(self, parents: Array, source: Array) -> Array:
        return np.sin(parents @ self.weights) + source

    def metadata(self) -> Dict[str, object]:
        return {"name": self.name, "weights": self.weights.tolist()}


@dataclass
class PostNonlinear(NodeMechanism):
    """x = z + c*tanh(z)  where  z = tanh(pa @ w) + s"""
    weights: Array
    c: float
    name: str = "pnl"

    def __call__(self, parents: Array, source: Array) -> Array:
        latent = np.tanh(parents @ self.weights) + source
        return latent + self.c * np.tanh(latent)

    def metadata(self) -> Dict[str, object]:
        return {"name": self.name, "weights": self.weights.tolist(), "c": self.c}


@dataclass
class LocationScaleNoise(NodeMechanism):
    """x = tanh(pa @ g) + h(pa) * (s + eta*s^3)"""
    weights_g: Array
    weights_h: Array
    eta: float = 0.05
    hmin: float = 0.7
    hmax: float = 1.3
    name: str = "lsnm"

    def _gain(self, parents: Array) -> Array:
        t = parents @ self.weights_h
        sig = 1.0 / (1.0 + np.exp(-np.clip(t, -40, 40)))
        return self.hmin + (self.hmax - self.hmin) * sig

    def __call__(self, parents: Array, source: Array) -> Array:
        g = np.tanh(parents @ self.weights_g)
        h = self._gain(parents)
        return g + h * (source + self.eta * source ** 3)

    def metadata(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "weights_g": self.weights_g.tolist(),
            "weights_h": self.weights_h.tolist(),
            "eta": self.eta,
        }


# ---------------------------------------------------------------------------
# Random neural network mechanism
# ---------------------------------------------------------------------------


@dataclass
class RandomNNMechanism(NodeMechanism):
    """Random invertible NN: x = LayerNorm(MLP([pa, s]))[0].

    Each layer is a square linear map followed by an invertible nonlinearity
    (tanh or leaky_relu).  LayerNorm on the final hidden state keeps the
    output scale bounded regardless of depth or weight scale.
    """
    weights: List[Array]        # depth × (dim, dim), dim = n_parents + 1
    nonlinearities: List[str]   # depth × {'tanh' | 'leaky_relu'}
    leaky_alpha: float = 0.2
    residual: bool = False
    name: str = "random_nn"

    def _nonlin(self, z: Array, nl: str) -> Array:
        if nl == "tanh":
            return np.tanh(z)
        return np.where(z > 0, z, self.leaky_alpha * z)

    def __call__(self, parents: Array, source: Array) -> Array:
        z = np.concatenate([parents, source[:, None]], axis=1)  # (n, dim)
        for W, nl in zip(self.weights, self.nonlinearities):
            h = self._nonlin(z @ W, nl)
            z = h + z if self.residual else h
        mu = z.mean(axis=0, keepdims=True)
        sig = z.std(axis=0, keepdims=True) + 1e-8
        z = (z - mu) / sig
        return z[:, 0]

    def metadata(self) -> Dict[str, object]:
        return {"name": self.name, "depth": len(self.weights), "residual": self.residual}


# ---------------------------------------------------------------------------
# Non-additive mechanisms
# ---------------------------------------------------------------------------


@dataclass
class ANMGPMechanism(NodeMechanism):
    """GP-parameterized ANM: x = GP(pa) + s (RFF approximation)."""
    omega: Array   # (n_features, n_parents)
    phase: Array   # (n_features,)
    alpha: Array   # (n_features,)
    name: str = "anm_gp"

    def _rff(self, x: Array) -> Array:
        n_feat = self.omega.shape[0]
        return np.sqrt(2.0 / n_feat) * np.cos(x @ self.omega.T + self.phase) @ self.alpha

    def __call__(self, parents: Array, source: Array) -> Array:
        return self._rff(parents) + source

    def metadata(self) -> Dict[str, object]:
        return {"name": self.name}


@dataclass
class GPMechanism(NodeMechanism):
    """GP-parameterized non-additive: x = GP([pa, s]) 
    """
    omega: Array   # (n_features, n_parents + 1)
    phase: Array   # (n_features,)
    alpha: Array   # (n_features,)
    c_s: float
    name: str = "gp"

    def _rff(self, x: Array) -> Array:
        n_feat = self.omega.shape[0]
        return np.sqrt(2.0 / n_feat) * np.cos(x @ self.omega.T + self.phase) @ self.alpha

    def __call__(self, parents: Array, source: Array) -> Array:
        z = np.concatenate([parents, source[:, None]], axis=1)
        return self._rff(z) + self.c_s * source

    def metadata(self) -> Dict[str, object]:
        return {"name": self.name}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_ADDITIVE_POOL = ["linear", "anm_tanh", "pnl", "lsnm", "anm_gp"]
_NONADDITIVE_POOL = ["gp"]
# _NOT_OBS_ID_POOL = ["linear", "gp"]
_NOT_OBS_ID_POOL = ["linear", "nn_residual"]
_OBS_ID_POOL = ["anm_tanh", "pnl", "lsnm", "anm_gp"]


def build_mechanism(name: Union[str, NodeMechanism], n_parents: int, rng: np.random.Generator) -> NodeMechanism:
    """Build a predefined mechanism or pass through a custom NodeMechanism."""
    if isinstance(name, NodeMechanism):
        return name
    if n_parents == 0:
        return RootMechanism()

    key = str(name).lower()

    if key in {"linear", "linear_additive"}:
        return LinearAdditive(weights=rng.uniform(-1.5, 1.5, size=n_parents))

    if key in {"anm", "anm_tanh", "tanh", "nonlinear_additive"}:
        return NonlinearAdditive(weights=rng.uniform(-1.5, 1.5, size=n_parents))

    if key == "pnl":
        return PostNonlinear(
            weights=rng.uniform(-1.5, 1.5, size=n_parents),
            c=float(rng.uniform(-0.7, 0.7)),
        )

    if key in {"lsnm", "location_scale", "location_scale_cubic"}:
        return LocationScaleNoise(
            weights_g=rng.uniform(-1.0, 1.0, size=n_parents),
            weights_h=rng.uniform(-1.0, 1.0, size=n_parents),
            eta=float(rng.uniform(0.02, 0.08)),
        )

    if key == "anm_gp":
        return ANMGPMechanism(
            omega=rng.normal(0.0, 1.0 / _GP_LENGTH_SCALE, size=(_N_RFF, n_parents)),
            phase=rng.uniform(0.0, 2.0 * np.pi, size=_N_RFF),
            alpha=rng.normal(0.0, 1.0, size=_N_RFF),
        )

    if key == "gp":
        return GPMechanism(
            omega=rng.normal(0.0, 1.0 / _GP_LENGTH_SCALE, size=(_N_RFF, n_parents + 1)),
            phase=rng.uniform(0.0, 2.0 * np.pi, size=_N_RFF),
            alpha=rng.normal(0.0, 1.0, size=_N_RFF),
            c_s=float(rng.uniform(0.3, 0.7)), # Residual connection from source to output, needed for stability when not additive
        )

    if key in {"nn", "nn_residual"}:
        dim = n_parents + 1
        weights = [rng.normal(1.0, 1.0, (dim, dim)) for _ in range(4)]
        nonlins = ["tanh"] * 4
        return RandomNNMechanism(weights=weights, nonlinearities=nonlins, residual=(key == "nn_residual"))

    if key in {"mixed", "mixed_mechanism"}:
        pool = _NONADDITIVE_POOL if rng.random() < 0.5 else _ADDITIVE_POOL
        chosen = str(rng.choice(pool))
        return build_mechanism(chosen, n_parents, rng)

    if key in {"not_obs_id", "non_obs_id"}:
        chosen = str(rng.choice(_NOT_OBS_ID_POOL))
        return build_mechanism(chosen, n_parents, rng)

    if key == "obs_id":
        chosen = str(rng.choice(_OBS_ID_POOL))
        return build_mechanism(chosen, n_parents, rng)

    raise ValueError(f"Unknown mechanism: {name}")


def build_mechanisms(
    adjacency: Array,
    mechanism: Union[str, Sequence[Union[str, NodeMechanism]], Callable[[int, Array, np.random.Generator], NodeMechanism]],
    rng: np.random.Generator,
) -> List[NodeMechanism]:
    """Build one mechanism per node.

    mechanism can be:
      * a string, used for every node;
      * a list of strings/NodeMechanisms, one per node;
      * a callable (node, parents, rng) -> NodeMechanism.
    """
    d = adjacency.shape[0]
    mechs: List[NodeMechanism] = []
    for node in range(d):
        parents = np.where(adjacency[node] != 0)[0]
        if callable(mechanism) and not isinstance(mechanism, str):
            mech = mechanism(node, parents, rng)
        elif isinstance(mechanism, Sequence) and not isinstance(mechanism, str):
            mech = build_mechanism(mechanism[node], len(parents), rng)
        else:
            mech = build_mechanism(str(mechanism), len(parents), rng)
        mechs.append(mech)
    return mechs
