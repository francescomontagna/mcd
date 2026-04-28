from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Union

import numpy as np

Array = np.ndarray

_N_HIDDEN = 16
_N_RFF = 32
_GP_LENGTH_SCALE = 1.0


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
    """x = tanh(pa @ w) + s"""
    weights: Array
    name: str = "anm_tanh"

    def __call__(self, parents: Array, source: Array) -> Array:
        return np.tanh(parents @ self.weights) + source

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
# Non-additive mechanisms
# ---------------------------------------------------------------------------


@dataclass
class TanhCubic(NodeMechanism):
    """x = tanh((pa @ w_pa + w_s * s)^3) — linear mixing of parents and source, cube, then tanh.

    Invertible: u = pa @ w_pa + w_s * s is linear in s; u^3 is bijective on R;
    tanh is injective. Given x: u = cbrt(atanh(x)), s = (u - pa @ w_pa) / w_s.
    w_s is kept strictly positive at construction time.
    """
    weights_pa: Array  # (n_parents,)
    weight_s: float    # strictly positive
    name: str = "tanh_cubic"

    def __call__(self, parents: Array, source: Array) -> Array:
        u = parents @ self.weights_pa + self.weight_s * source
        return np.tanh(u ** 3)

    def metadata(self) -> Dict[str, object]:
        return {"name": self.name, "weights_pa": self.weights_pa.tolist(), "weight_s": self.weight_s}


@dataclass
class NeuralNetMechanism(NodeMechanism):
    """Two-layer MLP on (parents, source) — non-additive, invertible in source.

    Invertible: abs(W2) and abs(w_s) are both positive, so each hidden unit
    contributes a strictly increasing function of s. Their sum is strictly
    increasing, hence x is a bijection in s for every fixed pa.
    """
    W_pa: Array  # (n_hidden, n_parents) — parent weights, unconstrained
    w_s: Array   # (n_hidden,) — source weights; abs() enforced in forward pass
    b1: Array    # (n_hidden,)
    W2: Array    # (n_hidden,) — output weights; abs() enforced in forward pass
    name: str = "neural_net"

    def __call__(self, parents: Array, source: Array) -> Array:
        h = np.tanh(parents @ self.W_pa.T + source[:, None] * np.abs(self.w_s) + self.b1)
        return h @ np.abs(self.W2)

    def metadata(self) -> Dict[str, object]:
        return {"name": self.name}


@dataclass
class GPMechanism(NodeMechanism):
    """GP-parameterized location-scale: x = mu(pa) + exp(log_sigma(pa)) * s.

    mu and log_sigma are independent RFF approximations of GP priors over parents.
    Invertible: exp(log_sigma(pa)) > 0 always, so s = (x - mu(pa)) * exp(-log_sigma(pa)).
    """
    omega_mu: Array   # (n_features, n_parents)
    phase_mu: Array   # (n_features,)
    alpha_mu: Array   # (n_features,)
    omega_ls: Array   # (n_features, n_parents)
    phase_ls: Array   # (n_features,)
    alpha_ls: Array   # (n_features,)
    name: str = "gp"

    def _rff(self, parents: Array, omega: Array, phase: Array, alpha: Array) -> Array:
        n_feat = omega.shape[0]
        z = np.sqrt(2.0 / n_feat) * np.cos(parents @ omega.T + phase)
        return z @ alpha

    def __call__(self, parents: Array, source: Array) -> Array:
        mu = self._rff(parents, self.omega_mu, self.phase_mu, self.alpha_mu)
        log_sigma = self._rff(parents, self.omega_ls, self.phase_ls, self.alpha_ls)
        return mu + np.exp(log_sigma) * source

    def metadata(self) -> Dict[str, object]:
        return {"name": self.name}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_ADDITIVE_POOL = ["linear", "anm_tanh", "pnl", "lsnm"]
_NONADDITIVE_POOL = ["tanh_cubic", "neural_net", "gp"]


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

    if key == "tanh_cubic":
        return TanhCubic(
            weights_pa=rng.uniform(-1.5, 1.5, size=n_parents),
            weight_s=float(rng.uniform(0.5, 1.5)),
        )

    if key in {"nn", "neural_net"}:
        return NeuralNetMechanism(
            W_pa=rng.normal(0.0, 1.0 / np.sqrt(n_parents), size=(_N_HIDDEN, n_parents)),
            w_s=rng.uniform(0.5, 1.5, size=_N_HIDDEN),
            b1=np.zeros(_N_HIDDEN),
            # Divide by _N_HIDDEN so sum|W2| ≈ 1, keeping the output O(1) at any depth.
            W2=rng.uniform(0.5, 1.5, size=_N_HIDDEN) / _N_HIDDEN,
        )

    if key == "gp":
        return GPMechanism(
            omega_mu=rng.normal(0.0, 1.0 / _GP_LENGTH_SCALE, size=(_N_RFF, n_parents)),
            phase_mu=rng.uniform(0.0, 2.0 * np.pi, size=_N_RFF),
            alpha_mu=rng.normal(0.0, 1.0, size=_N_RFF),
            omega_ls=rng.normal(0.0, 1.0 / _GP_LENGTH_SCALE, size=(_N_RFF, n_parents)),
            phase_ls=rng.uniform(0.0, 2.0 * np.pi, size=_N_RFF),
            alpha_ls=rng.normal(0.0, 0.5, size=_N_RFF),
        )

    if key in {"mixed", "mixed_mechanism"}:
        pool = _NONADDITIVE_POOL if rng.random() < 0.5 else _ADDITIVE_POOL
        chosen = str(rng.choice(pool))
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
