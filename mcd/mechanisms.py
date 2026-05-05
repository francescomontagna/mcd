from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Union

import numpy as np

Array = np.ndarray

_N_HIDDEN = 32
_N_DEPTH = 8
_N_RFF = 32
_GP_LENGTH_SCALE = 2.0


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
class NeuralNetANM(NodeMechanism):
    """x = deep_MLP(pa) + s — _N_DEPTH-layer MLP with LayerNorm before output.

    Architecture: (_N_DEPTH-1) tanh hidden layers, then LayerNorm, then linear output.
    """
    hidden_weights: List[Array]  # len = _N_DEPTH-1; shapes (n_hidden, n_in), (n_hidden, n_hidden), ...
    hidden_biases: List[Array]   # len = _N_DEPTH-1; each (n_hidden,)
    out_weight: Array            # (n_hidden,)
    ln_gamma: Array              # (n_hidden,) LayerNorm scale
    ln_beta: Array               # (n_hidden,) LayerNorm shift
    name: str = "anm_nn"

    def __call__(self, parents: Array, source: Array) -> Array:
        h = parents
        for W, b in zip(self.hidden_weights, self.hidden_biases):
            h = np.tanh(h @ W.T + b)
        mu = h.mean(axis=-1, keepdims=True)
        std = h.std(axis=-1, keepdims=True) + 1e-5
        h = (h - mu) / std * self.ln_gamma + self.ln_beta
        return h @ self.out_weight + source

    def metadata(self) -> Dict[str, object]:
        return {"name": self.name}


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
class NeuralNetMechanism(NodeMechanism):
    """Non-additive deep MLP: pa and s mixed in the first layer, then _N_DEPTH-1 more layers.

    The residual c_s * s ensures unbounded support and strict monotonicity in s.
    LayerNorm is applied before the final output layer.
    """
    W_in: Array          # (n_hidden, n_parents)
    w_s: Array           # (n_hidden,) — abs() enforced in forward pass
    b_in: Array          # (n_hidden,)
    hidden_weights: List[Array]  # len = _N_DEPTH-2; each (n_hidden, n_hidden)
    hidden_biases: List[Array]   # len = _N_DEPTH-2; each (n_hidden,)
    out_weight: Array    # (n_hidden,) — abs() enforced in forward pass
    ln_gamma: Array      # (n_hidden,) LayerNorm scale
    ln_beta: Array       # (n_hidden,) LayerNorm shift
    c_s: float           # strictly positive residual source coupling
    name: str = "neural_net"

    def __call__(self, parents: Array, source: Array) -> Array:
        pre = parents @ self.W_in.T + source[:, None] * np.abs(self.w_s) + self.b_in
        h = np.tanh(pre)
        for W, b in zip(self.hidden_weights, self.hidden_biases):
            h = np.tanh(h @ W.T + b)
        mu = h.mean(axis=-1, keepdims=True)
        std = h.std(axis=-1, keepdims=True) + 1e-5
        h = (h - mu) / std * self.ln_gamma + self.ln_beta
        return h @ np.abs(self.out_weight) + self.c_s * source

    def metadata(self) -> Dict[str, object]:
        return {"name": self.name}


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
    """GP-parameterized non-additive: x = GP([pa, s]) + c_s * s.

    s is concatenated with parents before the RFF so it enters nonlinearly.
    The c_s * s residual ensures strict monotonicity in s.
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
        return self._rff(z)

    def metadata(self) -> Dict[str, object]:
        return {"name": self.name}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_ADDITIVE_POOL = ["linear", "anm_tanh", "anm_nn", "pnl", "lsnm", "anm_gp"]
_NONADDITIVE_POOL = ["neural_net", "gp"]
_NOT_OBS_ID_POOL = ["linear", "gp"]
_OBS_ID_POOL = ["anm_tanh", "anm_nn", "pnl", "lsnm", "anm_gp"]


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

    if key in {"anm_nn", "neural_net_anm"}:
        hw = [rng.normal(0.0, 1.0 / np.sqrt(n_parents), size=(_N_HIDDEN, n_parents))]
        hb = [np.zeros(_N_HIDDEN)]
        for _ in range(_N_DEPTH - 2):
            hw.append(rng.normal(0.0, 1.0 / np.sqrt(_N_HIDDEN), size=(_N_HIDDEN, _N_HIDDEN)))
            hb.append(np.zeros(_N_HIDDEN))
        return NeuralNetANM(
            hidden_weights=hw,
            hidden_biases=hb,
            out_weight=rng.normal(0.0, 1.0 / np.sqrt(_N_HIDDEN), size=_N_HIDDEN),
            ln_gamma=np.ones(_N_HIDDEN),
            ln_beta=np.zeros(_N_HIDDEN),
        )

    if key in {"nn", "neural_net"}:
        hw = []
        hb = []
        for _ in range(_N_DEPTH - 2):
            hw.append(rng.normal(0.0, 1.0 / np.sqrt(_N_HIDDEN), size=(_N_HIDDEN, _N_HIDDEN)))
            hb.append(np.zeros(_N_HIDDEN))
        return NeuralNetMechanism(
            W_in=rng.normal(0.0, 1.0 / np.sqrt(n_parents), size=(_N_HIDDEN, n_parents)),
            w_s=rng.uniform(0.3, 0.7, size=_N_HIDDEN),
            b_in=np.zeros(_N_HIDDEN),
            hidden_weights=hw,
            hidden_biases=hb,
            out_weight=rng.uniform(0.3, 0.7, size=_N_HIDDEN) / _N_HIDDEN,
            ln_gamma=np.ones(_N_HIDDEN),
            ln_beta=np.zeros(_N_HIDDEN),
            c_s=float(rng.uniform(0.3, 0.7)),
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
            c_s=float(rng.uniform(0.3, 0.7)),
        )

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
