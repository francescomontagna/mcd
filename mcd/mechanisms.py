from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

Array = np.ndarray


class NodeMechanism:
    """Base class for mechanisms x_i = F_i(x_pa(i), s_i).

    Implementations must be invertible in the scalar source s_i for every fixed
    parent value. The sampler only needs the forward map.
    """

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


@dataclass
class LinearAdditive(NodeMechanism):
    weights: Array
    name: str = "linear"

    def __call__(self, parents: Array, source: Array) -> Array:
        return parents @ self.weights + source

    def metadata(self) -> Dict[str, object]:
        return {"name": self.name, "weights": self.weights.tolist()}


@dataclass
class TanhAdditive(NodeMechanism):
    weights: Array
    name: str = "anm_tanh"

    def __call__(self, parents: Array, source: Array) -> Array:
        return np.tanh(parents @ self.weights) + source

    def metadata(self) -> Dict[str, object]:
        return {"name": self.name, "weights": self.weights.tolist()}


@dataclass
class SensorWarp(NodeMechanism):
    weights_linear: Array
    weights_nonlinear: Array
    c: float
    name: str = "sensor_warp"

    @staticmethod
    def _phi(u: Array, c: float) -> Array:
        # Strictly increasing when |c| < 1.
        return u + c * np.tanh(u)

    def __call__(self, parents: Array, source: Array) -> Array:
        latent = parents @ self.weights_linear + np.tanh(parents @ self.weights_nonlinear) + source
        return self._phi(latent, self.c)

    def metadata(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "weights_linear": self.weights_linear.tolist(),
            "weights_nonlinear": self.weights_nonlinear.tolist(),
            "c": self.c,
        }


@dataclass
class LocationScaleCubic(NodeMechanism):
    weights_g: Array
    weights_h: Array
    hmin: float = 0.7
    hmax: float = 1.3
    eta: float = 0.05
    name: str = "location_scale_cubic"

    def _gain(self, parents: Array) -> Array:
        t = parents @ self.weights_h
        sig = 1.0 / (1.0 + np.exp(-np.clip(t, -40, 40)))
        return self.hmin + (self.hmax - self.hmin) * sig

    def __call__(self, parents: Array, source: Array) -> Array:
        g = np.tanh(parents @ self.weights_g)
        h = self._gain(parents)
        psi = source + self.eta * source**3
        return g + h * psi

    def metadata(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "weights_g": self.weights_g.tolist(),
            "weights_h": self.weights_h.tolist(),
            "hmin": self.hmin,
            "hmax": self.hmax,
            "eta": self.eta,
        }


@dataclass
class ParentDependentWarp(NodeMechanism):
    weights_g: Array
    weights_lam: Array
    weights_c: Array
    lam_min: float = 1.0
    lam_max: float = 1.6
    c_max: float = 0.5
    name: str = "parent_dependent_warp"

    def __call__(self, parents: Array, source: Array) -> Array:
        g = np.tanh(parents @ self.weights_g)
        t_lam = parents @ self.weights_lam
        sig = 1.0 / (1.0 + np.exp(-np.clip(t_lam, -40, 40)))
        lam = self.lam_min + (self.lam_max - self.lam_min) * sig
        c = self.c_max * np.tanh(parents @ self.weights_c)
        # derivative wrt source is lam + c * sech(source)^2 >= lam_min - c_max > 0.
        return g + lam * source + c * np.tanh(source)

    def metadata(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "weights_g": self.weights_g.tolist(),
            "weights_lam": self.weights_lam.tolist(),
            "weights_c": self.weights_c.tolist(),
            "lam_min": self.lam_min,
            "lam_max": self.lam_max,
            "c_max": self.c_max,
        }


@dataclass
class PostNonlinear(NodeMechanism):
    weights: Array
    c: float
    name: str = "pnl"

    def __call__(self, parents: Array, source: Array) -> Array:
        latent = np.tanh(parents @ self.weights) + source
        return latent + self.c * np.tanh(latent)

    def metadata(self) -> Dict[str, object]:
        return {"name": self.name, "weights": self.weights.tolist(), "c": self.c}


@dataclass
class MixedEdgeMechanism(NodeMechanism):
    """A node mechanism whose parent-specific edge functions are sampled uniformly.

    The mechanism is
        x_i = sum_j phi_j(x_j) + h(x_pa) * (s_i + eta s_i^3),
    where every phi_j is sampled independently from a small library. This gives
    the requested "mixed mechanism" at the edge level while retaining
    invertibility in s_i because h(x_pa) is positive.
    """

    edge_types: List[str]
    edge_weights: Array
    gain_weights: Array
    eta: float = 0.03
    hmin: float = 0.8
    hmax: float = 1.2
    name: str = "mixed"

    def _edge_eval(self, x: Array, typ: str, w: float) -> Array:
        if typ == "linear":
            return w * x
        if typ == "tanh":
            return w * np.tanh(x)
        if typ == "sin":
            return w * np.sin(x)
        if typ == "softsign":
            return w * x / (1.0 + np.abs(x))
        if typ == "cubic_mild":
            return w * (x + 0.05 * x**3)
        raise ValueError(f"Unknown edge type: {typ}")

    def _gain(self, parents: Array) -> Array:
        t = parents @ self.gain_weights
        sig = 1.0 / (1.0 + np.exp(-np.clip(t, -40, 40)))
        return self.hmin + (self.hmax - self.hmin) * sig

    def __call__(self, parents: Array, source: Array) -> Array:
        edge_sum = np.zeros(parents.shape[0])
        for j, typ in enumerate(self.edge_types):
            edge_sum += self._edge_eval(parents[:, j], typ, float(self.edge_weights[j]))
        psi = source + self.eta * source**3
        return edge_sum + self._gain(parents) * psi

    def metadata(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "edge_types": list(self.edge_types),
            "edge_weights": self.edge_weights.tolist(),
            "gain_weights": self.gain_weights.tolist(),
            "eta": self.eta,
        }


def build_mechanism(name: Union[str, NodeMechanism], n_parents: int, rng: np.random.Generator) -> NodeMechanism:
    """Build a predefined mechanism or pass through a custom NodeMechanism."""
    if isinstance(name, NodeMechanism):
        return name
    if n_parents == 0:
        return RootMechanism()

    key = str(name).lower()
    if key in {"linear", "linear_additive"}:
        return LinearAdditive(weights=rng.uniform(-1.5, 1.5, size=n_parents))
    if key in {"anm", "anm_tanh", "tanh"}:
        return TanhAdditive(weights=rng.uniform(-1.5, 1.5, size=n_parents))
    if key == "sensor_warp":
        return SensorWarp(
            weights_linear=rng.uniform(-1.0, 1.0, size=n_parents),
            weights_nonlinear=rng.uniform(-1.0, 1.0, size=n_parents),
            c=float(rng.uniform(-0.7, 0.7)),
        )
    if key in {"nan_warp", "location_scale", "location_scale_cubic"}:
        return LocationScaleCubic(
            weights_g=rng.uniform(-1.0, 1.0, size=n_parents),
            weights_h=rng.uniform(-1.0, 1.0, size=n_parents),
            eta=float(rng.uniform(0.02, 0.08)),
        )
    if key == "parent_dependent_warp":
        return ParentDependentWarp(
            weights_g=rng.uniform(-1.0, 1.0, size=n_parents),
            weights_lam=rng.uniform(-1.0, 1.0, size=n_parents),
            weights_c=rng.uniform(-1.0, 1.0, size=n_parents),
        )
    if key == "pnl":
        return PostNonlinear(weights=rng.uniform(-1.5, 1.5, size=n_parents), c=float(rng.uniform(-0.7, 0.7)))
    if key in {"mixed", "mixed_edges", "mixed_mechanism"}:
        edge_library = ["linear", "tanh", "sin", "softsign", "cubic_mild"]
        edge_types = [str(rng.choice(edge_library)) for _ in range(n_parents)]
        return MixedEdgeMechanism(
            edge_types=edge_types,
            edge_weights=rng.uniform(-1.2, 1.2, size=n_parents),
            gain_weights=rng.uniform(-1.0, 1.0, size=n_parents),
            eta=float(rng.uniform(0.01, 0.05)),
        )
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
