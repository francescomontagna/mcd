from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .graphs import full_dag_from_order


@dataclass
class RandomOrderConfig:
    random_state: int = 0


class RandomOrderBaseline:
    """Random source-to-sink order baseline.

    This baseline ignores the data except for the number of nodes. It is useful
    as a sanity check for Dtop.
    """

    name = "random"

    def __init__(self, config: Optional[RandomOrderConfig] = None):
        self.config = config or RandomOrderConfig()

    def fit(self, X_env: NDArray) -> "RandomOrderBaseline":
        X_env = np.asarray(X_env)
        if X_env.ndim != 3:
            raise ValueError("X_env must have shape (k, n, d).")
        d = X_env.shape[-1]
        rng = np.random.default_rng(self.config.random_state)
        self.order_ = [int(x) for x in rng.permutation(d)]
        self.adjacency_ = full_dag_from_order(self.order_)
        return self
