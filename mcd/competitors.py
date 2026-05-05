from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .graphs import full_dag_from_order


def _base_env(X_env: NDArray) -> NDArray:
    X_env = np.asarray(X_env, dtype=float)
    if X_env.ndim != 3:
        raise ValueError("X_env must have shape (k, n, d).")
    return X_env[0]


class NoGAMWrapper:
    name = "nogam"

    def __init__(self, random_state: int = 0):
        self.random_state = random_state

    def fit(self, X_env: NDArray) -> "NoGAMWrapper":
        from dodiscover import make_context
        from dodiscover.toporder import NoGAM

        X = _base_env(X_env)
        df = pd.DataFrame(X)
        ctx = make_context().variables(data=df).build()
        model = NoGAM()
        model.learn_graph(df, ctx)
        self.order_ = [int(v) for v in model.order_]
        self.adjacency_ = full_dag_from_order(self.order_)
        return self


class CAMWrapper:
    name = "cam"

    def __init__(self, random_state: int = 0):
        self.random_state = random_state

    def fit(self, X_env: NDArray) -> "CAMWrapper":
        from dodiscover import make_context
        from dodiscover.toporder import CAM

        X = _base_env(X_env)
        df = pd.DataFrame(X)
        ctx = make_context().variables(data=df).build()
        model = CAM()
        model.learn_graph(df, ctx)
        self.order_ = [int(v) for v in model.order_]
        self.adjacency_ = full_dag_from_order(self.order_)
        return self


class LiNGAMWrapper:
    name = "lingam"

    def __init__(self, random_state: int = 0):
        self.random_state = random_state

    def fit(self, X_env: NDArray) -> "LiNGAMWrapper":
        from lingam import DirectLiNGAM

        X = _base_env(X_env)
        model = DirectLiNGAM(random_state=self.random_state)
        model.fit(X)
        self.order_ = [int(v) for v in model.causal_order_]
        self.adjacency_ = full_dag_from_order(self.order_)
        return self


class RESITWrapper:
    name = "resit"

    def __init__(self, random_state: int = 0):
        self.random_state = random_state

    def fit(self, X_env: NDArray) -> "RESITWrapper":
        from lingam import RESIT
        from sklearn.gaussian_process import GaussianProcessRegressor

        X = _base_env(X_env)
        regressor = GaussianProcessRegressor(random_state=self.random_state)
        model = RESIT(regressor=regressor, random_state=self.random_state)
        model.fit(X)
        self.order_ = [int(v) for v in model.causal_order_]
        self.adjacency_ = full_dag_from_order(self.order_)
        return self
