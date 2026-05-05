from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .graphs import full_dag_from_order
from .hsic import hsic_test_gamma
from .stein import SteinMixin, SteinScoreModel, standardize_apply, standardize_fit

SinkRule = Literal[
    "variance",
    "hessian",
    "hsic",
]
Aggregate = Literal["sum", "mean"]


@dataclass(frozen=True)
class EnvSplit:
    """Environment assignment used by MCD.

    env 0 is always the base. All remaining environments are split alternately
    into group1 and group2. eval_env=-1 indicates that the evaluation set is
    constructed internally from held-out portions of every training environment.
    """

    base: int
    group1: List[int]
    group2: List[int]
    eval_env: int  # -1 = no dedicated eval env; eval is pooled from all training envs


@dataclass
class MCDConfig:
    """Configuration for recursive multienvironment causal discovery.

    Two-stage pipeline:
    1. Fit one Stein/KRR score model per training environment.
    2. Evaluate all fitted score models on a common X_eval.
    3. Build one of the sink statistics from score-difference contrasts and
       identify sinks recursively.

    For each node i and sample x, form U_i(x) = (Delta_i^(1)(x), Delta_i^(2)(x))
    normalized to the unit circle. The sink statistic is applied to this
    2-vector target; no ratio is formed.

    All implemented MCD sink rules use a held-out evaluation pool. The total
    held-out budget is controlled by score_train_frac and split equally across
    all fitting environments.
    """

    sink_rule: SinkRule = "hessian"
    eta_G: float = 1e-2
    eta_H: float = 1e-2
    krr_reg: float = 1e-4
    denominator_eps: float = 1e-3
    score_train_frac: float = 0.5
    aggregate_envs: Aggregate = "sum"
    standardize: bool = True
    random_state: int = 0
    verbose: bool = False


def make_env_split(k: int) -> EnvSplit:
    """Assign environments to base / group1 / group2."""
    if k < 3:
        raise ValueError("MCD needs at least three environments: base + two auxiliary groups.")
    if k == 3:
        return EnvSplit(base=0, group1=[1], group2=[2], eval_env=-1)
    # k > 3: env k-1 is dedicated eval; envs 1,...,k-2 are auxiliary
    pool = list(range(1, k - 1))
    group1 = pool[::2]
    group2 = pool[1::2]
    return EnvSplit(base=0, group1=group1, group2=group2, eval_env=k - 1)


class MCD:
    """Recursive sink-peeling algorithm from multienvironment score contrasts.

    The returned order is source-to-sink. The dense adjacency follows the
    repository convention A[child, parent] = 1. No graph-pruning step is run.
    """

    name = "mcd"

    def __init__(self, config: Optional[MCDConfig] = None):
        self.config = config or MCDConfig()

    def fit(self, X_env: NDArray) -> "MCD":
        X_env = np.asarray(X_env, dtype=float)
        if X_env.ndim != 3:
            raise ValueError("X_env must have shape (k, n, d).")
        k, _, d = X_env.shape

        rng = np.random.default_rng(self.config.random_state)
        split = make_env_split(k)
        self.env_split_ = split

        active = list(range(d))
        sinks: List[int] = []
        while active:
            if len(active) == 1:
                sink = active[0]
                sinks.append(sink)
                break

            delta1, delta2, X_eval_active, _, train_by_env = self._fit_scores_and_compute_deltas(
                X_env, active, split, rng
            )
            local_sink, scores = self._identify_sink(delta1, delta2, X_eval_active, train_by_env, split, rng)
            sink = active[local_sink]
            if self.config.verbose:
                print(f"active={active}, sink={sink}, scores={scores}")

            sinks.append(sink)
            active = [node for node in active if node != sink]

        self.sinks_ = sinks
        self.order_ = list(reversed(sinks))
        self.adjacency_ = full_dag_from_order(self.order_)
        return self

    def _fit_scores_and_compute_deltas(
        self,
        X_env: NDArray,
        active: List[int],
        split: EnvSplit,
        rng: np.random.Generator,
    ) -> Tuple[NDArray, NDArray, NDArray, Dict[int, SteinScoreModel], Dict[int, NDArray]]:
        """Fit score models and evaluate two score-difference contrasts.

        Returns delta1, delta2, X_eval, score_models, train_by_env. When
        split.eval_env != -1, all training-env data is used for fitting and the
        dedicated eval env supplies X_eval. Otherwise each training env holds
        out (1 - score_train_frac) of its samples and X_eval is pooled from them.
        """
        _, n, _ = X_env.shape
        active_arr = np.asarray(active, dtype=int)
        fit_envs = [split.base] + split.group1 + split.group2

        train_by_env: Dict[int, NDArray] = {}
        if split.eval_env != -1:
            for e in fit_envs:
                train_by_env[e] = X_env[e][:, active_arr]
            X_eval = X_env[split.eval_env][:, active_arr]
        else:
            if not 0.0 < self.config.score_train_frac < 1.0:
                raise ValueError("score_train_frac must be strictly between 0 and 1.")
            n_train = max(2, int(round(self.config.score_train_frac * n)))
            eval_parts: List[NDArray] = []
            for e in fit_envs:
                perm = rng.permutation(n)
                train_by_env[e] = X_env[e][perm[:n_train]][:, active_arr]
                eval_parts.append(X_env[e][perm[n_train:]][:, active_arr])
            X_eval = np.concatenate(eval_parts, axis=0)

        if self.config.standardize:
            pooled = np.concatenate(list(train_by_env.values()), axis=0)
            mean, std = standardize_fit(pooled)
            X_eval_used = standardize_apply(X_eval, mean, std)
        else:
            mean = np.zeros(len(active), dtype=float)
            std = np.ones(len(active), dtype=float)
            X_eval_used = X_eval

        score_models: Dict[int, SteinScoreModel] = {}
        score_by_env: Dict[int, NDArray] = {}
        for e in fit_envs:
            X_train = train_by_env[e]
            if self.config.standardize:
                X_train = standardize_apply(X_train, mean, std)
            model = SteinScoreModel(eta_G=self.config.eta_G, krr_reg=self.config.krr_reg).fit(X_train)
            score_models[e] = model
            score_by_env[e] = model.score(X_eval_used)

        def aggregate(group: List[int]) -> NDArray:
            values = np.stack([score_by_env[e] for e in group], axis=0)
            if self.config.aggregate_envs == "mean":
                return values.mean(axis=0)
            return values.sum(axis=0)

        delta1 = score_by_env[split.base] - aggregate(split.group1)
        delta2 = score_by_env[split.base] - aggregate(split.group2)
        delta1 = np.clip(delta1, -1e6, 1e6)
        delta2 = np.clip(delta2, -1e6, 1e6)

        train_by_env_used = train_by_env
        if self.config.standardize:
            train_by_env_used = {e: standardize_apply(train_by_env[e], mean, std) for e in fit_envs}
        return delta1, delta2, X_eval_used, score_models, train_by_env_used

    def _identify_sink(
        self,
        delta1: NDArray,
        delta2: NDArray,
        X_eval: NDArray,
        train_by_env: Optional[Dict[int, NDArray]] = None,
        split: Optional[EnvSplit] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[int, NDArray]:
        return self._identify_sink_from_direction(delta1, delta2, X_eval)


    def _identify_sink_from_direction(
        self,
        delta1: NDArray,
        delta2: NDArray,
        X_eval: NDArray,
    ) -> Tuple[int, NDArray]:
        U = self._compute_normalized_direction(delta1, delta2)
        if self.config.sink_rule == "variance":
            scores = np.array([float(np.trace(np.cov(u, rowvar=False))) for u in U])
        elif self.config.sink_rule == "hessian":
            scores = self._hessian_independence_scores_by_node(U, X_eval)
        elif self.config.sink_rule == "hsic":
            scores = self._hsic_independence_scores_by_node(U, X_eval)
        else:
            raise ValueError(f"Unexpected direction sink rule: {self.config.sink_rule}")
        return int(np.argmin(scores)), scores

    def _compute_normalized_direction(self, delta1: NDArray, delta2: NDArray) -> List[NDArray]:
        """Return U_i = (delta1_i, delta2_i) / ||(delta1_i, delta2_i)|| for each node i.

        Returns a list of (n_eval, 2) arrays, one per active node.
        """
        denom = np.sqrt(delta1**2 + delta2**2 + self.config.denominator_eps**2)
        u1 = delta1 / denom
        u2 = delta2 / denom
        d_active = delta1.shape[1]
        return [np.column_stack([u1[:, i], u2[:, i]]) for i in range(d_active)]

    def _hessian_independence_scores_by_node(self, targets: List[NDArray], X_eval: NDArray) -> NDArray:
        _, d_active = X_eval.shape
        mixin = SteinMixin()
        scores = np.zeros(d_active, dtype=float)
        for loc in range(d_active):
            target = targets[loc]
            x_minus = np.delete(X_eval, loc, axis=1)
            z = np.column_stack([target, x_minus])
            mean, std = standardize_fit(z)
            z = standardize_apply(z, mean, std)
            H = mixin.hessian(z, eta_G=self.config.eta_G, eta_H=self.config.eta_H)
            target_dim = target.shape[1]
            block1 = H[:, :target_dim, target_dim:]
            block2 = H[:, target_dim:, :target_dim]
            scores[loc] = 0.5 * (np.mean(np.sum(block1**2, axis=(1, 2))) + np.mean(np.sum(block2**2, axis=(1, 2))))
        return scores

    def _hsic_independence_scores_by_node(self, targets: List[NDArray], X_eval: NDArray) -> NDArray:
        _, d_active = X_eval.shape
        scores = np.zeros(d_active, dtype=float)
        for loc in range(d_active):
            target = targets[loc]
            x_minus = np.delete(X_eval, loc, axis=1)
            test_stat, _ = hsic_test_gamma(target, x_minus)
            scores[loc] = test_stat
        return scores