from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .graphs import full_dag_from_order
from .stein import SteinMixin, SteinScoreModel, standardize_apply, standardize_fit

SinkRule = Literal["variance", "hessian"]
Aggregate = Literal["sum", "mean"]


@dataclass(frozen=True)
class EnvSplit:
    """Environment assignment used by MCD.

    env 0 is always the base. All remaining environments are split alternately
    into group1 and group2. eval_env=-1 indicates that the evaluation set is
    constructed internally (held-out portions of every training env for hessian;
    pooled training data for variance).
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
    2. Evaluate all fitted score models on a common X_eval, form score-difference
       ratios pointwise, and identify sinks.

    score_train_frac sets the total eval-pool size as a fraction of n when
    sink_rule="hessian": n_eval_total = score_train_frac * n, split equally
    across all k environments (n_eval_per_env = n_eval_total / k). Each env
    then uses n - n_eval_per_env samples for score fitting.
    For sink_rule="variance" no split is performed: all samples are used for
    score fitting and the full pooled data serves as X_eval.
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
    """Assign environments to base / group1 / group2.

    All k environments participate in score fitting. Environments 1..k-1 are
    split alternately into group1 and group2. eval_env=-1 because evaluation
    data is built internally from held-out samples of every training environment.
    """
    if k < 3:
        raise ValueError("MCD needs at least three environments: base + two auxiliary groups.")
    pool = list(range(1, k))
    group1 = pool[::2]
    group2 = pool[1::2]
    return EnvSplit(base=0, group1=group1, group2=group2, eval_env=-1)


def _safe_ratio(num: NDArray, den: NDArray, eps: float) -> NDArray:
    sign = np.where(den >= 0, 1.0, -1.0)
    den_safe = np.where(np.abs(den) < eps, sign * eps, den)
    return num / den_safe


class MCD:
    """Recursive sink-peeling algorithm from score-difference ratios.

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
        k, n, d = X_env.shape

        rng = np.random.default_rng(self.config.random_state)
        split = make_env_split(k)
        self.env_split_ = split
        self.sink_trace_: List[Dict[str, object]] = []
        self.score_model_trace_: List[Dict[int, SteinScoreModel]] = []

        active = list(range(d))
        sinks: List[int] = []
        while active:
            if len(active) == 1:
                sink = active[0]
                sinks.append(sink)
                self.sink_trace_.append({"active": active.copy(), "sink": sink, "scores": {sink: 0.0}})
                break

            ratio, X_eval_active, score_models = self._fit_scores_and_compute_ratio(X_env, active, split, rng)
            local_sink, scores = self._identify_sink(ratio, X_eval_active)
            sink = active[local_sink]
            if self.config.verbose:
                print(f"active={active}, sink={sink}, scores={scores}")

            self.score_model_trace_.append(score_models)
            self.sink_trace_.append(
                {"active": active.copy(), "sink": sink, "scores": {active[i]: float(scores[i]) for i in range(len(active))}}
            )
            sinks.append(sink)
            active = [node for node in active if node != sink]

        self.sinks_ = sinks
        self.order_ = list(reversed(sinks))
        self.adjacency_ = full_dag_from_order(self.order_)
        return self

    def _fit_scores_and_compute_ratio(
        self,
        X_env: NDArray,
        active: List[int],
        split: EnvSplit,
        rng: np.random.Generator,
    ) -> Tuple[NDArray, NDArray, Dict[int, SteinScoreModel]]:
        """Fit score models on all training envs and evaluate on a common X_eval.

        variance: no held-out split; all n samples used for score fitting. X_eval
          is the concatenation of all training data (in-sample evaluation is
          harmless for the variance criterion).
        hessian: each env is shuffled and split at n_train=round(score_train_frac*n).
          Held-out rows from all envs are concatenated into the eval pool, using
          all available held-out samples (no subsampling).
        """
        k, n, _ = X_env.shape
        active_arr = np.asarray(active, dtype=int)
        fit_envs = [split.base] + split.group1 + split.group2

        if self.config.sink_rule == "variance":
            train_by_env = {e: X_env[e][:, active_arr] for e in fit_envs}
            X_eval = np.concatenate([train_by_env[e] for e in fit_envs], axis=0)
        else:
            if not 0.0 < self.config.score_train_frac < 1.0:
                raise ValueError("score_train_frac must be strictly between 0 and 1 for hessian.")
            # score_train_frac sets the TOTAL eval size as a fraction of n;
            # that budget is distributed equally across all k training environments.
            n_eval_per_env = max(1, int(round(self.config.score_train_frac * n / len(fit_envs))))
            n_train = max(2, n - n_eval_per_env)
            train_by_env: Dict[int, NDArray] = {}
            eval_parts: List[NDArray] = []
            for e in fit_envs:
                perm = rng.permutation(n)
                train_by_env[e] = X_env[e][perm[:n_train]][:, active_arr]
                eval_parts.append(X_env[e][perm[n_train:]][:, active_arr])
            X_eval = np.concatenate(eval_parts, axis=0)

        if self.config.standardize:
            pooled = np.concatenate(list(train_by_env.values()) + [X_eval], axis=0)
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
        ratio = _safe_ratio(delta1, delta2, self.config.denominator_eps)
        ratio = np.clip(ratio, -1e6, 1e6)
        return ratio, X_eval_used, score_models

    def _identify_sink(self, ratio: NDArray, X_eval: NDArray) -> Tuple[int, NDArray]:
        if self.config.sink_rule == "variance":
            scores = np.var(ratio, axis=0)
            return int(np.argmin(scores)), scores
        if self.config.sink_rule == "hessian":
            scores = self._hessian_independence_scores(ratio, X_eval)
            return int(np.argmin(scores)), scores
        raise ValueError(f"Unknown sink_rule: {self.config.sink_rule}")

    def _hessian_independence_scores(self, ratio: NDArray, X_eval: NDArray) -> NDArray:
        _, d_active = X_eval.shape
        mixin = SteinMixin()
        scores = np.zeros(d_active, dtype=float)
        for loc in range(d_active):
            x_minus = np.delete(X_eval, loc, axis=1)
            z = np.column_stack([ratio[:, loc], x_minus])
            mean, std = standardize_fit(z)
            z = standardize_apply(z, mean, std)
            H = mixin.hessian(z, eta_G=self.config.eta_G, eta_H=self.config.eta_H)
            block1 = H[:, 0, 1:]
            block2 = H[:, 1:, 0]
            scores[loc] = 0.5 * (np.mean(np.sum(block1**2, axis=1)) + np.mean(np.sum(block2**2, axis=1)))
        return scores
