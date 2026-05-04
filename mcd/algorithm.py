from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .graphs import full_dag_from_order
from .hsic import hsic_test_gamma
from .stein import SteinMixin, SteinScoreModel, standardize_apply, standardize_fit

SinkRule = Literal["variance", "hessian", "hsic", "projective", "predictive", "pit"]
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

            ratio, X_eval_active, score_models, train_by_env = self._fit_scores_and_compute_ratio(X_env, active, split, rng)
            local_sink, scores = self._identify_sink(ratio, X_eval_active, train_by_env, split)
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
    ) -> Tuple[NDArray, NDArray, Dict[int, SteinScoreModel], Dict[int, NDArray]]:
        """Fit score models on all training envs and evaluate on a common X_eval.

        variance/projective/predictive: no held-out split; all n samples used for score fitting. X_eval
          is the concatenation of all training data.
        hessian/hsic/pit: each env is shuffled and split at n_train=round(score_train_frac*n).
          Held-out rows from all envs are concatenated into the eval pool, using
          all available held-out samples (no subsampling).
        """
        k, n, _ = X_env.shape
        active_arr = np.asarray(active, dtype=int)
        fit_envs = [split.base] + split.group1 + split.group2

        if self.config.sink_rule in {"variance", "projective", "predictive"}:
            train_by_env = {e: X_env[e][:, active_arr] for e in fit_envs}
            X_eval = np.concatenate([train_by_env[e] for e in fit_envs], axis=0)
        else:
            if not 0.0 < self.config.score_train_frac < 1.0:
                raise ValueError("score_train_frac must be strictly between 0 and 1 for hessian/hsic/pit.")
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
        train_by_env_used = train_by_env
        if self.config.standardize:
            train_by_env_used = {e: standardize_apply(train_by_env[e], mean, std) for e in fit_envs}
        return ratio, X_eval_used, score_models, train_by_env_used

    def _identify_sink(
        self,
        ratio: NDArray,
        X_eval: NDArray,
        train_by_env: Optional[Dict[int, NDArray]] = None,
        split: Optional[EnvSplit] = None,
    ) -> Tuple[int, NDArray]:
        if self.config.sink_rule == "variance":
            scores = np.var(ratio, axis=0)
            return int(np.argmin(scores)), scores
        if self.config.sink_rule == "projective":
            scores = np.var(self._projective_coordinate(ratio), axis=0)
            return int(np.argmin(scores)), scores
        if self.config.sink_rule == "predictive":
            scores = self._predictability_scores(self._projective_coordinate(ratio), X_eval)
            return int(np.argmin(scores)), scores
        if self.config.sink_rule == "hessian":
            scores = self._hessian_independence_scores(ratio, X_eval)
            return int(np.argmin(scores)), scores
        if self.config.sink_rule == "hsic":
            scores = self._hsic_independence_scores(ratio, X_eval)
            return int(np.argmin(scores)), scores
        if self.config.sink_rule == "pit":
            if train_by_env is None or split is None:
                raise ValueError("PIT sink identification requires per-environment training data and an environment split.")
            scores = self._pit_predictability_scores(X_eval, train_by_env, split)
            return int(np.argmin(scores)), scores
        raise ValueError(f"Unknown sink_rule: {self.config.sink_rule}")

    def _projective_coordinate(self, ratio: NDArray) -> NDArray:
        ratio = np.asarray(ratio, dtype=float)
        return ratio / np.sqrt(1.0 + ratio**2)

    def _predictability_scores(self, target: NDArray, X_eval: NDArray) -> NDArray:
        n, d_active = X_eval.shape
        scores = np.zeros(d_active, dtype=float)
        rng = np.random.default_rng(self.config.random_state)
        perm = rng.permutation(n)
        folds = np.array_split(perm, 2)
        for loc in range(d_active):
            x_minus = np.delete(X_eval, loc, axis=1)
            y = target[:, loc:loc+1]
            scores[loc] = self._crossfit_ridge_score(x_minus, y, folds)
        return scores

    def _pit_predictability_scores(
        self,
        X_eval: NDArray,
        train_by_env: Dict[int, NDArray],
        split: EnvSplit,
    ) -> NDArray:
        _, d_active = X_eval.shape
        scores = np.zeros(d_active, dtype=float)
        for loc in range(d_active):
            features_eval = np.delete(X_eval, loc, axis=1)
            y_eval = X_eval[:, loc]
            pit_base = self._conditional_pit(train_by_env[split.base], features_eval, y_eval, loc)[:, None]
            pit_g1 = self._aggregate_group_pit(train_by_env, split.group1, features_eval, y_eval, loc)[:, None]
            pit_g2 = self._aggregate_group_pit(train_by_env, split.group2, features_eval, y_eval, loc)[:, None]
            target = self._logit_clip(np.concatenate([pit_base, pit_g1, pit_g2], axis=1))
            scores[loc] = self._crossfit_ridge_score(features_eval, target)
        return scores

    def _aggregate_group_pit(
        self,
        train_by_env: Dict[int, NDArray],
        group: List[int],
        features_eval: NDArray,
        y_eval: NDArray,
        loc: int,
    ) -> NDArray:
        values = [self._conditional_pit(train_by_env[e], features_eval, y_eval, loc) for e in group]
        return np.mean(np.stack(values, axis=0), axis=0)

    def _conditional_pit(
        self,
        X_train: NDArray,
        features_eval: NDArray,
        y_eval: NDArray,
        loc: int,
    ) -> NDArray:
        y_train = X_train[:, loc]
        x_minus_train = np.delete(X_train, loc, axis=1)
        if x_minus_train.shape[1] == 0:
            return (y_train[None, :] <= y_eval[:, None]).mean(axis=1)
        widths = np.sqrt(np.maximum(np.sum((x_minus_train[:, None, :] - x_minus_train[None, :, :]) ** 2, axis=2), 0.0))
        vals = widths[np.triu_indices_from(widths, k=1)]
        vals = vals[np.isfinite(vals) & (vals > 1e-12)]
        width = float(np.median(vals)) if vals.size else 1.0
        d2 = np.maximum(np.sum((features_eval[:, None, :] - x_minus_train[None, :, :]) ** 2, axis=2), 0.0)
        K = np.exp(-d2 / (2.0 * width**2))
        weights = K / np.maximum(K.sum(axis=1, keepdims=True), 1e-12)
        indicators = (y_train[None, :] <= y_eval[:, None]).astype(float)
        return np.sum(weights * indicators, axis=1)

    def _crossfit_ridge_score(
        self,
        X: NDArray,
        Y: NDArray,
        folds: Optional[List[NDArray]] = None,
    ) -> float:
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y[:, None]
        n = X.shape[0]
        if folds is None:
            rng = np.random.default_rng(self.config.random_state)
            folds = np.array_split(rng.permutation(n), 2)
        sse = 0.0
        sst = 0.0
        p = X.shape[1]
        eye = np.eye(p)
        for fold in folds:
            train = np.setdiff1d(np.arange(n), fold, assume_unique=False)
            if train.size == 0 or fold.size == 0:
                continue
            Xtr = X[train]
            Xva = X[fold]
            Ytr = Y[train]
            Yva = Y[fold]
            x_mean = Xtr.mean(axis=0, keepdims=True)
            x_std = Xtr.std(axis=0, keepdims=True)
            x_std = np.where(x_std < 1e-8, 1.0, x_std)
            Xtrn = (Xtr - x_mean) / x_std
            Xvan = (Xva - x_mean) / x_std
            y_mean = Ytr.mean(axis=0, keepdims=True)
            Ytrc = Ytr - y_mean
            beta = np.linalg.solve(Xtrn.T @ Xtrn + self.config.krr_reg * eye, Xtrn.T @ Ytrc) if p > 0 else np.zeros((0, Y.shape[1]))
            pred = (Xvan @ beta if p > 0 else np.zeros_like(Yva)) + y_mean
            sse += float(np.sum((Yva - pred) ** 2))
            centered = Yva - y_mean
            sst += float(np.sum(centered ** 2))
        return 1.0 - sse / max(sst, 1e-12)

    def _logit_clip(self, U: NDArray, eps: float = 1e-4) -> NDArray:
        U = np.clip(U, eps, 1.0 - eps)
        return np.log(U / (1.0 - U))

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

    def _hsic_independence_scores(self, ratio: NDArray, X_eval: NDArray) -> NDArray:
        _, d_active = X_eval.shape
        scores = np.zeros(d_active, dtype=float)
        for loc in range(d_active):
            x_minus = np.delete(X_eval, loc, axis=1)
            r_i = ratio[:, loc]
            test_stat, _ = hsic_test_gamma(r_i, x_minus)
            scores[loc] = test_stat
        return scores