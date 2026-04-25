from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


def pairwise_sq_dists(X: NDArray, Y: Optional[NDArray] = None) -> NDArray:
    X = np.asarray(X, dtype=float)
    Y = X if Y is None else np.asarray(Y, dtype=float)
    X2 = np.sum(X * X, axis=1, keepdims=True)
    Y2 = np.sum(Y * Y, axis=1, keepdims=True).T
    D = X2 + Y2 - 2.0 * X @ Y.T
    return np.maximum(D, 0.0)


def rbf_kernel_np(X: NDArray, Y: Optional[NDArray] = None, *, gamma: float = 1.0) -> NDArray:
    return np.exp(-gamma * pairwise_sq_dists(X, Y))


def kernel_width(X: NDArray) -> float:
    """Median-distance bandwidth with safeguards."""
    X = np.asarray(X, dtype=float)
    if X.shape[0] < 2:
        return 1.0
    D = np.sqrt(pairwise_sq_dists(X))
    vals = D[np.triu_indices_from(D, k=1)]
    vals = vals[np.isfinite(vals) & (vals > 1e-12)]
    if vals.size == 0:
        return 1.0
    return float(np.median(vals))


class SteinMixin:
    """Small self-contained version of the dodiscover Stein score/Hessian mixin.

    The methods follow the attached dodiscover-style interface:
        score(X, eta_G)
        hessian(X, eta_G, eta_H)
    and return estimates at the samples X. A lightweight out-of-sample wrapper is
    provided below for the ratio construction on held-out samples.
    """

    def hessian(self, X: NDArray, eta_G: float, eta_H: float) -> NDArray:
        _, d = X.shape
        s = kernel_width(X)
        K = self._evaluate_kernel(X, s=s)
        nablaK = self._evaluate_nablaK(K, X, s)
        G = self.score(X, eta_G, K, nablaK)
        H = np.stack([self._hessian_col(X, G, col, eta_H, K, s) for col in range(d)], axis=1)
        return H

    def score(
        self,
        X: NDArray,
        eta_G: float,
        K: Optional[NDArray] = None,
        nablaK: Optional[NDArray] = None,
    ) -> NDArray:
        n, _ = X.shape
        if K is None:
            s = kernel_width(X)
            K = self._evaluate_kernel(X, s)
            nablaK = self._evaluate_nablaK(K, X, s)
        assert nablaK is not None
        return np.linalg.solve(K + eta_G * np.eye(n), nablaK)

    def _hessian_col(self, X: NDArray, G: NDArray, c: int, eta: float, K: NDArray, s: float) -> NDArray:
        X_diff = self._X_diff(X)
        Gv = np.einsum("i,ij->ij", G[:, c], G)
        nabla2vK = np.einsum("ik,ikj,ik->ij", X_diff[:, :, c], X_diff, K) / s**4
        nabla2vK[:, c] -= np.einsum("ik->i", K) / s**2
        H_col = -Gv + np.linalg.solve(K + eta * np.eye(X.shape[0]), nabla2vK)
        return H_col

    def hessian_diagonal(self, X: NDArray, eta_G: float, eta_H: float) -> NDArray:
        n, _ = X.shape
        s = kernel_width(X)
        K = self._evaluate_kernel(X, s)
        nablaK = self._evaluate_nablaK(K, X, s)
        G = self.score(X, eta_G, K, nablaK)
        X_diff = self._X_diff(X)
        nabla2K = np.einsum("kij,ik->kj", -1 / s**2 + X_diff**2 / s**4, K)
        return -(G**2) + np.linalg.solve(K + eta_H * np.eye(n), nabla2K)

    def _evaluate_kernel(self, X: NDArray, s: float) -> NDArray:
        return rbf_kernel_np(X, gamma=1 / (2 * s**2)) / s

    def _evaluate_nablaK(self, K: NDArray, X: NDArray, s: float) -> NDArray:
        return -np.einsum("kij,ik->kj", self._X_diff(X), K) / s**2

    def _X_diff(self, X: NDArray) -> NDArray:
        return np.expand_dims(X, axis=1) - X


@dataclass
class SteinScoreModel:
    """Kernel-ridge out-of-sample extension of the Stein score estimator.

    First estimate Stein scores at the training samples using SteinMixin.score.
    Then smooth/interpolate those scores to query points by kernel ridge regression.
    This keeps the ratio-evaluation samples disjoint from the score-fitting samples.
    """

    eta_G: float = 1e-2
    krr_reg: float = 1e-4

    def fit(self, X: NDArray) -> "SteinScoreModel":
        X = np.asarray(X, dtype=float)
        self.X_train_ = X
        self.width_ = kernel_width(X)
        self.mixin_ = SteinMixin()
        K = self.mixin_._evaluate_kernel(X, self.width_)
        nablaK = self.mixin_._evaluate_nablaK(K, X, self.width_)
        self.G_train_ = self.mixin_.score(X, self.eta_G, K=K, nablaK=nablaK)
        self.beta_ = np.linalg.solve(K + self.krr_reg * np.eye(X.shape[0]), self.G_train_)
        return self

    def score(self, X_query: NDArray) -> NDArray:
        X_query = np.asarray(X_query, dtype=float)
        Kq = rbf_kernel_np(X_query, self.X_train_, gamma=1 / (2 * self.width_**2)) / self.width_
        return Kq @ self.beta_


def standardize_fit(X: NDArray) -> Tuple[NDArray, NDArray]:
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def standardize_apply(X: NDArray, mean: NDArray, std: NDArray) -> NDArray:
    return (X - mean) / std
