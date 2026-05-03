"""
HSIC-based independence test.

Adapted from the LiNGAM project (https://sites.google.com/view/sshimizu06/lingam)
with reference to causal-learn (causallearn/search/FCMBased/lingam/hsic.py).
"""

import numpy as np
from scipy.stats import gamma
from statsmodels.nonparametric import bandwidths

__all__ = ["get_kernel_width", "get_gram_matrix", "hsic_teststat", "hsic_test_gamma"]


def get_kernel_width(X):
    n_samples = X.shape[0]
    if n_samples > 100:
        X_med = X[:100, :]
        n_samples = 100
    else:
        X_med = X

    G = np.sum(X_med * X_med, 1).reshape(n_samples, 1)
    dists = G + G.T - 2 * np.dot(X_med, X_med.T)
    dists = dists - np.tril(dists)
    dists = dists.reshape(n_samples**2, 1)

    return np.sqrt(0.5 * np.median(dists[dists > 0]))


def _rbf_dot(X, width):
    G = np.sum(X * X, axis=1)
    H = G[None, :] + G[:, None] - 2 * np.dot(X, X.T)
    return np.exp(-H / 2 / (width**2))


def get_gram_matrix(X, width):
    n = X.shape[0]
    K = _rbf_dot(X, width)
    K_colsums = K.sum(axis=0)
    K_rowsums = K.sum(axis=1)
    K_allsum = K_rowsums.sum()
    Kc = K - (K_colsums[None, :] + K_rowsums[:, None]) / n + (K_allsum / n**2)
    return K, Kc


def hsic_teststat(Kc, Lc, n):
    return 1 / n * np.sum(Kc.T * Lc)


def hsic_test_gamma(X, Y, bw_method="mdbs"):
    """HSIC independence test with gamma approximation for the null distribution.

    Parameters
    ----------
    X, Y : array-like, shape (n_samples, n_features)
    bw_method : str
        Bandwidth selection: ``mdbs`` (median distance), ``scott``, or ``silverman``.

    Returns
    -------
    test_stat : float
    p : float
    """
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y

    if bw_method == "scott":
        width_x = bandwidths.bw_scott(X)
        width_y = bandwidths.bw_scott(Y)
    elif bw_method == "silverman":
        width_x = bandwidths.bw_silverman(X)
        width_y = bandwidths.bw_silverman(Y)
    else:
        width_x = get_kernel_width(X)
        width_y = get_kernel_width(Y)

    K, Kc = get_gram_matrix(X, width_x)
    L, Lc = get_gram_matrix(Y, width_y)

    n = X.shape[0]
    test_stat = hsic_teststat(Kc, Lc, n)

    var = (1 / 6 * Kc * Lc) ** 2
    var = 1 / n / (n - 1) * (np.sum(var) - np.trace(var))
    var = 72 * (n - 4) * (n - 5) / n / (n - 1) / (n - 2) / (n - 3) * var

    K[np.diag_indices(n)] = 0
    L[np.diag_indices(n)] = 0
    mu_X = 1 / n / (n - 1) * K.sum()
    mu_Y = 1 / n / (n - 1) * L.sum()
    mean = 1 / n * (1 + mu_X * mu_Y - mu_X - mu_Y)

    alpha = mean**2 / var
    beta = var * n / mean
    p = gamma.sf(test_stat, alpha, scale=beta)

    return test_stat, p
