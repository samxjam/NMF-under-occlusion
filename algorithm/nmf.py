"""Non-negative matrix factorization algorithms, implemented with numpy only.

Every solver shares one signature:

    W, H, history = solver(V, W0, H0, max_iter=..., tol=..., **hyper)

V is a (d, n) non-negative data matrix whose columns are vectorised images,
W0 (d, k) and H0 (k, n) are non-negative initial factors, and `history`
records the solver's own objective value after every iteration. Each solver
stops when the average relative change of its objective over the last 10
iterations drops below `tol` (after at least 100 iterations).

For speed, the loops write (d, n) intermediates into preallocated buffers
via `out=` arguments: allocating fresh 17 MB temporaries every iteration
costs more than the arithmetic. This does not change any result.

Implemented algorithms
----------------------
* ``l2``   Standard NMF with the squared Frobenius loss (Lee & Seung, 2001).
* ``l21``  L2,1-norm NMF (Kong, Ding & Huang, 2011).
* ``rnmf`` L1-norm regularised robust NMF, V = WH + E with sparse E
           (Zhang et al., 2011), solved with semi-NMF updates (Ding et al., 2010).
* ``cim``  Correntropy-induced metric NMF via half-quadratic optimisation
           (Du, Li & Shen, 2012).
"""

import numpy as np

EPS = 1e-10


def _converged(history, tol, window=10, min_iter=100):
    """Average relative change of the monitored value over the last `window`
    iterations is below `tol`. The window and the minimum number of
    iterations stop a short early plateau from ending the run prematurely."""
    if len(history) < max(min_iter, window + 1):
        return False
    prev, curr = history[-1 - window], history[-1]
    return abs(prev - curr) / max(abs(prev), EPS) < tol * window


def _pos(X):
    return (np.abs(X) + X) / 2


def _neg(X):
    return (np.abs(X) - X) / 2


def nmf_l2(V, W0, H0, max_iter=3000, tol=1e-4):
    """Standard NMF: min ||V - WH||_F^2  s.t. W, H >= 0.

    Lee-Seung multiplicative updates; the objective is non-increasing.
    """
    W, H = W0.copy(), H0.copy()
    history = []
    R = np.empty_like(V)
    for _ in range(max_iter):
        H *= (W.T @ V) / (W.T @ W @ H + EPS)
        W *= (V @ H.T) / (W @ (H @ H.T) + EPS)
        np.matmul(W, H, out=R)
        history.append(np.linalg.norm(np.subtract(V, R, out=R)) ** 2)
        if _converged(history, tol):
            break
    return W, H, history


def nmf_l21(V, W0, H0, max_iter=3000, tol=1e-4):
    """L2,1-norm NMF: min sum_i ||v_i - W h_i||_2  s.t. W, H >= 0.

    Each column (image) contributes its un-squared residual norm, so an image
    with a large error is down-weighted rather than dominating the fit. The
    updates are the reweighted multiplicative rules of Kong et al. (2011)
    with column weights D_ii = 1 / ||v_i - W h_i||.
    """
    W, H = W0.copy(), H0.copy()
    history = []
    R = np.empty_like(V)
    for _ in range(max_iter):
        # The column weights cancel in the H update, leaving the plain rule.
        H *= (W.T @ V) / (W.T @ W @ H + EPS)
        np.matmul(W, H, out=R)
        col_err = np.linalg.norm(np.subtract(V, R, out=R), axis=0)
        history.append(col_err.sum())
        if _converged(history, tol):
            break
        d = 1.0 / (col_err + EPS)
        W *= (np.multiply(V, d, out=R) @ H.T) / (W @ ((H * d) @ H.T) + EPS)
    return W, H, history


def huber_lambda(V, W0, H0):
    """Choose lam for RNMF-L1 without ground truth, by Huber's classical rule.

    Minimising the RNMF objective over E gives the Huber loss with threshold
    lam / 2. Huber's rule sets the threshold to 1.345 sigma, where sigma is a
    robust (MAD) estimate of the residual scale; we estimate it from a short
    L2-NMF fit on the noisy data.
    """
    W, H, _ = nmf_l2(V, W0, H0, max_iter=200, tol=0)
    R = V - W @ H
    sigma = 1.4826 * np.median(np.abs(R - np.median(R)))
    return float(2 * 1.345 * sigma)


def nmf_rnmf(V, W0, H0, lam=None, max_iter=3000, tol=1e-4):
    """L1-regularised robust NMF:

        min ||V - WH - E||_F^2 + lam * ||E||_1   s.t. W, H >= 0.

    E absorbs sparse, large-magnitude corruption. Alternating minimisation:
    E has the closed-form soft-threshold solution; W and H use the semi-NMF
    multiplicative updates of Ding, Li & Jordan (2010), including their square
    root, which keep the objective non-increasing although the target
    X = V - E can contain negative entries. lam=None selects lam with
    `huber_lambda`.

    The outlier estimate can be recovered afterwards with `rnmf_outliers`.
    """
    if lam is None:
        lam = huber_lambda(V, W0, H0)
    W, H = W0.copy(), H0.copy()
    history = []
    t = lam / 2
    R, E, X = np.empty_like(V), np.empty_like(V), np.empty_like(V)
    for _ in range(max_iter):
        np.subtract(V, W @ H, out=R)
        # E = S_t(R) = R - clip(R, -t, t), the soft threshold in two passes.
        np.clip(R, -t, t, out=X)
        np.subtract(R, X, out=E)
        fit = np.linalg.norm(np.subtract(R, E, out=X)) ** 2
        history.append(fit + lam * np.abs(E, out=R).sum())
        if _converged(history, tol):
            break
        np.subtract(V, E, out=X)
        WtX = W.T @ X
        H *= np.sqrt(_pos(WtX) / (W.T @ W @ H + _neg(WtX) + EPS))
        XHt = X @ H.T
        W *= np.sqrt(_pos(XHt) / (W @ (H @ H.T) + _neg(XHt) + EPS))
    return W, H, history


def rnmf_outliers(V, W, H, lam):
    """Closed-form E for fixed W, H in the RNMF objective (soft threshold)."""
    R = V - W @ H
    return R - np.clip(R, -lam / 2, lam / 2)


def cim_weights(V, W, H):
    """Half-quadratic pixel weights Omega of CIM-NMF for fixed W, H."""
    R = V - W @ H
    return np.exp(-R ** 2 / (2 * (np.mean(R ** 2) + EPS)))


def nmf_cim(V, W0, H0, max_iter=3000, tol=1e-4, sigma_rule='mean'):
    """Correntropy-induced metric (CIM) NMF:

        max sum_ij exp(-(V - WH)_ij^2 / (2 sigma^2))   s.t. W, H >= 0.

    Half-quadratic optimisation turns this into a sequence of weighted
    least-squares NMF problems. Each pixel gets the weight
    Omega_ij = exp(-r_ij^2 / (2 sigma^2)), so pixels with huge residuals
    (occluded pixels) receive weight close to zero. sigma^2 is re-estimated
    every iteration as the mean squared residual (our implementation choice).

    Because sigma follows the residual, the objective is nearly scale-invariant
    and can stall while the fit still improves, so convergence is tested on
    the residual norm ||V - WH||_F instead; `history` still records the CIM
    objective. sigma_rule='mad' instead estimates sigma robustly as
    1.4826 * median|r_ij| (used only in the ablation of extra_studies.py).
    """
    W, H = W0.copy(), H0.copy()
    history, residual_norms = [], []
    WH, Om, OV = np.empty_like(V), np.empty_like(V), np.empty_like(V)
    for _ in range(max_iter):
        np.matmul(W, H, out=WH)
        np.subtract(V, WH, out=Om)
        if sigma_rule == 'mad':
            mad_sigma2 = (1.4826 * np.median(np.abs(Om))) ** 2 + EPS
        np.square(Om, out=Om)
        sigma2 = np.mean(Om) + EPS
        residual_norms.append(np.sqrt(sigma2 * Om.size))
        if sigma_rule == 'mad':
            sigma2 = mad_sigma2
        np.divide(Om, -2 * sigma2, out=Om)
        np.exp(Om, out=Om)
        history.append(Om.size - Om.sum())
        if _converged(residual_norms, tol):
            break
        np.multiply(Om, V, out=OV)
        H *= (W.T @ OV) / (W.T @ np.multiply(Om, WH, out=WH) + EPS)
        np.matmul(W, H, out=WH)
        W *= (OV @ H.T) / (np.multiply(Om, WH, out=WH) @ H.T + EPS)
    return W, H, history


ALGORITHMS = {
    'L2-NMF': nmf_l2,
    'L21-NMF': nmf_l21,
    'RNMF-L1': nmf_rnmf,
    'CIM-NMF': nmf_cim,
}


def random_init(d, n, k, scale, rng):
    """Uniform random non-negative factors whose product matches the data scale."""
    W0 = rng.uniform(0, 1, size=(d, k)).astype(np.float32)
    H0 = rng.uniform(0, 1, size=(k, n)).astype(np.float32)
    factor = np.sqrt(scale / (W0 @ H0).mean())
    return W0 * factor, H0 * factor
