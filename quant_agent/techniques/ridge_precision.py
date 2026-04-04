"""
Technique 11 — Ridge-Regularized Precision Estimator

Stabilise the covariance inversion by adding λI before inverting.
Tiny eigenvalues (sampling noise) are clipped at λ from below, preventing
extreme inverse weights. Downstream: use Θ̂_λ anywhere Σ⁻¹ is needed
(min-variance portfolio, Mahalanobis distance, factor scoring).
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class RidgePrecisionResult:
    theta_lambda: np.ndarray       # regularised precision matrix (n, n)
    weights_mv: np.ndarray         # minimum-variance weights using Θ̂_λ
    eigenvalues_raw: np.ndarray    # eigenvalues of raw S
    eigenvalues_reg: np.ndarray    # effective eigenvalues of (S + λI)
    lambda_used: float
    condition_number_raw: float
    condition_number_reg: float
    shrinkage_pct: float           # how much λ moved the smallest eigenvalue (%)


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def _sample_cov(returns: np.ndarray) -> tuple:
    """Returns (S, eigvals, eigvecs) — eigendecomposition of sample covariance."""
    S = np.cov(returns.T, ddof=1)
    eigvals, eigvecs = np.linalg.eigh(S)   # ascending
    return S, eigvals, eigvecs


def estimate(returns: np.ndarray,
             lam: float | None = None,
             auto_lambda: bool = True) -> RidgePrecisionResult:
    """
    Parameters
    ----------
    returns      : (T, n) daily returns
    lam          : ridge parameter λ. If None and auto_lambda=True, use
                   cross-validated rule: λ = median eigenvalue / 10
    auto_lambda  : auto-select λ from spectrum

    Returns
    -------
    RidgePrecisionResult
    """
    T, n = returns.shape
    S, eigvals, eigvecs = _sample_cov(returns)

    if lam is None:
        # practical default: set λ = 10th percentile of eigenvalues
        # (clips the noisiest eigenvalues without touching the signal ones)
        lam = float(np.percentile(eigvals, 10)) if auto_lambda else 0.01

    lam = max(lam, 1e-8)

    # regularised precision: Θ̂_λ = (S + λI)⁻¹
    #   eigendecomposition trick: U diag(1/(λ_i + λ)) U^T
    reg_eigvals = eigvals + lam
    theta_lambda = eigvecs @ np.diag(1.0 / reg_eigvals) @ eigvecs.T

    # minimum-variance weights: w* = Θ̂_λ 1 / (1^T Θ̂_λ 1)
    ones = np.ones(n)
    raw_w = theta_lambda @ ones
    weights_mv = raw_w / (ones @ raw_w)

    cond_raw = eigvals[-1] / max(eigvals[0], 1e-12)
    cond_reg  = reg_eigvals[-1] / reg_eigvals[0]
    shrinkage = (lam / max(eigvals[0], 1e-12)) * 100

    return RidgePrecisionResult(
        theta_lambda=theta_lambda,
        weights_mv=weights_mv,
        eigenvalues_raw=eigvals,
        eigenvalues_reg=reg_eigvals,
        lambda_used=lam,
        condition_number_raw=cond_raw,
        condition_number_reg=cond_reg,
        shrinkage_pct=shrinkage,
    )


def rolling_weights(returns: np.ndarray,
                    window: int = 60,
                    lam: float | None = None) -> np.ndarray:
    """Rolling minimum-variance weights using ridge precision. Returns (T, n)."""
    T, n = returns.shape
    weights = np.full((T, n), np.nan)
    for t in range(window, T):
        chunk = returns[t - window: t]
        res = estimate(chunk, lam=lam)
        weights[t] = res.weights_mv
    return weights


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(101)

    T, n = 120, 15       # short sample → noisy covariance
    # true factor model: 2 factors
    F = rng.normal(0, 1, (T, 2))
    B = rng.normal(0, 0.5, (n, 2))
    idio = rng.normal(0, 0.3, (T, n))
    returns = F @ B.T + idio

    lambdas = [0.0001, 0.01, 0.05, 0.1, 0.5]
    results = [estimate(returns, lam=l) for l in lambdas]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Technique 11 — Ridge-Regularized Precision Estimator", fontweight="bold")

    # condition number vs λ
    conds = [r.condition_number_reg for r in results]
    axes[0].semilogx(lambdas, conds, "o-", color="steelblue")
    axes[0].set_xlabel("λ"); axes[0].set_ylabel("Condition number")
    axes[0].set_title("Condition number of (S + λI)⁻¹\nfalls rapidly with λ")
    axes[0].axhline(results[0].condition_number_raw, color="red", linestyle="--",
                    label=f"Raw cond = {results[0].condition_number_raw:.0f}")
    axes[0].legend(fontsize=8)

    # min-var weights stability (std across assets) vs λ
    w_stds = [r.weights_mv.std() for r in results]
    axes[1].semilogx(lambdas, w_stds, "s-", color="crimson")
    axes[1].set_xlabel("λ"); axes[1].set_ylabel("Std of weights")
    axes[1].set_title("Weight concentration falls with λ\n(more diversified)")

    # weight profile for two λ extremes
    x = np.arange(n)
    axes[2].bar(x - 0.2, results[0].weights_mv, 0.35,
                label=f"λ={lambdas[0]} (unstable)", color="crimson", alpha=0.7)
    axes[2].bar(x + 0.2, results[3].weights_mv, 0.35,
                label=f"λ={lambdas[3]} (stable)", color="steelblue", alpha=0.7)
    axes[2].axhline(1/n, color="gray", linestyle="--", label="Equal weight")
    axes[2].set_xlabel("Asset"); axes[2].set_ylabel("Weight")
    axes[2].set_title("Min-variance weights: raw vs regularised")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("demos/output_11_ridge_precision.png", dpi=120)
    plt.close()
    print("[11] Ridge Precision — saved demos/output_11_ridge_precision.png")
    res_auto = estimate(returns)
    print(f"     Auto λ = {res_auto.lambda_used:.4f}")
    print(f"     Condition: raw={res_auto.condition_number_raw:.1f}  →  reg={res_auto.condition_number_reg:.1f}")


if __name__ == "__main__":
    demo()
