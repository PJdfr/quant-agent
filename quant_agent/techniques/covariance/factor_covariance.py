"""
Technique 13 — Factor Model Covariance

Move the dependence problem to a low-dimensional factor space.
Estimate k×k factor covariance (stable), rebuild asset covariance through
exposures (betas): Σ̂_exp = B Σ_f B^T + D_ε.
Structure comes from the factor model rather than noisy pairwise correlations.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class FactorCovResult:
    B_hat: np.ndarray           # (n, k) exposure matrix (betas)
    sigma_f: np.ndarray         # (k, k) factor covariance
    D_eps: np.ndarray           # (n, n) diagonal idiosyncratic covariance
    sigma_exp: np.ndarray       # (n, n) full reconstructed asset covariance
    sigma_sample: np.ndarray    # (n, n) raw sample covariance (for comparison)
    r_squared: np.ndarray       # (n,) R² per asset
    residuals: np.ndarray       # (T, n) idiosyncratic residuals
    k: int
    asset_names: list
    factor_names: list


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def estimate(returns: np.ndarray,
             factors: np.ndarray,
             asset_names: list | None = None,
             factor_names: list | None = None) -> FactorCovResult:
    """
    Parameters
    ----------
    returns      : (T, n) asset returns
    factors      : (T, k) factor returns (e.g. Fama-French, PCA factors)
    asset_names  : list of n strings
    factor_names : list of k strings
    """
    T, n = returns.shape
    k = factors.shape[1]
    asset_names  = asset_names  or [f"A{i}" for i in range(n)]
    factor_names = factor_names or [f"F{j}" for j in range(k)]

    # OLS regression: β_i = argmin Σ_t (r_it - β^T f_t)²
    # Design matrix with intercept
    X = np.column_stack([np.ones(T), factors])   # (T, k+1)
    B_ext = np.linalg.lstsq(X, returns, rcond=None)[0]  # (k+1, n)
    alpha = B_ext[0]          # intercepts
    B_hat = B_ext[1:].T       # (n, k) betas

    # Residuals
    resid = returns - (X @ B_ext)   # (T, n)

    # Factor covariance: Σ_f = Cov(f_t)
    sigma_f = np.cov(factors.T, ddof=1)  # (k, k)

    # Idiosyncratic variances (diagonal only)
    idio_var = resid.var(axis=0, ddof=1)
    D_eps = np.diag(idio_var)

    # Reconstructed covariance
    sigma_exp = B_hat @ sigma_f @ B_hat.T + D_eps

    # R² per asset
    ss_res = (resid ** 2).sum(axis=0)
    ss_tot = ((returns - returns.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1 - ss_res / np.maximum(ss_tot, 1e-12)

    sigma_sample = np.cov(returns.T, ddof=1)

    return FactorCovResult(
        B_hat=B_hat,
        sigma_f=sigma_f,
        D_eps=D_eps,
        sigma_exp=sigma_exp,
        sigma_sample=sigma_sample,
        r_squared=r2,
        residuals=resid,
        k=k,
        asset_names=asset_names,
        factor_names=factor_names,
    )


def pca_factors(returns: np.ndarray, k: int) -> np.ndarray:
    """Extract k PCA factors from returns (useful when no observable factors available)."""
    centred = returns - returns.mean(axis=0)
    _, _, Vt = np.linalg.svd(centred, full_matrices=False)
    return centred @ Vt[:k].T   # (T, k) factor returns


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(303)

    T, n, k = 252, 20, 3
    names = [f"Asset{i+1}" for i in range(n)]
    factor_names = ["Market", "Value", "Momentum"]

    # True factor model
    true_F = rng.normal(0, 1, (T, k))
    true_B = rng.normal(0, 0.4, (n, k))
    true_B[:, 0] = np.abs(rng.normal(0.8, 0.2, n))   # market beta all positive
    idio = rng.normal(0, 0.2, (T, n))
    returns = true_F @ true_B.T + idio

    # Use PCA factors (since we don't "know" true factors)
    factors = pca_factors(returns, k)
    res = estimate(returns, factors, names, factor_names)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Technique 13 — Factor Model Covariance", fontweight="bold")

    # covariance comparison: sample vs factor-model
    vmax = max(np.abs(res.sigma_sample).max(), np.abs(res.sigma_exp).max())
    im0 = axes[0].imshow(res.sigma_sample, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0].set_title("Sample covariance Σ\n(noisy, all pairwise)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(res.sigma_exp, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[1].set_title(f"Factor-model covariance Σ̂_exp\n(k={k} factors, stable)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # R² per asset
    axes[2].barh(range(n), res.r_squared * 100, color="steelblue", alpha=0.8)
    axes[2].set_yticks(range(n)); axes[2].set_yticklabels(names, fontsize=7)
    axes[2].axvline(50, color="red", linestyle="--", label="50% threshold")
    axes[2].set_xlabel("R² (%)"); axes[2].set_title(f"Factor model fit R² per asset\n(mean={res.r_squared.mean()*100:.1f}%)")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("demos/output_13_factor_covariance.png", dpi=120)
    plt.close()
    print("[13] Factor Covariance — saved demos/output_13_factor_covariance.png")
    print(f"     Mean R² = {res.r_squared.mean()*100:.1f}%")
    print(f"     Factor cov condition: {np.linalg.cond(res.sigma_f):.1f}  "
          f"vs sample cov: {np.linalg.cond(res.sigma_sample):.1f}")


if __name__ == "__main__":
    demo()
