"""
Technique 2 — Marchenko-Pastur Denoising

Use Random Matrix Theory to distinguish genuine co-movement from sampling noise
in a correlation matrix. Eigenvalues inside the MP bulk are noise; eigenvalues
above λ+ carry signal. Project out noise components → cleaner covariance.
"""

import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize_scalar


@dataclass
class MPResult:
    raw_corr: np.ndarray          # sample correlation matrix
    clean_corr: np.ndarray        # denoised correlation matrix
    eigenvalues: np.ndarray       # all eigenvalues of raw corr
    lambda_plus: float            # MP upper edge
    lambda_minus: float           # MP lower edge
    sigma_sq: float               # fitted noise variance
    n_signal: int                 # eigenvalues above λ+
    explained_noise_pct: float    # % variance removed as noise


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def _mp_pdf(x: np.ndarray, q: float, sigma_sq: float) -> np.ndarray:
    """Marchenko-Pastur probability density."""
    lam_plus  = sigma_sq * (1 + np.sqrt(q)) ** 2
    lam_minus = sigma_sq * (1 - np.sqrt(q)) ** 2
    pdf = np.zeros_like(x, dtype=float)
    mask = (x >= lam_minus) & (x <= lam_plus)
    xm = x[mask]
    pdf[mask] = (np.sqrt((lam_plus - xm) * (xm - lam_minus))
                 / (2 * np.pi * sigma_sq * q * xm))
    return pdf


def _fit_sigma(eigenvalues: np.ndarray, q: float) -> float:
    """Fit σ² by minimising KS distance between empirical and MP CDF."""
    def ks(log_s):
        s = np.exp(log_s)
        lp = s * (1 + np.sqrt(q)) ** 2
        lm = s * (1 - np.sqrt(q)) ** 2
        bulk = eigenvalues[(eigenvalues >= lm) & (eigenvalues <= lp)]
        if len(bulk) < 3:
            return 1.0
        # compare empirical variance inside bulk to σ²
        return abs(np.mean(bulk) - s)

    res = minimize_scalar(ks, bounds=(-3, 1), method="bounded")
    return np.exp(res.x)


def denoise(returns: np.ndarray) -> MPResult:
    """
    Parameters
    ----------
    returns : (T, n) array — T observations, n assets

    Returns
    -------
    MPResult
    """
    T, n = returns.shape
    q = n / T                                       # ratio

    # standardise each asset
    std = returns.std(axis=0, ddof=1)
    std[std < 1e-8] = 1.0
    R = (returns - returns.mean(axis=0)) / std

    corr = (R.T @ R) / T                            # sample correlation
    eigvals, eigvecs = np.linalg.eigh(corr)         # ascending order

    sigma_sq = _fit_sigma(eigvals, q)
    lam_plus  = sigma_sq * (1 + np.sqrt(q)) ** 2
    lam_minus = sigma_sq * (1 - np.sqrt(q)) ** 2

    # replace noise eigenvalues with their mean (bulk mean = σ²)
    signal_mask = eigvals > lam_plus
    n_signal = signal_mask.sum()

    cleaned_eigvals = eigvals.copy()
    noise_mean = eigvals[~signal_mask].mean() if (~signal_mask).any() else sigma_sq
    cleaned_eigvals[~signal_mask] = noise_mean

    clean_corr = eigvecs @ np.diag(cleaned_eigvals) @ eigvecs.T
    # re-normalise diagonal to 1
    d = np.sqrt(np.diag(clean_corr))
    clean_corr = clean_corr / np.outer(d, d)

    explained = (eigvals[~signal_mask].sum() / eigvals.sum()) * 100

    return MPResult(
        raw_corr=corr,
        clean_corr=clean_corr,
        eigenvalues=eigvals,
        lambda_plus=lam_plus,
        lambda_minus=lam_minus,
        sigma_sq=sigma_sq,
        n_signal=n_signal,
        explained_noise_pct=explained,
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)

    n, T = 30, 252          # 30 assets, 1 year daily
    # True factor model: 3 factors + idiosyncratic noise
    factors = rng.normal(0, 1, (T, 3))
    loadings = rng.normal(0, 0.4, (n, 3))
    idio = rng.normal(0, 0.8, (T, n))
    returns = factors @ loadings.T + idio   # (T, n)

    res = denoise(returns)

    q = n / T
    x_grid = np.linspace(0.01, res.eigenvalues.max() * 1.1, 300)
    mp = _mp_pdf(x_grid, q, res.sigma_sq)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Technique 2 — Marchenko-Pastur Denoising", fontweight="bold")

    # left: eigenvalue histogram vs MP PDF
    ax = axes[0]
    ax.hist(res.eigenvalues, bins=20, density=True, alpha=0.6,
            color="steelblue", label="Empirical eigenvalues")
    ax.plot(x_grid, mp, "r-", lw=2, label="MP PDF (noise)")
    ax.axvline(res.lambda_plus, color="red", linestyle="--",
               label=f"λ+ = {res.lambda_plus:.2f}")
    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("Density")
    ax.set_title(f"Eigenvalue spectrum  ({res.n_signal} signal / {n} total)")
    ax.legend(fontsize=8)

    # right: raw vs clean correlation (first 10 assets)
    k = 10
    vmax = 1.0
    diff = np.abs(res.raw_corr[:k, :k] - res.clean_corr[:k, :k])
    im = axes[1].imshow(diff, cmap="Reds", vmin=0, vmax=0.4)
    axes[1].set_title(f"|Raw − Clean| correlation  (first {k} assets)\n"
                      f"{res.explained_noise_pct:.1f}% variance removed as noise")
    axes[1].set_xlabel("Asset"); axes[1].set_ylabel("Asset")
    plt.colorbar(im, ax=axes[1])

    plt.tight_layout()
    plt.savefig("demos/output_02_marchenko_pastur.png", dpi=120)
    plt.close()
    print("[2] Marchenko-Pastur — saved demos/output_02_marchenko_pastur.png")
    print(f"    Signal eigenvalues: {res.n_signal}, noise removed: {res.explained_noise_pct:.1f}%")


if __name__ == "__main__":
    demo()
