"""
Technique 22 — Implied Skew Dependence Map

Options-implied skew co-movement as a forward-looking dependence measure.
Skew shocks reveal which assets are jointly repriced in tail scenarios,
anchoring correlation estimates to what the options market is pricing.

    s_{i,t}(T) = σ_imp(25Δ put, T) − σ_imp(25Δ call, T)
    z_{i,t}    = (Δs_{i,t} − mean(Δs_i)) / (std(Δs_i) + ε)

Implied dependence map:   R_skew_{ij} = Corr(z_i, z_j)
Stabilized dependence:    R̂(α) = (1−α) R_realized + α R_skew,  α ∈ [0,1]
Tail-linking pressure:    p_t = (1/n) Σ_i z_{i,t}
Cross-asset clustering:   κ_t = ((Σ_i z_{i,t})² − Σ_i z_{i,t}²) / (n(n−1))
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class SkewDependenceResult:
    skew_corr:     np.ndarray         # (n, n) implied skew correlation R_skew
    blended_corr:  np.ndarray         # (n, n) stabilized R̂(α)
    realized_corr: np.ndarray         # (n, n) realized return correlation
    tail_pressure: np.ndarray         # (T,) p_t — system-wide skew stress
    clustering:    np.ndarray         # (T,) κ_t — cross-asset skew synchrony
    skew_shocks:   np.ndarray         # (T, n) standardized z_{i,t}
    asset_names:   list = field(default_factory=list)
    alpha:         float = 0.3


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def _psd_project(R: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Project to nearest positive-semidefinite matrix, reset diagonal to 1."""
    vals, vecs = np.linalg.eigh(R)
    vals = np.maximum(vals, eps)
    R_psd = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.diag(R_psd))
    R_psd = R_psd / np.outer(d, d)
    np.fill_diagonal(R_psd, 1.0)
    return R_psd


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(skew_series:  np.ndarray,
        returns:      np.ndarray | None = None,
        asset_names:  list | None = None,
        alpha:        float = 0.3,
        eps:          float = 1e-6) -> SkewDependenceResult:
    """
    Parameters
    ----------
    skew_series  : (T, n) array of implied skew levels (put_vol − call_vol per asset)
    returns      : (T, n) realized returns (optional; used for R_realized blend)
    asset_names  : list of n asset names
    alpha        : blend weight on R_skew (0 = pure realized, 1 = pure skew)
    eps          : numerical floor for std normalization
    """
    T, n = skew_series.shape
    if asset_names is None:
        asset_names = [f"Asset_{i}" for i in range(n)]

    # skew shocks: first difference, then z-score
    delta_s = np.diff(skew_series, axis=0, prepend=skew_series[[0]])
    mu_s  = delta_s.mean(axis=0)
    std_s = delta_s.std(axis=0, ddof=1) + eps
    z = (delta_s - mu_s) / std_s   # (T, n)

    # implied skew correlation
    R_skew = np.corrcoef(z.T)
    np.fill_diagonal(R_skew, 1.0)
    R_skew = _psd_project(R_skew)

    # realized correlation
    if returns is not None:
        R_real = np.corrcoef(returns.T)
        np.fill_diagonal(R_real, 1.0)
        R_real = _psd_project(R_real)
    else:
        R_real = np.eye(n)

    # stabilized blend R̂(α)
    R_blend = (1.0 - alpha) * R_real + alpha * R_skew
    R_blend = _psd_project(R_blend)

    # tail-linking pressure p_t = mean(z_{i,t})
    tail_pressure = z.mean(axis=1)

    # cross-asset clustering κ_t = ((Σ z_i)² − Σ z_i²) / (n(n−1))
    denom      = n * (n - 1) if n > 1 else 1.0
    row_sum    = z.sum(axis=1)
    row_ssq    = (z ** 2).sum(axis=1)
    clustering = (row_sum ** 2 - row_ssq) / denom

    return SkewDependenceResult(
        skew_corr=R_skew,
        blended_corr=R_blend,
        realized_corr=R_real,
        tail_pressure=tail_pressure,
        clustering=clustering,
        skew_shocks=z,
        asset_names=asset_names,
        alpha=alpha,
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(22)

    T, n = 500, 6
    names = [f"Asset {i+1}" for i in range(n)]

    # baseline: slow-moving, mostly uncorrelated skew
    skew = (rng.normal(0.05, 0.003, (T, n))
            + 0.002 * np.cumsum(rng.normal(0, 1, (T, n)), axis=0))

    # stress episode: coordinated skew steepening (all assets together)
    common_shock = rng.normal(-0.003, 0.002, (80, 1))
    skew[200:280] += common_shock + rng.normal(-0.001, 0.001, (80, n))

    returns = rng.normal(0, 0.010, (T, n))
    returns[200:280] += rng.normal(-0.002, 0.005, (80, n))

    res = run(skew, returns=returns, asset_names=names, alpha=0.3)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle("Technique 22 — Implied Skew Dependence Map", fontweight="bold")

    t = np.arange(T)
    axes[0].plot(t, res.tail_pressure, color="darkorange", lw=1.5, label="p_t (tail-linking pressure)")
    axes[0].axvspan(200, 280, alpha=0.15, color="red", label="Stress episode")
    axes[0].axhline(0, color="gray", lw=0.5)
    axes[0].set_ylabel("p_t"); axes[0].legend(fontsize=8)
    axes[0].set_title("Tail-linking pressure — broad-based skew steepening indicator")

    axes[1].plot(t, res.clustering, color="crimson", lw=1.5, label="κ_t (cross-asset clustering)")
    axes[1].axvspan(200, 280, alpha=0.15, color="red")
    axes[1].axhline(0, color="gray", lw=0.5)
    axes[1].set_ylabel("κ_t"); axes[1].legend(fontsize=8)
    axes[1].set_title("Cross-asset skew clustering — how collective the tail repricing is")

    im = axes[2].imshow(res.blended_corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    axes[2].set_xticks(range(n)); axes[2].set_xticklabels(names, fontsize=8)
    axes[2].set_yticks(range(n)); axes[2].set_yticklabels(names, fontsize=8)
    plt.colorbar(im, ax=axes[2])
    axes[2].set_title(f"Stabilized dependence R̂(α={res.alpha}) — realized + implied skew blend")

    plt.tight_layout()
    plt.savefig("demos/output_22_skew_dependence.png", dpi=120)
    plt.close()
    print("[22] Skew Dependence — saved demos/output_22_skew_dependence.png")
    print(f"     Peak tail-linking pressure: {abs(res.tail_pressure).max():.4f}")
    print(f"     R_skew vs R_realized max diff: {abs(res.skew_corr - res.realized_corr).max():.4f}")


if __name__ == "__main__":
    demo()
