"""
Technique 14 — Factor Orthogonalization / Whitening

Rotate the factor space into an orthogonal basis so factors are decorrelated.
Then whiten (unit-variance) for a clean, independent basis.
Risk attribution becomes diagonal: each factor's contribution is independent.
Exposures (B̃ = BU, B̃̃ = BUΛ^{-1/2}) are better conditioned.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class FactorRotationResult:
    # Rotated (orthogonal, but not unit-variance)
    G: np.ndarray               # (T, k) rotated factors g_t = U^T f_t
    B_tilde: np.ndarray         # (n, k) rotated exposures B̃ = BU
    sigma_g: np.ndarray         # (k, k) diagonal factor cov (= Λ)

    # Whitened (orthogonal + unit-variance)
    H: np.ndarray               # (T, k) whitened factors h_t = Λ^{-1/2} U^T f_t
    B_tilde2: np.ndarray        # (n, k) whitened exposures B̃ = BUΛ^{1/2}
    sigma_h: np.ndarray         # (k, k) = I_k

    # Attribution
    factor_var_contribution: np.ndarray   # (k,) each factor's variance contribution
    asset_attribution: np.ndarray         # (n, k) per-asset, per-factor variance

    U: np.ndarray               # rotation matrix
    Lambda: np.ndarray          # factor eigenvalues (diagonal)
    k: int
    factor_names_rotated: list


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def rotate(B: np.ndarray,
           factors: np.ndarray,
           asset_names: list | None = None) -> FactorRotationResult:
    """
    Parameters
    ----------
    B       : (n, k) original factor exposure matrix
    factors : (T, k) original factor returns

    Returns
    -------
    FactorRotationResult
    """
    T, k = factors.shape
    n = B.shape[0]

    # Factor covariance and its eigendecomposition
    Sigma_f = np.cov(factors.T, ddof=1)
    eigvals, U = np.linalg.eigh(Sigma_f)  # ascending
    # Descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    U = U[:, idx]
    Lambda = np.diag(eigvals)

    # Rotated factors: g_t = U^T f_t
    G = factors @ U   # (T, k)
    B_tilde = B @ U   # (n, k)

    # Whitened factors: h_t = Λ^{-1/2} g_t
    Lambda_inv_half = np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1e-10)))
    Lambda_half     = np.diag(np.sqrt(np.maximum(eigvals, 1e-10)))

    H = G @ Lambda_inv_half             # (T, k)
    B_tilde2 = B_tilde @ Lambda_half    # (n, k)  note: B̃ = BUΛ^{1/2} for whitened

    # Variance attribution in rotated basis: Var(w^T r) ≈ β_g^T Λ β_g + w^T D_ε w
    # Per-asset, per-factor: B̃_{ij}² * λ_j
    asset_attribution = B_tilde ** 2 * eigvals[np.newaxis, :]  # (n, k)
    factor_var_contribution = (B_tilde ** 2 * eigvals).sum(axis=0)  # (k,)
    factor_var_contribution /= factor_var_contribution.sum()

    return FactorRotationResult(
        G=G,
        B_tilde=B_tilde,
        sigma_g=Lambda,
        H=H,
        B_tilde2=B_tilde2,
        sigma_h=np.eye(k),
        factor_var_contribution=factor_var_contribution,
        asset_attribution=asset_attribution,
        U=U,
        Lambda=Lambda,
        k=k,
        factor_names_rotated=[f"RF{i+1}" for i in range(k)],
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(404)

    T, n, k = 300, 12, 4
    names = [f"A{i}" for i in range(n)]

    # Correlated factor returns (pre-rotation)
    cov_f = np.array([
        [1.0, 0.7, 0.3, 0.1],
        [0.7, 1.0, 0.4, 0.2],
        [0.3, 0.4, 1.0, 0.5],
        [0.1, 0.2, 0.5, 1.0],
    ])
    L = np.linalg.cholesky(cov_f)
    factors = rng.normal(0, 1, (T, k)) @ L.T   # correlated
    B = rng.normal(0, 0.5, (n, k))
    B[:, 0] = np.abs(B[:, 0])                  # market beta positive

    res = rotate(B, factors)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Technique 14 — Factor Orthogonalization / Whitening", fontweight="bold")

    # original factor correlation
    orig_corr = np.corrcoef(factors.T)
    im0 = axes[0, 0].imshow(orig_corr, vmin=-1, vmax=1, cmap="RdBu_r")
    axes[0, 0].set_title("Original factor correlation\n(off-diagonals non-zero)")
    plt.colorbar(im0, ax=axes[0, 0])

    # rotated factor correlation (should be ~identity)
    rot_corr = np.corrcoef(res.G.T)
    im1 = axes[0, 1].imshow(rot_corr, vmin=-1, vmax=1, cmap="RdBu_r")
    axes[0, 1].set_title("Rotated factor correlation\n(≈ diagonal = independent factors)")
    plt.colorbar(im1, ax=axes[0, 1])

    # variance contribution per rotated factor
    axes[1, 0].bar(range(k), res.factor_var_contribution * 100,
                   color="steelblue", alpha=0.8)
    axes[1, 0].set_xticks(range(k))
    axes[1, 0].set_xticklabels([f"RF{i+1}" for i in range(k)])
    axes[1, 0].set_ylabel("% of factor variance")
    axes[1, 0].set_title("Variance explained per rotated factor\n(scree plot in rotated basis)")

    # per-asset attribution heatmap
    im3 = axes[1, 1].imshow(res.asset_attribution, aspect="auto", cmap="YlOrRd")
    axes[1, 1].set_xticks(range(k)); axes[1, 1].set_xticklabels([f"RF{i+1}" for i in range(k)])
    axes[1, 1].set_yticks(range(n)); axes[1, 1].set_yticklabels(names, fontsize=7)
    axes[1, 1].set_title("Per-asset, per-factor variance attribution\n(diagonal = each factor independent)")
    plt.colorbar(im3, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig("demos/output_14_factor_rotation.png", dpi=120)
    plt.close()
    print("[14] Factor Rotation — saved demos/output_14_factor_rotation.png")
    print(f"     Original factor corr off-diag max: {np.abs(orig_corr - np.eye(k)).max():.3f}")
    print(f"     Rotated factor corr off-diag max:  {np.abs(rot_corr - np.eye(k)).max():.3f}")
    print(f"     Factor variance shares: {(res.factor_var_contribution*100).round(1)}")


if __name__ == "__main__":
    demo()
