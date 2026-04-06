"""
Technique 7 — Grassmann Rotation (Risk Subspace Detector)

Track the principal risk subspace (top-k PCA eigenvectors) on the Grassmann
manifold. Use principal angles between consecutive subspaces as a sign- and
basis-invariant measure. A large rotation Θt signals factor crowding shifts,
correlation regime breaks, or hedge decay.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class GrassmannResult:
    times: np.ndarray
    rotation_score: np.ndarray     # Θt = ‖θ‖_2 (k principal angles)
    principal_angles: np.ndarray   # (T, k) all principal angles
    alarm: np.ndarray              # bool — rotation > threshold
    eigenvalues: np.ndarray        # (T, k) dominant eigenvalues over time
    tau: float


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def _principal_angles(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Principal angles between column spaces of U (n×k) and V (n×k).
    Uses SVD of U^T V.  Returns k angles in [0, π/2].
    """
    # orthonormalise both bases
    Qu, _ = np.linalg.qr(U)
    Qv, _ = np.linalg.qr(V)
    M = Qu.T @ Qv                           # (k, k)
    sigmas = np.linalg.svd(M, compute_uv=False)
    sigmas = np.clip(sigmas, -1.0, 1.0)
    return np.arccos(sigmas)                # angles in radians


def _pca_subspace(returns_window: np.ndarray, k: int) -> tuple:
    """Top-k eigenvectors and eigenvalues of the covariance matrix."""
    cov = np.cov(returns_window.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # descending order
    idx = np.argsort(eigvals)[::-1]
    return eigvecs[:, idx[:k]], eigvals[idx[:k]]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(returns: np.ndarray,
        k: int = 3,
        window: int = 60,
        tau: float = 0.3) -> GrassmannResult:
    """
    Parameters
    ----------
    returns : (T, n) daily returns
    k       : number of principal components to track
    window  : rolling estimation window
    tau     : alarm threshold (radians)
    """
    T, n = returns.shape
    scores, all_angles, all_eigvals = [], [], []

    prev_U = None

    for t in range(window, T):
        chunk = returns[t - window: t]
        U, lam = _pca_subspace(chunk, k)

        if prev_U is None:
            prev_U = U
            scores.append(0.0)
            all_angles.append(np.zeros(k))
            all_eigvals.append(lam)
            continue

        angles = _principal_angles(prev_U, U)
        Theta = np.linalg.norm(angles)

        scores.append(Theta)
        all_angles.append(angles)
        all_eigvals.append(lam)

        prev_U = U

    scores = np.array(scores)
    return GrassmannResult(
        times=np.arange(window, T),
        rotation_score=scores,
        principal_angles=np.array(all_angles),
        alarm=scores > tau,
        eigenvalues=np.array(all_eigvals),
        tau=tau,
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(17)

    T, n, k_factors = 600, 20, 3

    def make_factor_returns(T, n, k, seed_offset=0):
        F = rng.normal(0, 1, (T, k))
        L = rng.normal(0, 0.5, (n, k))
        idio = rng.normal(0, 0.3, (T, n))
        return F @ L.T + idio

    # Regime 1: factors A,B,C
    r1 = make_factor_returns(300, n, k_factors)
    # Factor rotation at t=300: factor structure shifts
    r2 = make_factor_returns(300, n, k_factors, seed_offset=99)
    returns = np.vstack([r1, r2])

    res = run(returns, k=k_factors, window=60, tau=0.25)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("Technique 7 — Grassmann Rotation (Risk Subspace)", fontweight="bold")

    t = res.times
    axes[0].plot(t, res.rotation_score, color="purple", lw=1.5, label="Θt rotation score")
    axes[0].axhline(res.tau, color="red", linestyle="--", label=f"τ = {res.tau:.2f}")
    axes[0].fill_between(t, 0, res.rotation_score, where=res.alarm,
                         alpha=0.3, color="red", label="Crowding / regime alarm")
    axes[0].axvline(300, color="orange", linestyle=":", lw=2, label="True factor shift")
    axes[0].set_ylabel("Θt (rad)"); axes[0].legend(fontsize=8)
    axes[0].set_title("Rotation score Θt = ‖principal angles‖  — spike = risk basis changed")

    for i in range(k_factors):
        axes[1].plot(t, res.principal_angles[:, i] * 180 / np.pi,
                     lw=1, label=f"PC{i+1} angle")
    axes[1].axvline(300, color="orange", linestyle=":", lw=2)
    axes[1].set_ylabel("Angle (°)")
    axes[1].set_title("Individual principal angles between consecutive subspaces")
    axes[1].legend(fontsize=8)

    for i in range(k_factors):
        axes[2].plot(t, res.eigenvalues[:, i], lw=1, label=f"λ{i+1}")
    axes[2].axvline(300, color="orange", linestyle=":", lw=2)
    axes[2].set_ylabel("Eigenvalue")
    axes[2].set_xlabel("Day")
    axes[2].set_title("Top-k eigenvalues over time  (explained variance per factor)")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("demos/output_07_grassmann_rotation.png", dpi=120)
    plt.close()
    print("[7] Grassmann Rotation — saved demos/output_07_grassmann_rotation.png")
    print(f"    Alarm days: {res.alarm.sum()}, peak Θt = {res.rotation_score.max():.4f} rad")


if __name__ == "__main__":
    demo()
