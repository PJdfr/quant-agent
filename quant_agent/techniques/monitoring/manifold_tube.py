"""
Technique 1 — Manifold Tube (Signal Stability Monitor)

Treat rolling-calibrated model parameters as a path on the statistical manifold.
Build a tubular neighbourhood using Fisher-Rao distance. A tube exit (δt > ε)
signals structural departure → trade allowed, size ∝ excess distance.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class TubeResult:
    times: np.ndarray          # time index
    delta: np.ndarray          # distance-to-path score δt
    in_tube: np.ndarray        # bool mask
    signal: np.ndarray         # 0 = hold, +1/-1 = directional signal
    size: np.ndarray           # position size ∈ [0, 1]
    epsilon: float             # tube radius used
    theta: np.ndarray          # rolling parameters (μ, σ) at each t


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def _fit_gaussian(returns: np.ndarray) -> Tuple[float, float]:
    """MLE fit of Gaussian: returns (μ, σ)."""
    mu = returns.mean()
    sigma = returns.std(ddof=1) + 1e-8
    return mu, sigma


def _fisher_rao_gaussian(mu1: float, sigma1: float,
                          mu2: float, sigma2: float) -> float:
    """
    Fisher-Rao distance between N(μ1,σ1²) and N(μ2,σ2²).
    Exact closed form on the Gaussian sub-manifold.
    """
    s1, s2 = max(sigma1, 1e-8), max(sigma2, 1e-8)
    # Parameterise as (μ, σ) in the upper half-plane with metric ds²
    # Closed-form: sqrt(2) * arcosh(1 + [(μ1-μ2)²+2(σ1-σ2)²] / (2σ1σ2))
    arg = 1.0 + ((mu1 - mu2) ** 2 + 2 * (s1 - s2) ** 2) / (2 * s1 * s2)
    arg = max(arg, 1.0)
    return np.sqrt(2) * np.arccosh(arg)


def _distance_to_path(theta_t: Tuple[float, float],
                       path: List[Tuple[float, float]]) -> float:
    """inf_u d(θ_t, γ(u)) — brute-force over the discrete path."""
    dists = [_fisher_rao_gaussian(theta_t[0], theta_t[1], p[0], p[1])
             for p in path]
    return min(dists)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(returns: np.ndarray,
        window: int = 60,
        epsilon: float = 0.15,
        size_max: float = 1.0) -> TubeResult:
    """
    Parameters
    ----------
    returns : 1-D array of daily returns
    window  : rolling calibration window (days)
    epsilon : tube radius (Fisher-Rao units)
    size_max: max position size

    Returns
    -------
    TubeResult with per-day scores, signals, sizes
    """
    T = len(returns)
    thetas, deltas, signals, sizes = [], [], [], []

    path: List[Tuple[float, float]] = []

    for t in range(window, T):
        window_ret = returns[t - window: t]
        mu, sigma = _fit_gaussian(window_ret)
        theta_t = (mu, sigma)
        thetas.append(theta_t)

        if len(path) == 0:
            path.append(theta_t)
            deltas.append(0.0)
            signals.append(0)
            sizes.append(0.0)
            continue

        delta_t = _distance_to_path(theta_t, path)
        deltas.append(delta_t)

        in_tube = delta_t <= epsilon
        if in_tube:
            path.append(theta_t)          # extend path inside tube
            signals.append(0)
            sizes.append(0.0)
        else:
            excess = (delta_t - epsilon) / epsilon
            sz = min(1.0, excess) * size_max
            # signal direction: mean above/below historical path mean
            path_means = [p[0] for p in path]
            sig = 1 if mu > np.mean(path_means) else -1
            signals.append(sig)
            sizes.append(sz)

    arr = np.array
    times = arr(range(window, T))
    return TubeResult(
        times=times,
        delta=arr(deltas),
        in_tube=arr(deltas) <= epsilon,
        signal=arr(signals),
        size=arr(sizes),
        epsilon=epsilon,
        theta=arr(thetas),
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(42)

    T = 500
    # Regime 1: μ=0.0003, σ=0.01  (first 300 days)
    r1 = rng.normal(0.0003, 0.010, 300)
    # Regime 2: μ=0.0010, σ=0.020  (sudden shift at day 300)
    r2 = rng.normal(0.0010, 0.020, 200)
    returns = np.concatenate([r1, r2])

    res = run(returns, window=60, epsilon=0.15)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Technique 1 — Manifold Tube", fontweight="bold")

    t = res.times

    axes[0].plot(t, res.delta, color="steelblue", label="δt (dist-to-path)")
    axes[0].axhline(res.epsilon, color="red", linestyle="--", label=f"ε = {res.epsilon}")
    axes[0].axvline(300, color="orange", linestyle=":", label="Regime shift")
    axes[0].fill_between(t, 0, res.epsilon, alpha=0.1, color="green")
    axes[0].set_ylabel("Fisher-Rao distance")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Distance-to-path score δt  (above red = tube exit → trade)")

    axes[1].scatter(t[res.signal != 0], res.size[res.signal != 0],
                    c=res.signal[res.signal != 0], cmap="RdYlGn",
                    vmin=-1, vmax=1, s=20, label="Trade signal")
    axes[1].set_ylabel("Position size")
    axes[1].set_title("Tube-adaptive position size  (coloured by direction)")
    axes[1].legend(fontsize=8)

    axes[2].plot(t, res.theta[:, 0] * 252, color="purple", label="μ annualised")
    axes[2].plot(t, res.theta[:, 1] * np.sqrt(252), color="darkorange", label="σ annualised")
    axes[2].axvline(300, color="orange", linestyle=":")
    axes[2].set_ylabel("Parameter value")
    axes[2].set_xlabel("Day")
    axes[2].set_title("Rolling parameter path θ(t) = (μ, σ)")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("demos/output_01_manifold_tube.png", dpi=120)
    plt.close()
    print("[1] Manifold Tube — saved demos/output_01_manifold_tube.png")
    print(f"    Tube exits detected: {(res.signal != 0).sum()} / {len(res.signal)} days")


if __name__ == "__main__":
    demo()
