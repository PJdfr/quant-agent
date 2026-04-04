"""
Technique 4 — Esscher Tilt / Crash Premium

Move from the historical return law P along an exponential-tilt path to Q_θ.
Choose θ so that tilted moments match option-implied targets (mean, variance,
skew). The tilt θ* is an entropy-minimal change of measure and summarises the
crash premium the market currently prices.
"""

import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize


@dataclass
class EsscherResult:
    theta_star: float           # optimal tilt (scalar or vector)
    crash_premium: float        # |θ*|  — higher = more tail fear
    tilted_mean: float
    tilted_var: float
    tilted_skew: float
    historical_mean: float
    historical_var: float
    historical_skew: float
    kl_distance: float          # D_KL(Q‖P) — entropy cost of the tilt


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def _moment_generating(returns: np.ndarray, theta: float) -> float:
    """K(θ) = log E_P[e^{θR}]  (cumulant generating function)."""
    return np.log(np.mean(np.exp(theta * returns)))


def _tilted_moments(returns: np.ndarray, theta: float) -> tuple:
    """E_Q[R], Var_Q[R], Skew_Q[R] under the Esscher measure Q_θ."""
    w = np.exp(theta * returns)
    w /= w.sum()
    m1 = np.sum(w * returns)
    m2 = np.sum(w * (returns - m1) ** 2)
    m3 = np.sum(w * (returns - m1) ** 3)
    skew = m3 / (m2 ** 1.5 + 1e-12)
    return m1, m2, skew


def _kl_divergence(returns: np.ndarray, theta: float) -> float:
    """D_KL(Q_θ ‖ P) = θ·E_Q[R] - K(θ)."""
    m1, _, _ = _tilted_moments(returns, theta)
    K = _moment_generating(returns, theta)
    return theta * m1 - K


def calibrate(returns: np.ndarray,
              target_mean: float | None = None,
              target_var: float | None = None,
              target_skew: float | None = None) -> EsscherResult:
    """
    Find θ* that minimises distance between tilted moments and option targets.
    If no targets provided, defaults to negatively skewed option surface.

    Parameters
    ----------
    returns      : 1-D historical daily returns
    target_mean  : option-implied mean (annualised if needed, here daily scale)
    target_var   : option-implied variance
    target_skew  : option-implied skew (negative = fear of crash)
    """
    hist_m1, hist_v, hist_sk = _tilted_moments(returns, 0.0)

    # Default option-implied targets: same mean, +30% variance, skew = -1.5
    if target_mean is None:
        target_mean = hist_m1
    if target_var is None:
        target_var = hist_v * 1.30
    if target_skew is None:
        target_skew = -1.5

    targets = np.array([target_mean, target_var, target_skew])
    weights = np.array([1.0, 1.0, 0.5])          # weight each moment

    def loss(theta_vec):
        th = theta_vec[0]
        m, v, sk = _tilted_moments(returns, th)
        diff = np.array([m, v, sk]) - targets
        return np.sum(weights * diff ** 2)

    res = minimize(loss, x0=[0.0], method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 5000})
    theta_star = float(res.x[0])

    m, v, sk = _tilted_moments(returns, theta_star)
    kl = _kl_divergence(returns, theta_star)

    return EsscherResult(
        theta_star=theta_star,
        crash_premium=abs(theta_star),
        tilted_mean=m,
        tilted_var=v,
        tilted_skew=sk,
        historical_mean=hist_m1,
        historical_var=hist_v,
        historical_skew=hist_sk,
        kl_distance=kl,
    )


def rolling_crash_premium(returns: np.ndarray,
                           window: int = 60,
                           target_skew: float = -1.5) -> np.ndarray:
    """Rolling θ* as a crash-premium time series."""
    T = len(returns)
    premium = np.full(T, np.nan)
    for t in range(window, T):
        chunk = returns[t - window: t]
        try:
            res = calibrate(chunk, target_skew=target_skew)
            premium[t] = res.crash_premium
        except Exception:
            pass
    return premium


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(3)

    T = 600
    # calm period: small returns
    r1 = rng.normal(0.0003, 0.008, 300)
    # stress period: fat left tail (crash premium should spike)
    r2 = rng.normal(-0.0002, 0.018, 150) + rng.choice([-0.03, 0], size=150,
                                                        p=[0.05, 0.95])
    # recovery
    r3 = rng.normal(0.0005, 0.010, 150)
    returns = np.concatenate([r1, r2, r3])

    premium = rolling_crash_premium(returns, window=60, target_skew=-1.5)

    # Single calibration snapshot
    snap = calibrate(returns[200:260], target_skew=-1.5)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
    fig.suptitle("Technique 4 — Esscher Tilt / Crash Premium", fontweight="bold")

    ax0 = axes[0]
    ax0.plot(premium, color="crimson", lw=1.5, label="|θ*| crash premium")
    ax0.axvspan(300, 450, alpha=0.15, color="red", label="Stress period")
    ax0.axhline(np.nanmean(premium), color="gray", linestyle="--", lw=1,
                label="Mean premium")
    ax0.set_title("Rolling crash premium  |θ*|  (higher = more fear priced in)")
    ax0.set_ylabel("|θ*|")
    ax0.set_xlabel("Day")
    ax0.legend(fontsize=8)

    ax1 = axes[1]
    x = np.linspace(-0.07, 0.07, 300)
    from scipy.stats import norm
    p_hist = norm.pdf(x, snap.historical_mean, np.sqrt(snap.historical_var))
    # approximate Q_θ via re-weighting a fine grid
    rng2 = np.random.default_rng(99)
    samp = rng2.normal(snap.historical_mean, np.sqrt(snap.historical_var), 10000)
    w = np.exp(snap.theta_star * samp); w /= w.sum()
    ax1.plot(x, p_hist, "b-", lw=2, label="P (historical)")
    ax1.hist(samp, bins=100, weights=w * len(samp), density=True,
             alpha=0.5, color="red", label=f"Q_θ* (option-implied, θ*={snap.theta_star:.2f})")
    ax1.set_title(f"Historical P vs tilted Q_θ*  |  KL = {snap.kl_distance:.4f}")
    ax1.set_xlabel("Daily return")
    ax1.set_ylabel("Density")
    ax1.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("demos/output_04_esscher_tilt.png", dpi=120)
    plt.close()
    print("[4] Esscher Tilt — saved demos/output_04_esscher_tilt.png")
    print(f"    θ* = {snap.theta_star:.4f}  |  crash premium = {snap.crash_premium:.4f}")
    print(f"    Hist skew = {snap.historical_skew:.3f}  →  Tilted skew = {snap.tilted_skew:.3f}")


if __name__ == "__main__":
    demo()
