"""
Technique 9 — Distribution Path Speed & Acceleration

Fit a distribution to returns each day from a rolling window — each day is
a point on the statistical manifold. Compute Fisher-Rao geodesic speed
(first difference) and acceleration (second difference). Sudden speed spikes
= regime transition. Sharp bends (triangle-defect curvature) = structural snap.

Two modes:
  'gaussian'  — parametric Gaussian fit, analytical Fisher-Rao distance
  'histogram' — non-parametric discrete density, Bhattacharyya-form FR distance
                d_FR(p, q) = 2 arccos(Σ_k √(p_k q_k))
                Flag rule: flag_t = 1{D_t > Quantile_{1-α}(D_{t-M,...,t-1})}
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class PathSpeedResult:
    times:        np.ndarray
    speed:        np.ndarray   # vt = d(t, t-Δ)
    acceleration: np.ndarray   # at = Δ speed
    curvature:    np.ndarray   # κt (triangle defect) ≤ 0
    dist:         np.ndarray   # raw FR distances
    alarm_speed:  np.ndarray   # bool: speed > tau_speed
    alarm_curve:  np.ndarray   # bool: curvature < tau_curve
    alarm_flag:   np.ndarray   # bool: quantile-based flag (histogram mode)
    tau_speed:    float
    tau_curve:    float
    mode:         str


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def _fr_distance_gaussian(mu1, sigma1, mu2, sigma2):
    s1, s2 = max(sigma1, 1e-8), max(sigma2, 1e-8)
    arg = 1.0 + ((mu1 - mu2) ** 2 + 2 * (s1 - s2) ** 2) / (2 * s1 * s2)
    arg = max(arg, 1.0)
    return np.sqrt(2) * np.arccosh(arg)


def _fr_distance_histogram(p: np.ndarray, q: np.ndarray) -> float:
    """d_FR(p, q) = 2 arccos(Σ_k √(p_k q_k))  — square-root / Bhattacharyya form."""
    p = np.maximum(p, 1e-12); p = p / p.sum()
    q = np.maximum(q, 1e-12); q = q / q.sum()
    bc = np.sum(np.sqrt(p * q))
    return 2.0 * np.arccos(np.clip(bc, -1.0, 1.0))


def _histogram_density(chunk: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(chunk, bins=edges)
    p = counts.astype(float) + 1e-3
    return p / p.sum()


def _fit(chunk):
    return chunk.mean(), chunk.std(ddof=1) + 1e-8


def run(returns:   np.ndarray,
        window:    int   = 40,
        delta:     int   = 1,
        tau_speed: float = 0.15,
        tau_curve: float = -0.05,
        mode:      str   = "gaussian",
        n_bins:    int   = 30,
        flag_alpha: float = 0.05,
        flag_lookback: int = 60) -> PathSpeedResult:
    """
    Parameters
    ----------
    returns       : (T,) daily returns
    window        : rolling window for distribution estimation
    delta         : step size Δ for speed/acceleration
    tau_speed     : threshold for speed alarm (gaussian mode)
    tau_curve     : threshold for curvature alarm (gaussian mode, negative)
    mode          : 'gaussian' (parametric) or 'histogram' (non-parametric)
    n_bins        : number of histogram bins (histogram mode only)
    flag_alpha    : tail quantile for regime-shift flag (histogram mode)
    flag_lookback : history length M for quantile threshold (histogram mode)
    """
    T = len(returns)

    if mode == "histogram":
        lo = np.percentile(returns, 1)
        hi = np.percentile(returns, 99)
        edges = np.linspace(lo, hi, n_bins + 1)

        densities = []
        times_idx = []
        for t in range(window, T):
            p = _histogram_density(returns[t - window: t], edges)
            densities.append(p)
            times_idx.append(t)

        dists = np.array([
            _fr_distance_histogram(densities[i - 1], densities[i])
            for i in range(1, len(densities))
        ])
        times = np.array(times_idx[1:])

    else:  # gaussian
        thetas = []
        for t in range(window, T):
            mu, sigma = _fit(returns[t - window: t])
            thetas.append((mu, sigma, t))

        dists = np.array([
            _fr_distance_gaussian(thetas[i-1][0], thetas[i-1][1],
                                  thetas[i][0],   thetas[i][1])
            for i in range(1, len(thetas))
        ])
        times = np.array([t for _, _, t in thetas[1:]])

    speed = dists.copy()
    accel = np.diff(speed, prepend=speed[0])

    # triangle-defect curvature: κt = d_{t-2} − (d_t + d_{t-1}) ≤ 0 means curved
    kappa = np.zeros(len(dists))
    for i in range(2, len(dists)):
        kappa[i] = dists[i - 2] - (dists[i] + dists[i - 1])

    # quantile-based flag: flag_t = 1{D_t > Quantile_{1-alpha}(D_{t-M,...,t-1})}
    flag = np.zeros(len(dists), dtype=bool)
    for i in range(flag_lookback, len(dists)):
        threshold = np.quantile(dists[i - flag_lookback: i], 1.0 - flag_alpha)
        flag[i] = dists[i] > threshold

    return PathSpeedResult(
        times=times,
        speed=speed,
        acceleration=accel,
        curvature=kappa,
        dist=dists,
        alarm_speed=speed > tau_speed,
        alarm_curve=kappa < tau_curve,
        alarm_flag=flag,
        tau_speed=tau_speed,
        tau_curve=tau_curve,
        mode=mode,
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(33)

    T = 700
    r1 = rng.normal(0.0002, 0.008, 250)        # calm
    r2 = rng.normal(-0.001, 0.025, 100)         # volatile burst — speed spike
    r3 = rng.normal(0.0001, 0.009, 150)         # calm again
    # sharp distributional bend: skewed then normal
    r4 = np.concatenate([
        rng.laplace(0, 0.012, 50),
        rng.normal(0.0003, 0.010, 150),
    ])
    returns = np.concatenate([r1, r2, r3, r4])

    res = run(returns, window=40, tau_speed=0.12, tau_curve=-0.03)

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    fig.suptitle("Technique 9 — Distribution Path Speed & Acceleration", fontweight="bold")

    t = res.times
    axes[0].plot(returns, color="steelblue", lw=0.7, alpha=0.8)
    axes[0].axvspan(250, 350, alpha=0.15, color="red")
    axes[0].axvspan(500, 600, alpha=0.15, color="purple")
    axes[0].set_ylabel("Return"); axes[0].set_title("Returns")

    axes[1].plot(t, res.speed, color="darkorange", lw=1.5, label="Speed vt")
    axes[1].axhline(res.tau_speed, color="red", linestyle="--", label=f"τ_speed={res.tau_speed}")
    axes[1].fill_between(t, 0, res.speed, where=res.alarm_speed,
                         alpha=0.3, color="red", label="Speed alarm (threshold)")
    axes[1].fill_between(t, 0, res.speed, where=res.alarm_flag,
                         alpha=0.2, color="orange", label="Quantile flag (5%)")
    axes[1].set_ylabel("FR distance / step")
    axes[1].set_title(f"Geodesic speed  vt = d(t, t-Δ)  [mode={res.mode}]  — spike = regime transition")
    axes[1].legend(fontsize=8)

    axes[2].plot(t, res.acceleration, color="purple", lw=1.5, label="Acceleration at")
    axes[2].axhline(0, color="gray", lw=0.5)
    axes[2].set_ylabel("Δ speed")
    axes[2].set_title("Geodesic acceleration  (second difference) — large = decelerating/accelerating path")
    axes[2].legend(fontsize=8)

    axes[3].plot(t, res.curvature, color="seagreen", lw=1.5, label="κt (triangle defect)")
    axes[3].axhline(res.tau_curve, color="red", linestyle="--", label=f"τ_curve={res.tau_curve}")
    axes[3].fill_between(t, res.curvature, 0, where=res.alarm_curve,
                         alpha=0.3, color="red", label="Curvature alarm")
    axes[3].set_ylabel("κt"); axes[3].set_xlabel("Day")
    axes[3].set_title("Triangle-defect curvature  κt ≤ 0  (more negative = sharper bend in manifold path)")
    axes[3].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("demos/output_09_path_speed.png", dpi=120)
    plt.close()
    print("[9] Path Speed — saved demos/output_09_path_speed.png")
    print(f"    Speed alarms: {res.alarm_speed.sum()}, curvature alarms: {res.alarm_curve.sum()}")
    print(f"    Quantile flags: {res.alarm_flag.sum()}")


if __name__ == "__main__":
    demo()
