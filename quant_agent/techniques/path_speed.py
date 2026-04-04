"""
Technique 9 — Distribution Path Speed & Acceleration

Fit a distribution to returns each day from a rolling window — each day is
a point on the statistical manifold. Compute Fisher-Rao geodesic speed
(first difference) and acceleration (second difference). Sudden speed spikes
= regime transition. Sharp bends (triangle-defect curvature) = structural snap.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class PathSpeedResult:
    times: np.ndarray
    speed: np.ndarray          # vt = d(t, t-Δ)
    acceleration: np.ndarray   # at = d(t,t-Δ) - d(t-Δ, t-2Δ)
    curvature: np.ndarray      # κt (triangle defect) ≤ 0
    dist: np.ndarray           # raw FR distances
    alarm_speed: np.ndarray    # bool
    alarm_curve: np.ndarray    # bool
    tau_speed: float
    tau_curve: float


# ---------------------------------------------------------------------------
# Core math (reuse Gaussian FR distance)
# ---------------------------------------------------------------------------

def _fr_distance_gaussian(mu1, sigma1, mu2, sigma2):
    s1, s2 = max(sigma1, 1e-8), max(sigma2, 1e-8)
    arg = 1.0 + ((mu1 - mu2) ** 2 + 2 * (s1 - s2) ** 2) / (2 * s1 * s2)
    arg = max(arg, 1.0)
    return np.sqrt(2) * np.arccosh(arg)


def _fit(chunk):
    return chunk.mean(), chunk.std(ddof=1) + 1e-8


def run(returns: np.ndarray,
        window: int = 40,
        delta: int = 1,
        tau_speed: float = 0.15,
        tau_curve: float = -0.05) -> PathSpeedResult:
    """
    Parameters
    ----------
    returns    : 1-D daily returns
    window     : rolling window for distribution estimation
    delta      : step size Δ for speed/acceleration
    tau_speed  : alarm threshold for speed
    tau_curve  : alarm threshold for curvature (negative = bend)
    """
    T = len(returns)
    thetas = []

    for t in range(window, T):
        mu, sigma = _fit(returns[t - window: t])
        thetas.append((mu, sigma, t))

    N = len(thetas)
    # pairwise distances along time (sequential)
    dists = []
    for i in range(1, N):
        m1, s1, _ = thetas[i - 1]
        m2, s2, _ = thetas[i]
        dists.append(_fr_distance_gaussian(m1, s1, m2, s2))
    dists = np.array(dists)

    # speed: vt = d(t, t-delta)
    speed = dists.copy()

    # acceleration: at = d(t, t-delta) - d(t-delta, t-2delta)
    accel = np.diff(speed, prepend=speed[0])

    # triangle defect curvature: κt = d(t-2δ, t-2δ) - (d(t,t-δ) + d(t-δ,t-2δ))
    # = dists[i-2] - (dists[i] + dists[i-1])  → ≤ 0 means curved
    kappa = np.zeros(len(dists))
    for i in range(2, len(dists)):
        kappa[i] = dists[i - 2] - (dists[i] + dists[i - 1])

    times = np.array([t for _, _, t in thetas[1:]])

    return PathSpeedResult(
        times=times,
        speed=speed,
        acceleration=accel,
        curvature=kappa,
        dist=dists,
        alarm_speed=speed > tau_speed,
        alarm_curve=kappa < tau_curve,
        tau_speed=tau_speed,
        tau_curve=tau_curve,
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
                         alpha=0.3, color="red", label="Speed alarm")
    axes[1].set_ylabel("FR distance / step")
    axes[1].set_title("Geodesic speed  vt = d(t, t-Δ)  — spike = regime transition")
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


if __name__ == "__main__":
    demo()
