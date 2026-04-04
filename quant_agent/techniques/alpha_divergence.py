"""
Technique 5 — α-Divergence Decay Alarm

Compare the model's predictive distribution p_t against the realised
distribution q_t estimated from a rolling window of outcomes.
Rising D_α signals calibration drift, regime change, or execution slippage.
Use as a decay alarm and as a risk throttle.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class AlphaResult:
    times: np.ndarray           # day index
    divergence: np.ndarray      # D_α(p‖q) per window
    alarm: np.ndarray           # bool — divergence above threshold τ
    risk_multiplier: np.ndarray # ∈ (0,1] throttle on risk
    alpha: float
    tau: float
    c: float                    # throttle steepness


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def _alpha_divergence_discrete(p: np.ndarray, q: np.ndarray,
                                alpha: float) -> float:
    """
    D_α(p‖q) = 1/(α(1-α)) * (1 - Σ p_i^α * q_i^(1-α))
    α ∈ (0,1); α→0 = KL(q‖p), α→1 = KL(p‖q)
    """
    p = np.maximum(p, 1e-12)
    q = np.maximum(q, 1e-12)
    p = p / p.sum()
    q = q / q.sum()
    inner = np.sum(p ** alpha * q ** (1 - alpha))
    return (1.0 - inner) / (alpha * (1.0 - alpha))


def run(pred_params: np.ndarray,
        outcomes: np.ndarray,
        window: int = 40,
        alpha: float = 0.5,
        tau: float = 0.05,
        c: float = 10.0,
        n_bins: int = 25) -> AlphaResult:
    """
    Parameters
    ----------
    pred_params : (T, 2) — rolling (μ_pred, σ_pred) from model
    outcomes    : (T,)   — realised returns
    window      : window for estimating q_t from recent outcomes
    alpha       : α-divergence parameter  ∈ (0,1)
    tau         : alarm threshold
    c           : throttle steepness
    n_bins      : histogram bins for density estimation

    Returns
    -------
    AlphaResult
    """
    from scipy.stats import norm

    T = len(outcomes)
    lo = np.percentile(outcomes, 1)
    hi = np.percentile(outcomes, 99)
    edges = np.linspace(lo, hi, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]

    divs, alarms, throttles = [], [], []
    start = window

    for t in range(start, T):
        mu_p, sigma_p = pred_params[t, 0], max(pred_params[t, 1], 1e-6)
        # predictive distribution p_t: discretised Gaussian
        p = norm.pdf(centres, mu_p, sigma_p) * width
        p /= p.sum()

        # realised distribution q_t: histogram of recent outcomes
        chunk = outcomes[t - window: t]
        counts, _ = np.histogram(chunk, bins=edges)
        q = counts.astype(float) + 1e-3    # Laplace smoothing
        q /= q.sum()

        D = _alpha_divergence_discrete(p, q, alpha)
        risk = 1.0 / (1.0 + c * D)
        divs.append(D)
        alarms.append(D > tau)
        throttles.append(risk)

    return AlphaResult(
        times=np.arange(start, T),
        divergence=np.array(divs),
        alarm=np.array(alarms),
        risk_multiplier=np.array(throttles),
        alpha=alpha,
        tau=tau,
        c=c,
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(55)

    T = 600
    # model was calibrated on calm regime
    mu_true = np.concatenate([
        np.full(300, 0.0002),
        np.full(150, -0.0010),   # regime shift — model doesn't know
        np.full(150, 0.0003),
    ])
    sigma_true = np.concatenate([
        np.full(300, 0.008),
        np.full(150, 0.022),     # vol spike
        np.full(150, 0.009),
    ])

    outcomes = rng.normal(mu_true, sigma_true)

    # model predicts calm throughout (misses the shift)
    pred_params = np.column_stack([
        np.full(T, 0.0002),
        np.full(T, 0.008),
    ])

    res = run(pred_params, outcomes, window=40, alpha=0.5, tau=0.05)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("Technique 5 — α-Divergence Decay Alarm", fontweight="bold")

    t = res.times
    axes[0].plot(outcomes, color="steelblue", lw=0.8, alpha=0.7, label="Realised returns")
    axes[0].axvspan(300, 450, alpha=0.15, color="red")
    axes[0].set_ylabel("Return"); axes[0].set_title("Realised vs model-predicted returns")
    axes[0].axhline(0, color="gray", lw=0.5)

    axes[1].plot(t, res.divergence, color="darkorange", lw=1.5,
                 label=f"D_α (α={res.alpha})")
    axes[1].axhline(res.tau, color="red", linestyle="--", label=f"τ = {res.tau}")
    axes[1].fill_between(t, 0, res.divergence, where=res.alarm,
                         alpha=0.3, color="red", label="Alarm zone")
    axes[1].axvspan(300, 450, alpha=0.1, color="red")
    axes[1].set_ylabel("Divergence")
    axes[1].set_title("α-Divergence D_α(p‖q)  — rising = model drifting from reality")
    axes[1].legend(fontsize=8)

    axes[2].plot(t, res.risk_multiplier, color="seagreen", lw=1.5)
    axes[2].axhline(1.0, color="gray", linestyle="--", lw=1)
    axes[2].fill_between(t, res.risk_multiplier, 1.0, alpha=0.3, color="red",
                         label="Risk throttled")
    axes[2].axvspan(300, 450, alpha=0.1, color="red")
    axes[2].set_ylabel("Risk multiplier")
    axes[2].set_xlabel("Day")
    axes[2].set_title("Risk throttle  1/(1 + c·D_α)  — auto-size reduction when model decays")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("demos/output_05_alpha_divergence.png", dpi=120)
    plt.close()
    print("[5] α-Divergence — saved demos/output_05_alpha_divergence.png")
    print(f"    Alarm days: {res.alarm.sum()} / {len(res.alarm)}")
    print(f"    Min risk multiplier: {res.risk_multiplier.min():.3f}")


if __name__ == "__main__":
    demo()
