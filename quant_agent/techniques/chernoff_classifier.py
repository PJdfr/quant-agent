"""
Technique 19 — Chernoff Information Classifier

Given two hypotheses (calm H0: X~p0, stress H1: X~p1), Chernoff information
gives the optimal exponential error-decay rate of the Bayes classifier:

    C(p0, p1) = -log(inf_{s∈[0,1]} ∫ p0(x)^s p1(x)^{1-s} dx)

The minimising s* gives the "tilted boundary" density m_s*(x) ∝ p0^s* p1^{1-s*},
the geometric midpoint between the two distributions. The log-likelihood ratio

    Λ(x) = log p1(x)/p0(x)

with threshold τ derived from m_s* gives principled entry/exit rules where
classification errors decay exponentially.
"""

import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize_scalar
from scipy.stats import norm


@dataclass
class ChernoffResult:
    s_star:           float         # optimal tilt
    chernoff_info:    float         # C(p0, p1) — higher = better separable
    tau:              float         # LLR threshold
    bayes_error_rate: float         # approximate Bayes error ≈ exp(-C)
    llr_series:       np.ndarray    # Λ(x_t) for each observation
    decision:         np.ndarray    # +1 stress / 0 calm per observation
    alarm_exit:       np.ndarray    # bool: Λ > τ_exit
    alarm_entry:      np.ndarray    # bool: Λ < τ_entry
    tau_exit:         float
    tau_entry:        float
    p0_params:        dict          # calm distribution params
    p1_params:        dict          # stress distribution params


# ---------------------------------------------------------------------------
# Core math (Gaussian hypothesis pair)
# ---------------------------------------------------------------------------

def _chernoff_gaussian(mu0, sig0, mu1, sig1):
    """
    C(p0, p1) for Gaussians.
    Rényi α-divergence: D_α(p0‖p1) = 1/(α-1) log ∫ p0^α p1^{1-α}
    For Gaussians: analytic.
    We minimise over s ∈ (0,1).
    """
    def neg_log_mixture(s):
        # log ∫ N(μ0,σ0²)^s N(μ1,σ1²)^{1-s} dx  (Gaussian case, closed form)
        sigma_s_sq = 1.0 / (s / sig0**2 + (1-s) / sig1**2)
        mu_s = sigma_s_sq * (s * mu0 / sig0**2 + (1-s) * mu1 / sig1**2)
        log_nc = (0.5 * np.log(2*np.pi*sigma_s_sq)
                  - s * 0.5 * np.log(2*np.pi*sig0**2)
                  - (1-s) * 0.5 * np.log(2*np.pi*sig1**2)
                  - 0.5 * (s * mu0**2/sig0**2 + (1-s)*mu1**2/sig1**2 - mu_s**2/sigma_s_sq))
        return log_nc   # we want to minimise this (= -C)

    res = minimize_scalar(neg_log_mixture, bounds=(0.01, 0.99), method="bounded")
    s_star = float(res.x)
    C = float(-res.fun)
    return s_star, C


def _llr_gaussian(x: np.ndarray, mu0, sig0, mu1, sig1) -> np.ndarray:
    """Λ(x) = log p1(x)/p0(x) for Gaussian pair."""
    lp1 = norm.logpdf(x, mu1, sig1)
    lp0 = norm.logpdf(x, mu0, sig0)
    return lp1 - lp0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(observations:  np.ndarray,
        p0_params:     dict,
        p1_params:     dict,
        prior_ratio:   float = 1.0,
        tau_exit_mult: float = 1.0,
        tau_entry_mult: float = -0.5) -> ChernoffResult:
    """
    Parameters
    ----------
    observations   : (T,) feature values (e.g. rolling vol, return, spread)
    p0_params      : {'mu': float, 'sigma': float}  calm distribution
    p1_params      : {'mu': float, 'sigma': float}  stress distribution
    prior_ratio    : π0/π1  (prior odds calm vs stress); log gives τ offset
    tau_exit_mult  : scale τ for exit  (Λ > τ_exit → exit/de-lever)
    tau_entry_mult : scale τ for entry (Λ < τ_entry → allow entry)
    """
    mu0, s0 = p0_params["mu"], p0_params["sigma"]
    mu1, s1 = p1_params["mu"], p1_params["sigma"]

    s_star, C = _chernoff_gaussian(mu0, s0, mu1, s1)
    tau_base   = np.log(prior_ratio)   # Bayes threshold = log π0/π1

    tau = tau_base
    tau_exit  = tau * tau_exit_mult  if tau_exit_mult  >= 0 else 0.5 * C
    tau_entry = tau * tau_entry_mult if tau_entry_mult <= 0 else -0.5 * C

    llr = _llr_gaussian(observations, mu0, s0, mu1, s1)
    decision    = (llr > tau).astype(int)
    alarm_exit  = llr > tau_exit
    alarm_entry = llr < tau_entry

    return ChernoffResult(
        s_star=s_star,
        chernoff_info=C,
        tau=tau,
        bayes_error_rate=float(np.exp(-C)),
        llr_series=llr,
        decision=decision,
        alarm_exit=alarm_exit,
        alarm_entry=alarm_entry,
        tau_exit=tau_exit,
        tau_entry=tau_entry,
        p0_params=p0_params,
        p1_params=p1_params,
    )


def rolling_classify(observations: np.ndarray,
                     p0_params:    dict,
                     p1_params:    dict,
                     window:       int = 40) -> np.ndarray:
    """Smooth the LLR with a rolling mean before thresholding."""
    llr_raw = _llr_gaussian(observations,
                             p0_params["mu"], p0_params["sigma"],
                             p1_params["mu"], p1_params["sigma"])
    llr_smooth = np.convolve(llr_raw, np.ones(window)/window, mode="same")
    return llr_smooth


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(19)

    T = 800
    # Feature: rolling 20-day realised vol
    true_state = np.zeros(T, dtype=int)
    true_state[200:400] = 1   # stress
    true_state[600:720] = 1   # stress again

    feature = np.where(true_state == 0,
                       rng.normal(0.010, 0.002, T),   # calm: low vol
                       rng.normal(0.022, 0.004, T))   # stress: high vol

    p0 = {"mu": 0.010, "sigma": 0.003}
    p1 = {"mu": 0.022, "sigma": 0.005}

    res = run(feature, p0, p1, prior_ratio=1.0,
              tau_exit_mult=0.8, tau_entry_mult=-0.3)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("Technique 19 — Chernoff Information Classifier", fontweight="bold")

    t = np.arange(T)
    axes[0].plot(t, feature, color="steelblue", lw=0.8, label="Feature (realised vol)")
    axes[0].axhline(p0["mu"], color="seagreen", linestyle="--", label=f"μ_calm={p0['mu']}")
    axes[0].axhline(p1["mu"], color="crimson",  linestyle="--", label=f"μ_stress={p1['mu']}")
    axes[0].fill_between(t, 0, 0.04, where=true_state==1,
                         alpha=0.15, color="red", label="True stress")
    axes[0].set_ylabel("Feature"); axes[0].legend(fontsize=8)
    axes[0].set_title(f"Feature values  |  Chernoff info C = {res.chernoff_info:.4f}  "
                      f"(Bayes error ≈ {res.bayes_error_rate:.3f})")

    axes[1].plot(t, res.llr_series, color="purple", lw=1.0, label="Λ(x) = log p1/p0")
    axes[1].axhline(res.tau_exit,  color="red",     linestyle="--",
                    label=f"τ_exit = {res.tau_exit:.2f}  (de-lever if Λ > this)")
    axes[1].axhline(res.tau_entry, color="seagreen", linestyle="--",
                    label=f"τ_entry = {res.tau_entry:.2f}  (allow if Λ < this)")
    axes[1].fill_between(t, 0, res.llr_series, where=res.alarm_exit,
                         alpha=0.3, color="red", label="Exit zone")
    axes[1].fill_between(t, res.llr_series, 0, where=res.alarm_entry,
                         alpha=0.2, color="seagreen", label="Entry zone")
    axes[1].axhline(0, color="gray", lw=0.5)
    axes[1].set_ylabel("LLR Λ(x)"); axes[1].legend(fontsize=7)
    axes[1].set_title("Log-likelihood ratio with Chernoff-derived thresholds")

    # accuracy
    accuracy = (res.decision == true_state).mean()
    axes[2].plot(t, res.decision, color="crimson", lw=0.8,
                 drawstyle="steps-post", label="Predicted state (1=stress)")
    axes[2].plot(t, true_state, color="steelblue", lw=0.8, alpha=0.5,
                 drawstyle="steps-post", label="True state", linestyle="--")
    axes[2].set_ylabel("State"); axes[2].set_xlabel("Day")
    axes[2].set_title(f"Classification accuracy: {accuracy*100:.1f}%  "
                      f"|  s* = {res.s_star:.3f}")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("demos/output_19_chernoff_classifier.png", dpi=120)
    plt.close()
    print("[19] Chernoff Classifier — saved demos/output_19_chernoff_classifier.png")
    print(f"     C = {res.chernoff_info:.4f}, s* = {res.s_star:.3f}, "
          f"Bayes error ≈ {res.bayes_error_rate:.4f}")
    print(f"     Classification accuracy: {accuracy*100:.1f}%")


if __name__ == "__main__":
    demo()
