"""
Technique 6 — Statistical Curvature Throttle

Compute the statistical curvature κ(θ) of the fitted model family.
High curvature means the Fisher (linear) approximation breaks quickly →
small misspecification causes large prediction errors.
Use κ as a continuous lever: reduce leverage, slow calibration, widen error bars.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class CurvatureResult:
    times: np.ndarray
    kappa: np.ndarray           # curvature scalar κ(θ) per window
    alpha_fit: np.ndarray       # trust multiplier ∈ (0,1]
    w_risk: np.ndarray          # adjusted risk weight
    theta: np.ndarray           # rolling (μ, σ) parameters


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def _fisher_information_gaussian(sigma: float) -> np.ndarray:
    """
    Fisher information matrix for N(μ, σ²):
    I(θ) = diag(1/σ², 2/σ²)   (θ = [μ, σ])
    """
    s2 = max(sigma, 1e-8) ** 2
    return np.diag([1.0 / s2, 2.0 / s2])


def _curvature_proxy_gaussian(returns: np.ndarray,
                               mu: float, sigma: float) -> float:
    """
    Practical scalar curvature proxy:
    κ(θ) = ‖I(θ)^{-1/2} E[∇²_θ log p] I(θ)^{-1/2}‖_F

    For Gaussian: ∇²_μ log p = -1/σ², ∇²_σ log p = 1/σ² - 3(r-μ)²/σ⁴
    We compute E[∇²] from the sample.
    """
    s = max(sigma, 1e-8)
    # second derivatives of log-likelihood
    d2_mu  = -1.0 / s ** 2                          # constant
    d2_sig = np.mean(1.0 / s ** 2 - 3 * (returns - mu) ** 2 / s ** 4)

    I = _fisher_information_gaussian(s)
    I_inv_half = np.diag(1.0 / np.sqrt(np.diag(I)))

    H = np.diag([d2_mu, d2_sig])
    standardised = I_inv_half @ H @ I_inv_half
    kappa = np.linalg.norm(standardised, "fro")
    return float(kappa)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(returns: np.ndarray,
        window: int = 60,
        c: float = 2.0,
        w_risk_base: float = 1.0) -> CurvatureResult:
    """
    Parameters
    ----------
    returns      : 1-D daily returns
    window       : rolling calibration window
    c            : throttle steepness
    w_risk_base  : baseline risk weight (full conviction)
    """
    T = len(returns)
    kappas, alphas, ws, thetas = [], [], [], []

    for t in range(window, T):
        chunk = returns[t - window: t]
        mu = chunk.mean()
        sigma = chunk.std(ddof=1)

        kappa = _curvature_proxy_gaussian(chunk, mu, sigma)
        alpha_fit = 1.0 / (1.0 + c * kappa)
        w = alpha_fit * w_risk_base

        kappas.append(kappa)
        alphas.append(alpha_fit)
        ws.append(w)
        thetas.append([mu, sigma])

    return CurvatureResult(
        times=np.arange(window, T),
        kappa=np.array(kappas),
        alpha_fit=np.array(alphas),
        w_risk=np.array(ws),
        theta=np.array(thetas),
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(11)

    T = 600
    # Curvature will be high near a bimodal / mixture region
    r1 = rng.normal(0.0002, 0.008, 200)
    # Transition: mixture of two regimes → high curvature
    mix = np.where(rng.random(150) < 0.5,
                   rng.normal(0.002, 0.005, 150),
                   rng.normal(-0.002, 0.018, 150))
    r3 = rng.normal(-0.0001, 0.010, 250)
    returns = np.concatenate([r1, mix, r3])

    res = run(returns, window=60, c=3.0)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Technique 6 — Statistical Curvature Throttle", fontweight="bold")

    t = res.times
    axes[0].plot(returns, color="steelblue", lw=0.7, alpha=0.8)
    axes[0].axvspan(200, 350, alpha=0.15, color="orange", label="Mixture / transition zone")
    axes[0].set_ylabel("Return"); axes[0].set_title("Returns (mixture zone in orange)")
    axes[0].legend(fontsize=8)

    axes[1].plot(t, res.kappa, color="darkorange", lw=1.5, label="κ(θ) curvature")
    axes[1].axvspan(200, 350, alpha=0.15, color="orange")
    axes[1].set_ylabel("κ(θ)")
    axes[1].set_title("Curvature κ(θ) — high in bimodal / misspecified regions")
    axes[1].legend(fontsize=8)

    axes[2].plot(t, res.w_risk, color="seagreen", lw=1.5, label="Adjusted risk weight")
    axes[2].axhline(1.0, color="gray", linestyle="--", lw=1, label="Full conviction")
    axes[2].fill_between(t, res.w_risk, 1.0, alpha=0.3, color="red",
                         label="Throttled")
    axes[2].axvspan(200, 350, alpha=0.1, color="orange")
    axes[2].set_ylabel("w_risk")
    axes[2].set_xlabel("Day")
    axes[2].set_title("Risk weight  α_fit(θ) = 1/(1+c·κ)  — auto-deleverage in curved regions")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("demos/output_06_curvature_throttle.png", dpi=120)
    plt.close()
    print("[6] Curvature Throttle — saved demos/output_06_curvature_throttle.png")
    print(f"    Peak κ = {res.kappa.max():.4f}  →  min risk weight = {res.w_risk.min():.3f}")


if __name__ == "__main__":
    demo()
