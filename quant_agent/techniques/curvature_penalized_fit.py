"""
Technique 16 — Curvature-Penalized Fitting

Add the local curvature κ(θ) of the statistical manifold to the fitting
objective so the optimizer gravitates toward flatter, more stable regions:

    θ* = argmin_θ  L(θ) + λ κ(θ)

In high-curvature zones small estimation errors blow up into large
distribution errors, so penalising curvature yields more robust forecasts.
Rolling version: κ also throttles the step-size of parameter updates.
"""

import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize


@dataclass
class CurvatureFitResult:
    theta_penalised: np.ndarray     # optimal parameters with curvature penalty
    theta_plain:     np.ndarray     # optimal parameters without penalty (baseline)
    kappa_penalised: float          # curvature at penalised solution
    kappa_plain:     float          # curvature at plain solution
    loss_penalised:  float          # total loss (data + curvature)
    loss_data_only:  float          # data-only loss at penalised solution
    size_t:          float          # fragility throttle: 1/(1+c·κ)
    lambda_used:     float


# ---------------------------------------------------------------------------
# Gaussian manifold helpers (same as curvature_throttle, self-contained)
# ---------------------------------------------------------------------------

def _nll_gaussian(theta: np.ndarray, returns: np.ndarray) -> float:
    """Negative log-likelihood of N(μ, σ²)."""
    mu, log_sigma = theta
    sigma = np.exp(log_sigma)
    return 0.5 * np.sum(((returns - mu) / sigma) ** 2) + len(returns) * log_sigma


def _kappa_gaussian(theta: np.ndarray, returns: np.ndarray) -> float:
    """
    Scalar curvature proxy κ(θ) = ‖I(θ)^{-1/2} E[∇²log p] I(θ)^{-1/2}‖_F.
    θ = [μ, log σ].
    """
    mu, log_sigma = theta
    sigma = max(np.exp(log_sigma), 1e-6)
    n = len(returns)
    # Fisher information for (μ, log σ): diag(n/σ², 2n)
    I_diag = np.array([n / sigma**2, 2.0 * n])
    I_inv_half = 1.0 / np.sqrt(np.maximum(I_diag, 1e-10))
    # E[∂²/∂μ² log p] = -n/σ²
    # E[∂²/∂(log σ)² log p] = -(1/n) Σ (r-μ)²/σ² + 1  (approx)
    d2_mu  = -n / sigma**2
    d2_sig = np.mean(-(returns - mu)**2 / sigma**2) * n
    H_diag = np.array([d2_mu, d2_sig])
    standardised = I_inv_half * H_diag * I_inv_half
    return float(np.linalg.norm(standardised))


# ---------------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------------

def fit(returns: np.ndarray, lam: float = 0.5) -> CurvatureFitResult:
    """
    Fit Gaussian parameters with and without curvature penalty.

    Parameters
    ----------
    returns : 1-D daily returns
    lam     : curvature penalty weight λ > 0
    """
    theta0 = np.array([returns.mean(), np.log(returns.std(ddof=1) + 1e-8)])

    # Plain MLE
    res_plain = minimize(lambda th: _nll_gaussian(th, returns), theta0,
                         method="Nelder-Mead", options={"xatol": 1e-6, "maxiter": 2000})
    theta_plain = res_plain.x

    # Curvature-penalised
    def penalised_obj(th):
        return _nll_gaussian(th, returns) + lam * _kappa_gaussian(th, returns)

    res_pen = minimize(penalised_obj, theta0,
                       method="Nelder-Mead", options={"xatol": 1e-6, "maxiter": 2000})
    theta_pen = res_pen.x

    kap_pen   = _kappa_gaussian(theta_pen, returns)
    kap_plain = _kappa_gaussian(theta_plain, returns)
    size_t    = 1.0 / (1.0 + kap_pen)

    return CurvatureFitResult(
        theta_penalised=theta_pen,
        theta_plain=theta_plain,
        kappa_penalised=kap_pen,
        kappa_plain=kap_plain,
        loss_penalised=float(res_pen.fun),
        loss_data_only=float(_nll_gaussian(theta_pen, returns)),
        size_t=size_t,
        lambda_used=lam,
    )


def rolling_fit(returns: np.ndarray, window: int = 60,
                lam: float = 0.5) -> dict:
    """Rolling curvature-penalised fit. Returns dict of arrays."""
    T = len(returns)
    thetas_pen, thetas_plain, kappas, sizes = [], [], [], []
    for t in range(window, T):
        r = fit(returns[t - window: t], lam=lam)
        thetas_pen.append(r.theta_penalised)
        thetas_plain.append(r.theta_plain)
        kappas.append(r.kappa_penalised)
        sizes.append(r.size_t)
    return dict(
        times=np.arange(window, T),
        theta_pen=np.array(thetas_pen),
        theta_plain=np.array(thetas_plain),
        kappa=np.array(kappas),
        size=np.array(sizes),
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(16)

    T = 500
    # calm → mixture → calm  (mixture = high curvature zone)
    r1 = rng.normal(0.0002, 0.008, 200)
    mix = np.where(rng.random(100) < 0.5,
                   rng.normal(0.003, 0.005, 100),
                   rng.normal(-0.003, 0.020, 100))
    r3 = rng.normal(0.0001, 0.009, 200)
    returns = np.concatenate([r1, mix, r3])

    out = rolling_fit(returns, window=60, lam=1.0)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("Technique 16 — Curvature-Penalized Fitting", fontweight="bold")

    t = out["times"]
    sigma_pen   = np.exp(out["theta_pen"][:, 1])   * np.sqrt(252)
    sigma_plain = np.exp(out["theta_plain"][:, 1]) * np.sqrt(252)

    axes[0].plot(t, sigma_plain, color="crimson",   lw=1.5, label="σ plain MLE")
    axes[0].plot(t, sigma_pen,   color="steelblue", lw=1.5, label="σ curvature-penalised")
    axes[0].axvspan(200, 300, alpha=0.15, color="orange", label="High-curvature zone")
    axes[0].set_ylabel("Annualised σ")
    axes[0].set_title("Fitted volatility: penalised fit avoids sharp curvature-zone excursions")
    axes[0].legend(fontsize=8)

    axes[1].plot(t, out["kappa"], color="darkorange", lw=1.5, label="κ(θ) curvature")
    axes[1].axvspan(200, 300, alpha=0.15, color="orange")
    axes[1].set_ylabel("κ(θ)")
    axes[1].set_title("Local manifold curvature — penalised optimizer steers away from peaks")
    axes[1].legend(fontsize=8)

    axes[2].plot(t, out["size"], color="seagreen", lw=1.5, label="Fragility throttle size_t")
    axes[2].axhline(1.0, color="gray", linestyle="--", lw=1)
    axes[2].fill_between(t, out["size"], 1.0, alpha=0.3, color="red")
    axes[2].axvspan(200, 300, alpha=0.1, color="orange")
    axes[2].set_ylabel("size_t = 1/(1+κ)"); axes[2].set_xlabel("Day")
    axes[2].set_title("Position size throttled by curvature of penalised solution")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("demos/output_16_curvature_penalized_fit.png", dpi=120)
    plt.close()
    print("[16] Curvature-Penalized Fit — saved demos/output_16_curvature_penalized_fit.png")
    snap = fit(returns[200:260], lam=1.0)
    print(f"     Mixture zone: κ_plain={snap.kappa_plain:.3f}  κ_penalised={snap.kappa_penalised:.3f}")
    print(f"     size_t = {snap.size_t:.3f}")


if __name__ == "__main__":
    demo()
