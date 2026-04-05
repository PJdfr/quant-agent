"""
Technique 17 — Geodesic Macro Regression

Treat the predictive distribution as a point p_θ on a statistical manifold.
Fit a geodesic map from macro covariates x_t (rates, VIX, spreads) to
distribution points via the exponential map:

    θ(x) = Exp_{θ_0}(Bx)

Minimise total squared manifold distance between observed and macro-implied
distribution points. Trade when the geometric deviation δ_t = d(p_θ_t, p̂_t)
exceeds a threshold ε — the market is mis-priced given the macro state.
"""

import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize


@dataclass
class GeodesicMacroResult:
    times:         np.ndarray   # time index
    delta:         np.ndarray   # deviation score δ_t = d(p_obs, p_macro_implied)
    signal:        np.ndarray   # +1 (obs > implied) / -1 / 0 (within ε)
    size:          np.ndarray   # position size ∝ (δ_t − ε)+
    theta_obs:     np.ndarray   # (T, 2) observed rolling parameters
    theta_implied: np.ndarray   # (T, 2) macro-implied parameters
    B_hat:         np.ndarray   # regression matrix (2, d)
    theta0:        np.ndarray   # intercept distribution point
    epsilon:       float
    macro_names:   list


# ---------------------------------------------------------------------------
# Geometry helpers (Gaussian manifold)
# ---------------------------------------------------------------------------

def _fr_dist(mu1, sig1, mu2, sig2):
    s1, s2 = max(sig1, 1e-8), max(sig2, 1e-8)
    arg = 1.0 + ((mu1 - mu2)**2 + 2*(s1 - s2)**2) / (2*s1*s2)
    return float(np.sqrt(2) * np.arccosh(max(arg, 1.0)))


def _exp_map(theta0: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Approximate exponential map on Gaussian manifold.
    θ0 = [μ0, σ0], v = tangent vector, returns new θ.
    We use the flat (Euclidean) approximation: θ = θ0 + v,
    then clip σ > 0.
    """
    th = theta0 + v
    th[1] = max(th[1], 1e-4)   # σ must be positive
    return th


# ---------------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------------

def fit_and_trade(returns:     np.ndarray,
                  macro:       np.ndarray,
                  window:      int   = 60,
                  epsilon:     float = 0.10,
                  macro_names: list  | None = None) -> GeodesicMacroResult:
    """
    Parameters
    ----------
    returns      : (T,)   daily returns
    macro        : (T, d) macro covariates (standardised recommended)
    window       : rolling window for observed distribution estimation
    epsilon      : deviation threshold for trading
    macro_names  : list of d covariate names
    """
    T = len(returns)
    d = macro.shape[1]
    macro_names = macro_names or [f"x{i}" for i in range(d)]

    # Step 1: compute rolling observed parameters θ_t = (μ_t, σ_t)
    thetas_obs = np.full((T, 2), np.nan)
    for t in range(window, T):
        chunk = returns[t - window: t]
        thetas_obs[t] = [chunk.mean(), chunk.std(ddof=1)]

    valid = ~np.isnan(thetas_obs[:, 0])
    idx   = np.where(valid)[0]
    Y     = thetas_obs[idx]           # (N, 2) observed parameters
    X     = macro[idx]                # (N, d) macro covariates

    # Step 2: fit geodesic regression
    # θ_t = θ_0 + B x_t  (linearised exponential map)
    # minimise Σ_t d(θ_obs_t, θ_0 + B x_t)²
    def loss(params):
        theta0 = params[:2]
        B      = params[2:].reshape(2, d)
        total  = 0.0
        for i in range(len(Y)):
            th_imp = _exp_map(theta0, B @ X[i])
            total += _fr_dist(Y[i, 0], Y[i, 1], th_imp[0], th_imp[1]) ** 2
        return total

    theta0_init = np.array([Y[:, 0].mean(), Y[:, 1].mean()])
    B_init      = np.zeros(2 * d)
    params0     = np.concatenate([theta0_init, B_init])

    res = minimize(loss, params0, method="Nelder-Mead",
                   options={"maxiter": 5000, "xatol": 1e-4, "fatol": 1e-4})
    theta0_hat = res.x[:2]
    B_hat      = res.x[2:].reshape(2, d)

    # Step 3: compute deviation scores and signals
    thetas_implied = np.full((T, 2), np.nan)
    deltas         = np.full(T, np.nan)
    signals        = np.zeros(T, dtype=int)
    sizes          = np.zeros(T)

    for t in idx:
        th_imp = _exp_map(theta0_hat, B_hat @ macro[t])
        thetas_implied[t] = th_imp
        delta_t = _fr_dist(thetas_obs[t, 0], thetas_obs[t, 1],
                           th_imp[0], th_imp[1])
        deltas[t] = delta_t
        if delta_t > epsilon:
            excess = delta_t - epsilon
            sizes[t] = min(1.0, excess / epsilon)
            # direction: obs mean above implied → positive signal
            signals[t] = 1 if thetas_obs[t, 0] > th_imp[0] else -1

    return GeodesicMacroResult(
        times=idx,
        delta=deltas[idx],
        signal=signals[idx],
        size=sizes[idx],
        theta_obs=thetas_obs[idx],
        theta_implied=thetas_implied[idx],
        B_hat=B_hat,
        theta0=theta0_hat,
        epsilon=epsilon,
        macro_names=macro_names,
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(17)

    T = 500
    # macro: VIX-like and rate-like (standardised)
    vix   = rng.normal(0, 1, T)
    rates = rng.normal(0, 1, T)
    macro = np.column_stack([vix, rates])

    # True return distribution: σ driven by VIX, μ driven by rates
    true_sig = 0.008 + 0.005 * np.abs(vix)
    true_mu  = 0.0002 + 0.0003 * rates
    returns  = rng.normal(true_mu, true_sig)

    # Introduce a dislocation: at t=300, market drifts away from macro-implied
    returns[300:380] += 0.005   # positive drift not explained by macro

    res = fit_and_trade(returns, macro, window=60, epsilon=0.08,
                        macro_names=["VIX", "Rates"])

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("Technique 17 — Geodesic Macro Regression", fontweight="bold")

    t = res.times
    axes[0].plot(t, res.theta_obs[:, 0] * 252,     color="steelblue", lw=1.2, label="Observed μ (ann.)")
    axes[0].plot(t, res.theta_implied[:, 0] * 252, color="crimson",   lw=1.2, linestyle="--",
                 label="Macro-implied μ (ann.)")
    axes[0].axvspan(300, 380, alpha=0.15, color="orange", label="Macro dislocation")
    axes[0].set_ylabel("μ (annualised)"); axes[0].legend(fontsize=8)
    axes[0].set_title("Observed vs macro-implied distribution mean")

    axes[1].plot(t, res.delta, color="darkorange", lw=1.5, label="δ_t (geo. deviation)")
    axes[1].axhline(res.epsilon, color="red", linestyle="--", label=f"ε = {res.epsilon}")
    axes[1].fill_between(t, 0, res.delta, where=res.delta > res.epsilon,
                         alpha=0.3, color="red", label="Trade zone")
    axes[1].axvspan(300, 380, alpha=0.1, color="orange")
    axes[1].set_ylabel("FR distance"); axes[1].legend(fontsize=8)
    axes[1].set_title("Deviation score δ_t = d(p_obs, p_macro_implied)  — trade when δ_t > ε")

    axes[2].scatter(t[res.signal > 0], res.size[res.signal > 0],
                    color="seagreen", s=15, label="Long signal")
    axes[2].scatter(t[res.signal < 0], res.size[res.signal < 0],
                    color="crimson",  s=15, label="Short signal")
    axes[2].axvspan(300, 380, alpha=0.1, color="orange")
    axes[2].set_ylabel("Position size"); axes[2].set_xlabel("Day")
    axes[2].set_title("Signals: size ∝ (δ_t − ε)+")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("demos/output_17_geodesic_macro_regression.png", dpi=120)
    plt.close()
    print("[17] Geodesic Macro Regression — saved demos/output_17_geodesic_macro_regression.png")
    print(f"     B_hat: VIX→μ={res.B_hat[0,0]:.4f}, VIX→σ={res.B_hat[1,0]:.4f}")
    print(f"     Trade signals: {(res.signal != 0).sum()} days")


if __name__ == "__main__":
    demo()
