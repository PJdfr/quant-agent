"""
Technique 10 — Option-Implied Density Manifold

Extract the risk-neutral density from the implied-vol surface via
Breeden-Litzenberger. Each day's density is a point on a density manifold.
Fisher-Rao distance measures shape changes in skew/tails. Compare to a
historical library to detect regime shifts. Use as trigger + risk throttle.
"""

import numpy as np
from dataclasses import dataclass
from scipy.stats import norm


@dataclass
class OptionDensityResult:
    times: np.ndarray
    fr_distance: np.ndarray        # FR distance from nearest historical density
    nearest_idx: np.ndarray        # index into history library
    alarm: np.ndarray              # bool
    risk_multiplier: np.ndarray
    densities: np.ndarray          # (T, n_strikes) risk-neutral densities
    strikes: np.ndarray
    tau: float


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def _implied_vol_surface(F: float, r: float, T_exp: float,
                          strikes: np.ndarray,
                          atm_vol: float,
                          skew: float,
                          kurt_adj: float) -> np.ndarray:
    """
    Parametric implied-vol smile: vol(K) = atm_vol + skew*(log(K/F)) + kurt*(log(K/F))²
    Returns array of implied vols for each strike.
    """
    x = np.log(strikes / F)
    return atm_vol + skew * x + kurt_adj * x ** 2


def _breeden_litzenberger(strikes: np.ndarray,
                           ivols: np.ndarray,
                           F: float, r: float, T_exp: float) -> np.ndarray:
    """
    Approximate second derivative of call price w.r.t. K → risk-neutral density.
    Uses finite differences on the Black-Scholes price grid.
    """
    from scipy.stats import norm as _norm

    def bs_call(K, iv):
        if iv <= 0:
            return max(F - K, 0)
        d1 = (np.log(F / K) + 0.5 * iv ** 2 * T_exp) / (iv * np.sqrt(T_exp))
        d2 = d1 - iv * np.sqrt(T_exp)
        return np.exp(-r * T_exp) * (F * _norm.cdf(d1) - K * _norm.cdf(d2))

    prices = np.array([bs_call(K, iv) for K, iv in zip(strikes, ivols)])

    # d²C/dK² via central finite differences
    dK = strikes[1] - strikes[0]
    q = np.zeros(len(strikes))
    q[1:-1] = np.exp(r * T_exp) * (prices[2:] - 2 * prices[1:-1] + prices[:-2]) / dK ** 2
    q[0] = q[1]; q[-1] = q[-2]

    q = np.maximum(q, 1e-10)
    q /= (q * dK).sum()         # normalise
    return q


def _fr_distance_densities(p: np.ndarray, q: np.ndarray, dK: float) -> float:
    """Fisher-Rao distance: 2 arccos(∫ √p √q dK)."""
    p = np.maximum(p, 1e-12); q = np.maximum(q, 1e-12)
    p = p / (p * dK).sum(); q = q / (q * dK).sum()
    inner = (np.sqrt(p * q) * dK).sum()
    inner = np.clip(inner, -1.0, 1.0)
    return 2 * np.arccos(inner)


def run(vol_params: np.ndarray,
        F: float = 100.0,
        r: float = 0.05,
        T_exp: float = 1/12,
        n_strikes: int = 50,
        history_size: int = 30,
        tau: float = 0.15,
        c: float = 8.0) -> OptionDensityResult:
    """
    Parameters
    ----------
    vol_params : (T, 3) — [atm_vol, skew, kurt_adj] per day
    F, r, T_exp: forward, rate, expiry
    n_strikes  : number of strike grid points
    history_size: library size (rolling)
    tau        : alarm threshold (FR distance)
    c          : throttle steepness
    """
    T = vol_params.shape[0]
    strikes = np.linspace(F * 0.7, F * 1.3, n_strikes)
    dK = strikes[1] - strikes[0]

    # compute all densities
    densities = []
    for t in range(T):
        atm, skew, kurt = vol_params[t]
        ivols = _implied_vol_surface(F, r, T_exp, strikes, atm, skew, kurt)
        ivols = np.maximum(ivols, 0.01)
        q = _breeden_litzenberger(strikes, ivols, F, r, T_exp)
        densities.append(q)
    densities = np.array(densities)

    # compare each day to history
    fr_dists, nearest_idxs = [], []
    for t in range(history_size, T):
        lib = densities[t - history_size: t]
        today = densities[t]
        dists = [_fr_distance_densities(today, lib[i], dK)
                 for i in range(history_size)]
        fr_dists.append(min(dists))
        nearest_idxs.append(np.argmin(dists) + t - history_size)

    fr_dists = np.array(fr_dists)
    alarm = fr_dists > tau
    risk_mult = 1.0 / (1.0 + c * fr_dists)

    return OptionDensityResult(
        times=np.arange(history_size, T),
        fr_distance=fr_dists,
        nearest_idx=np.array(nearest_idxs),
        alarm=alarm,
        risk_multiplier=risk_mult,
        densities=densities,
        strikes=strikes,
        tau=tau,
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(44)

    T = 300
    # Regime 1: low vol, mild negative skew
    atm1   = rng.normal(0.18, 0.005, 150)
    skew1  = rng.normal(-0.02, 0.003, 150)
    kurt1  = rng.normal(0.005, 0.001, 150)
    # Regime 2: vol spike, steep negative skew (crash pricing)
    atm2   = rng.normal(0.35, 0.010, 150)
    skew2  = rng.normal(-0.10, 0.005, 150)
    kurt2  = rng.normal(0.020, 0.003, 150)

    vol_params = np.column_stack([
        np.concatenate([atm1, atm2]),
        np.concatenate([skew1, skew2]),
        np.concatenate([kurt1, kurt2]),
    ])

    res = run(vol_params, tau=0.20, history_size=30)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    fig.suptitle("Technique 10 — Option-Implied Density Manifold", fontweight="bold")

    # top: FR distance over time
    t = res.times
    axes[0].plot(t, res.fr_distance, color="crimson", lw=1.5,
                 label="FR distance to nearest historical density")
    axes[0].axhline(res.tau, color="red", linestyle="--", label=f"τ = {res.tau}")
    axes[0].fill_between(t, 0, res.fr_distance, where=res.alarm,
                         alpha=0.3, color="red", label="Regime alarm")
    axes[0].axvline(150, color="orange", linestyle=":", lw=2, label="True regime shift")
    axes[0].set_ylabel("FR distance"); axes[0].set_xlabel("Day")
    axes[0].set_title("Fisher-Rao distance to historical density library — spike = tail/skew regime shift")
    axes[0].legend(fontsize=8)

    # middle: risk throttle
    axes[1].plot(t, res.risk_multiplier, color="seagreen", lw=1.5)
    axes[1].axhline(1.0, color="gray", linestyle="--")
    axes[1].fill_between(t, res.risk_multiplier, 1.0, alpha=0.3, color="red")
    axes[1].axvline(150, color="orange", linestyle=":", lw=2)
    axes[1].set_ylabel("Risk multiplier"); axes[1].set_xlabel("Day")
    axes[1].set_title("Vol risk throttle  1/(1+c·D_FR)")

    # bottom: density comparison at two dates
    ax = axes[2]
    day_calm   = 100
    day_stress = 200
    ax.plot(res.strikes, res.densities[day_calm],   color="steelblue",
            lw=2, label=f"Day {day_calm} — calm (low vol, mild skew)")
    ax.plot(res.strikes, res.densities[day_stress], color="crimson",
            lw=2, label=f"Day {day_stress} — stress (high vol, steep skew)")
    ax.set_xlabel("Strike"); ax.set_ylabel("Risk-neutral density")
    ax.set_title("Risk-neutral densities: shape shifts dramatically in stress regime")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("demos/output_10_option_density.png", dpi=120)
    plt.close()
    print("[10] Option Density — saved demos/output_10_option_density.png")
    print(f"     Alarm days: {res.alarm.sum()} / {len(res.alarm)}")
    print(f"     Peak FR distance: {res.fr_distance.max():.4f}")


if __name__ == "__main__":
    demo()
