"""
Technique 18 — JS Divergence Edge Persistence

A signal edge lives in the geometric gap between the state-conditional
distribution p_s and a baseline p_base. Measure that gap with Jensen-Shannon
divergence (symmetric, bounded ∈ [0, log 2]).

    ε_s = sqrt(D_JS(p_s, p_base))

When ε_s shrinks over time, the conditional approaches the baseline →
the edge is decaying geometrically. Also track across-state confusion:

    C_{s,s'} = sqrt(D_JS(p_s, p_s'))

If states become indistinguishable (C small), regime dependence weakens.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class EdgePersistenceResult:
    times:           np.ndarray   # time index (rolling window)
    edge_score:      np.ndarray   # ε_s(t) per time step
    confusion:       np.ndarray   # (T, S, S) across-state confusion matrix
    alarm_decay:     np.ndarray   # bool: edge score fell below threshold
    alarm_confusion: np.ndarray   # bool: states becoming indistinguishable
    state_labels:    np.ndarray   # (T,) active state at each time
    baseline_type:   str
    tau_edge:        float
    tau_confusion:   float


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """D_JS(p‖q) = 0.5 D_KL(p‖m) + 0.5 D_KL(q‖m),  m = 0.5(p+q)"""
    p = np.maximum(p, 1e-12); p = p / p.sum()
    q = np.maximum(q, 1e-12); q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def _density_from_returns(chunk: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(chunk, bins=edges)
    p = counts.astype(float) + 1e-3
    return p / p.sum()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(returns:        np.ndarray,
        state_labels:   np.ndarray,
        window:         int   = 60,
        n_bins:         int   = 30,
        tau_edge:       float = 0.05,
        tau_confusion:  float = 0.08,
        baseline:       str   = "unconditional") -> EdgePersistenceResult:
    """
    Parameters
    ----------
    returns       : (T,) daily returns
    state_labels  : (T,) integer state at each time (0 = baseline/no-edge)
    window        : rolling window for density estimation
    n_bins        : histogram bins
    tau_edge      : alarm when ε_s falls below this (edge decaying)
    tau_confusion : alarm when C_{s,s'} falls below this (states merging)
    baseline      : 'unconditional' or 'state_0'
    """
    T = len(returns)
    states = np.unique(state_labels)
    S = len(states)

    lo = np.percentile(returns, 1)
    hi = np.percentile(returns, 99)
    edges = np.linspace(lo, hi, n_bins + 1)

    edge_scores   = np.full(T, np.nan)
    confusions    = np.full((T, S, S), np.nan)
    alarm_decay   = np.zeros(T, dtype=bool)
    alarm_conf    = np.zeros(T, dtype=bool)

    for t in range(window, T):
        state_t = state_labels[t]
        chunk   = returns[t - window: t]
        labels_chunk = state_labels[t - window: t]

        # baseline density
        if baseline == "unconditional":
            p_base = _density_from_returns(chunk, edges)
        else:
            mask0 = labels_chunk == 0
            p_base = (_density_from_returns(chunk[mask0], edges)
                      if mask0.any() else _density_from_returns(chunk, edges))

        # conditional density for current state
        mask_s = labels_chunk == state_t
        if mask_s.sum() < 5:
            continue
        p_s = _density_from_returns(chunk[mask_s], edges)

        eps_s = np.sqrt(_js_divergence(p_s, p_base))
        edge_scores[t] = eps_s
        alarm_decay[t] = eps_s < tau_edge

        # across-state confusion matrix
        for i, si in enumerate(states):
            for j, sj in enumerate(states):
                if i >= j:
                    continue
                mi = labels_chunk == si
                mj = labels_chunk == sj
                if mi.sum() < 3 or mj.sum() < 3:
                    continue
                pi = _density_from_returns(chunk[mi], edges)
                pj = _density_from_returns(chunk[mj], edges)
                c = np.sqrt(_js_divergence(pi, pj))
                confusions[t, i, j] = confusions[t, j, i] = c
        min_conf = np.nanmin(confusions[t][confusions[t] > 0]) if np.any(confusions[t] > 0) else np.nan
        alarm_conf[t] = (not np.isnan(min_conf)) and (min_conf < tau_confusion)

    return EdgePersistenceResult(
        times=np.arange(T),
        edge_score=edge_scores,
        confusion=confusions,
        alarm_decay=alarm_decay,
        alarm_confusion=alarm_conf,
        state_labels=state_labels,
        baseline_type=baseline,
        tau_edge=tau_edge,
        tau_confusion=tau_confusion,
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(18)

    T = 700
    # 3 states: 0=normal, 1=momentum (fat right tail), 2=stress (fat left)
    state_seq = np.zeros(T, dtype=int)
    state_seq[100:300] = 1   # momentum regime
    state_seq[450:600] = 2   # stress regime
    # edge decays: state 1 gradually merges with baseline from t=250
    # state 2 stays distinct throughout

    returns = np.where(
        state_seq == 0, rng.normal(0.0002, 0.010, T),
        np.where(state_seq == 1,
                 rng.normal(0.0006, 0.009, T) + np.linspace(0, 0, T),   # momentum
                 rng.normal(-0.0005, 0.022, T)))                         # stress

    # Simulate edge decay: state 1 distribution drifts toward baseline after t=220
    returns[220:300] = rng.normal(0.0002, 0.010, 80)   # state 1 looks like baseline

    res = run(returns, state_seq, window=60, tau_edge=0.06, tau_confusion=0.10)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("Technique 18 — JS Divergence Edge Persistence", fontweight="bold")

    t = np.arange(T)
    c_map = {0: "steelblue", 1: "seagreen", 2: "crimson"}
    for s, c in c_map.items():
        mask = state_seq == s
        axes[0].fill_between(t, s - 0.4, s + 0.4, where=mask,
                             color=c, alpha=0.5, label=f"State {s}")
    axes[0].set_ylabel("State"); axes[0].set_yticks([0, 1, 2])
    axes[0].set_title("State labels over time")
    axes[0].legend(fontsize=8)

    valid = ~np.isnan(res.edge_score)
    axes[1].plot(t[valid], res.edge_score[valid], color="darkorange", lw=1.5,
                 label="ε_s = √D_JS(p_s, p_base)")
    axes[1].axhline(res.tau_edge, color="red", linestyle="--",
                    label=f"τ_edge = {res.tau_edge}")
    axes[1].fill_between(t[valid], 0, res.edge_score[valid],
                         where=res.alarm_decay[valid],
                         alpha=0.3, color="red", label="Edge decay alarm")
    axes[1].axvspan(220, 300, alpha=0.1, color="gray",
                    label="State 1 merging with baseline")
    axes[1].set_ylabel("ε_s"); axes[1].legend(fontsize=8)
    axes[1].set_title("Edge score: distance of state distribution from baseline")

    axes[2].plot(returns, color="steelblue", lw=0.6, alpha=0.7, label="Returns")
    axes[2].fill_between(t, -0.05, 0.05, where=res.alarm_decay,
                         alpha=0.4, color="red", label="Edge decay → stop trading signal")
    axes[2].set_ylabel("Return"); axes[2].set_xlabel("Day")
    axes[2].set_title("Trade only when edge is geometrically alive (ε_s > τ)")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("demos/output_18_js_edge_persistence.png", dpi=120)
    plt.close()
    print("[18] JS Edge Persistence — saved demos/output_18_js_edge_persistence.png")
    decay_days = res.alarm_decay.sum()
    print(f"     Edge decay alarms: {decay_days} days")
    print(f"     State confusion alarms: {res.alarm_confusion.sum()} days")


if __name__ == "__main__":
    demo()
