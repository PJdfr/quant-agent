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

Extended — JS inefficiency map (state-space view):
    I(x) = JS̃(A(x)) — spatial map of divergence across discretised state space.
    Directionality per region A:
        Δμ(A)   = E[Y_{t+h} | X_t ∈ A] − E[Y_{t+h}]
        Δq_α(A) = q_α(Y_{t+h} | X_t ∈ A) − q_α(Y_{t+h})
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class EdgePersistenceResult:
    times:           np.ndarray             # time index (rolling window)
    edge_score:      np.ndarray             # ε_s(t) per time step
    confusion:       np.ndarray             # (T, S, S) across-state confusion matrix
    alarm_decay:     np.ndarray             # bool: edge score fell below threshold
    alarm_confusion: np.ndarray             # bool: states becoming indistinguishable
    state_labels:    np.ndarray             # (T,) active state at each time
    baseline_type:   str
    tau_edge:        float
    tau_confusion:   float
    # directionality — populated when outcome is provided to run()
    direction_mu:    np.ndarray | None = None   # (T,) Δμ for current state at t
    direction_tail:  np.ndarray | None = None   # (T,) Δq_0.05 for current state


@dataclass
class InefficencyMapResult:
    state_edges: np.ndarray   # (J+1,) quantile bin edges for state X
    js_map:      np.ndarray   # (J,)   JS̃(A_j) — divergence per state bin
    delta_mu:    np.ndarray   # (J,)   Δμ(A_j) = E[Y|X∈A_j] − E[Y]
    delta_tail:  np.ndarray   # (J,)   Δq_α(A_j) — conditional tail shift
    alpha:       float        # quantile level used for tail shift


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """D_JS(p‖q) = 0.5 KL(p‖m) + 0.5 KL(q‖m),  m = 0.5(p+q)."""
    p = np.maximum(p, 1e-12); p = p / p.sum()
    q = np.maximum(q, 1e-12); q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def _density_from_returns(chunk: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(chunk, bins=edges)
    p = counts.astype(float) + 1e-3
    return p / p.sum()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(returns:       np.ndarray,
        state_labels:  np.ndarray,
        window:        int   = 60,
        n_bins:        int   = 30,
        tau_edge:      float = 0.05,
        tau_confusion: float = 0.08,
        baseline:      str   = "unconditional",
        outcome:       np.ndarray | None = None,
        tail_alpha:    float = 0.05) -> EdgePersistenceResult:
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
    outcome       : (T,) optional future outcome series for directionality
    tail_alpha    : quantile level for Δq_α directionality output
    """
    T = len(returns)
    states = np.unique(state_labels)
    S = len(states)

    lo = np.percentile(returns, 1)
    hi = np.percentile(returns, 99)
    edges = np.linspace(lo, hi, n_bins + 1)

    edge_scores  = np.full(T, np.nan)
    confusions   = np.full((T, S, S), np.nan)
    alarm_decay  = np.zeros(T, dtype=bool)
    alarm_conf   = np.zeros(T, dtype=bool)
    dir_mu       = np.full(T, np.nan) if outcome is not None else None
    dir_tail     = np.full(T, np.nan) if outcome is not None else None

    global_mu    = outcome.mean() if outcome is not None else 0.0
    global_q     = np.quantile(outcome, tail_alpha) if outcome is not None else 0.0

    for t in range(window, T):
        state_t      = state_labels[t]
        chunk        = returns[t - window: t]
        labels_chunk = state_labels[t - window: t]

        # baseline density
        if baseline == "unconditional":
            p_base = _density_from_returns(chunk, edges)
        else:
            mask0  = labels_chunk == 0
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
                c  = np.sqrt(_js_divergence(pi, pj))
                confusions[t, i, j] = confusions[t, j, i] = c

        min_conf = np.nanmin(confusions[t][confusions[t] > 0]) if np.any(confusions[t] > 0) else np.nan
        alarm_conf[t] = (not np.isnan(min_conf)) and (min_conf < tau_confusion)

        # directionality: Δμ and Δq_α for current state
        if outcome is not None and mask_s.any():
            out_chunk = outcome[t - window: t]
            cond_out  = out_chunk[mask_s]
            dir_mu[t]   = cond_out.mean() - global_mu
            dir_tail[t] = np.quantile(cond_out, tail_alpha) - global_q

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
        direction_mu=dir_mu,
        direction_tail=dir_tail,
    )


# ---------------------------------------------------------------------------
# JS inefficiency map (spatial, continuous state variable)
# ---------------------------------------------------------------------------

def inefficiency_map(state:    np.ndarray,
                     outcome:  np.ndarray,
                     horizon:  int   = 1,
                     n_bins:   int   = 10,
                     n_out_bins: int = 30,
                     alpha:    float = 0.05) -> InefficencyMapResult:
    """
    Build a spatial JS divergence map I(x) = JS̃(A(x)) over state space.

    Parameters
    ----------
    state    : (T,) continuous state variable X_t
    outcome  : (T,) future outcome Y_{t+h} (aligned: outcome[t] = Y_{t+h})
    horizon  : h — forward horizon (used for alignment if outcome is raw Y)
    n_bins   : J — number of quantile state bins
    n_out_bins : K — number of outcome histogram bins
    alpha    : quantile level for Δq_α directionality
    """
    X = state[: len(state) - horizon]
    Y = outcome[horizon:]

    s_edges = np.quantile(X, np.linspace(0, 1, n_bins + 1))
    s_edges[0] -= 1e-8; s_edges[-1] += 1e-8

    o_lo, o_hi = np.percentile(Y, 1), np.percentile(Y, 99)
    o_edges = np.linspace(o_lo, o_hi, n_out_bins + 1)

    p_base = _density_from_returns(Y, o_edges)
    global_mu = Y.mean()
    global_q  = np.quantile(Y, alpha)

    js_scores  = np.zeros(n_bins)
    delta_mu   = np.zeros(n_bins)
    delta_tail = np.zeros(n_bins)

    s_idx = np.clip(np.digitize(X, s_edges) - 1, 0, n_bins - 1)

    for j in range(n_bins):
        mask = s_idx == j
        if mask.sum() < 5:
            js_scores[j] = np.nan
            continue
        y_cond = Y[mask]
        p_cond = _density_from_returns(y_cond, o_edges)
        js_scores[j]  = np.sqrt(_js_divergence(p_cond, p_base))
        delta_mu[j]   = y_cond.mean() - global_mu
        delta_tail[j] = np.quantile(y_cond, alpha) - global_q

    return InefficencyMapResult(
        state_edges=s_edges,
        js_map=js_scores,
        delta_mu=delta_mu,
        delta_tail=delta_tail,
        alpha=alpha,
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(18)

    T = 700
    state_seq = np.zeros(T, dtype=int)
    state_seq[100:300] = 1   # momentum regime
    state_seq[450:600] = 2   # stress regime

    returns = np.where(
        state_seq == 0, rng.normal(0.0002, 0.010, T),
        np.where(state_seq == 1,
                 rng.normal(0.0006, 0.009, T),
                 rng.normal(-0.0005, 0.022, T)))

    # simulate edge decay: state 1 looks like baseline after t=220
    returns[220:300] = rng.normal(0.0002, 0.010, 80)

    # outcome = forward 1-day return (for directionality demo)
    outcome = np.roll(returns, -1)
    outcome[-1] = 0.0

    res = run(returns, state_seq, window=60, tau_edge=0.06, tau_confusion=0.10,
              outcome=outcome, tail_alpha=0.05)

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    fig.suptitle("Technique 18 — JS Divergence Edge Persistence", fontweight="bold")

    t = np.arange(T)
    c_map = {0: "steelblue", 1: "seagreen", 2: "crimson"}
    for s, c in c_map.items():
        axes[0].fill_between(t, s - 0.4, s + 0.4, where=state_seq == s,
                             color=c, alpha=0.5, label=f"State {s}")
    axes[0].set_ylabel("State"); axes[0].set_yticks([0, 1, 2])
    axes[0].set_title("State labels"); axes[0].legend(fontsize=8)

    valid = ~np.isnan(res.edge_score)
    axes[1].plot(t[valid], res.edge_score[valid], color="darkorange", lw=1.5, label="ε_s")
    axes[1].axhline(res.tau_edge, color="red", linestyle="--", label=f"τ_edge={res.tau_edge}")
    axes[1].fill_between(t[valid], 0, res.edge_score[valid], where=res.alarm_decay[valid],
                         alpha=0.3, color="red", label="Edge decay alarm")
    axes[1].set_ylabel("ε_s"); axes[1].legend(fontsize=8)
    axes[1].set_title("Edge score √D_JS(p_s, p_base)")

    dm_valid = (res.direction_mu is not None) and (~np.isnan(res.direction_mu)).any()
    if dm_valid:
        axes[2].bar(t, res.direction_mu, color=["crimson" if v < 0 else "seagreen"
                    for v in res.direction_mu], width=1.0, alpha=0.7, label="Δμ(state)")
        axes[2].axhline(0, color="gray", lw=0.5)
        axes[2].set_ylabel("Δμ"); axes[2].legend(fontsize=8)
        axes[2].set_title("Directionality Δμ(A) = E[Y|state] − E[Y]")

    axes[3].plot(returns, color="steelblue", lw=0.6, alpha=0.7)
    axes[3].fill_between(t, -0.05, 0.05, where=res.alarm_decay, alpha=0.4, color="red",
                         label="Edge decay → stop trading")
    axes[3].set_ylabel("Return"); axes[3].set_xlabel("Day"); axes[3].legend(fontsize=8)
    axes[3].set_title("Trade only when edge is geometrically alive")

    plt.tight_layout()
    plt.savefig("demos/output_18_js_edge_persistence.png", dpi=120)
    plt.close()
    print("[18] JS Edge Persistence — saved demos/output_18_js_edge_persistence.png")
    print(f"     Edge decay alarms: {res.alarm_decay.sum()} days")
    print(f"     State confusion alarms: {res.alarm_confusion.sum()} days")

    # also demo the spatial inefficiency map
    vol_state = np.abs(returns)
    map_res = inefficiency_map(vol_state, outcome, horizon=1, n_bins=10)
    print(f"     Inefficiency map — peak JS bin: {np.nanargmax(map_res.js_map)}, "
          f"value: {np.nanmax(map_res.js_map):.4f}")


if __name__ == "__main__":
    demo()
