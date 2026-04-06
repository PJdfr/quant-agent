"""
Technique 23 — Mutual Information Scanner

I(X;Y) = H(Y) − H(Y|X) measures how much knowing the current state X
reduces uncertainty about the future outcome Y, without assuming linearity.

Bin-based (plug-in) estimator:

    p̂_jk = (1/N) Σ_t 1_{X_t ∈ A_j, Y_{t+h} ∈ B_k}
    Î(X;Y) = Σ_j Σ_k p̂_jk log( p̂_jk / (p̂_j p̂_k) )

Local inefficiency intensity per state region A_j:

    ℓ̂(A_j) = Σ_k p̂(k|j) log( p̂(k|j) / p̂_k )

This gives a spatial map of predictive edge across state space: which states
carry information about the future, and how much.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class MutualInformationResult:
    mi:            float        # Î(X;Y) — global mutual information (nats)
    local_mi:      np.ndarray   # (J,) ℓ̂(A_j) local MI per state bin
    state_edges:   np.ndarray   # (J+1,) bin edges for state X
    outcome_edges: np.ndarray   # (K+1,) bin edges for outcome Y
    joint_hist:    np.ndarray   # (J, K) normalised joint distribution p̂_jk
    rolling_mi:    np.ndarray   # (T,) rolling MI (NaN before first window)
    horizon:       int


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def _mi_from_joint(p_jk: np.ndarray,
                   eps: float = 1e-12) -> tuple[float, np.ndarray]:
    """
    Global MI and local MI per state bin from a raw count or probability matrix.
    Returns (mi_scalar, local_mi array of shape J).
    """
    p = p_jk / (p_jk.sum() + eps)
    p_j = p.sum(axis=1, keepdims=True)   # (J, 1)
    p_k = p.sum(axis=0, keepdims=True)   # (1, K)

    # global MI: Σ p_jk log(p_jk / p_j p_k)
    denom   = p_j * p_k + eps
    ratio   = np.where(p > eps, p / denom, 1.0)
    mi      = float(np.sum(p * np.log(np.maximum(ratio, eps))))

    # local MI per state bin: ℓ̂(A_j) = Σ_k p(k|j) log(p(k|j)/p_k)
    p_cond  = p / (p_j + eps)            # (J, K) conditional p(k|j)
    r_local = np.where(p_cond > eps, p_cond / (p_k + eps), 1.0)
    local   = np.sum(p_cond * np.log(np.maximum(r_local, eps)), axis=1)

    return mi, local


def _build_joint(X: np.ndarray, Y: np.ndarray,
                 s_edges: np.ndarray, o_edges: np.ndarray,
                 J: int, K: int) -> np.ndarray:
    s_idx = np.clip(np.digitize(X, s_edges) - 1, 0, J - 1)
    o_idx = np.clip(np.digitize(Y, o_edges) - 1, 0, K - 1)
    p = np.zeros((J, K))
    for j, k in zip(s_idx, o_idx):
        p[j, k] += 1.0
    return p


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(state:         np.ndarray,
        outcome:       np.ndarray,
        horizon:       int = 1,
        n_state_bins:  int = 10,
        n_out_bins:    int = 20,
        window:        int = 120,
        step:          int = 10) -> MutualInformationResult:
    """
    Parameters
    ----------
    state        : (T,) or (T, d) state variable X_t
                   Multi-dimensional state is projected to 1D via first PC.
    outcome      : (T,) future outcome series (already aligned, or pass raw
                   and use horizon to shift internally)
    horizon      : h — forward horizon; outcome[t+h] is paired with state[t]
    n_state_bins : J — number of quantile state bins
    n_out_bins   : K — number of quantile outcome bins
    window       : rolling window length for MI estimation
    step         : stride between successive rolling estimates
    """
    # project multi-dimensional state to scalar via first PC
    if state.ndim > 1:
        centered = state - state.mean(axis=0)
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        state = centered @ Vt[0]

    T = len(state)
    X = state[: T - horizon]
    Y = outcome[horizon:]
    N = len(X)

    # global quantile bins on full sample
    s_edges = np.quantile(X, np.linspace(0, 1, n_state_bins + 1))
    s_edges[0] -= 1e-8; s_edges[-1] += 1e-8
    o_edges = np.quantile(Y, np.linspace(0, 1, n_out_bins + 1))
    o_edges[0] -= 1e-8; o_edges[-1] += 1e-8

    p_jk_global = _build_joint(X, Y, s_edges, o_edges, n_state_bins, n_out_bins)
    global_mi, local_mi = _mi_from_joint(p_jk_global)

    # rolling MI
    rolling = np.full(T, np.nan)
    for t in range(window, N, step):
        chunk_X = X[t - window: t]
        chunk_Y = Y[t - window: t]

        se = np.quantile(chunk_X, np.linspace(0, 1, n_state_bins + 1))
        se[0] -= 1e-8; se[-1] += 1e-8
        oe = np.quantile(chunk_Y, np.linspace(0, 1, n_out_bins + 1))
        oe[0] -= 1e-8; oe[-1] += 1e-8

        p = _build_joint(chunk_X, chunk_Y, se, oe, n_state_bins, n_out_bins)
        mi_t, _ = _mi_from_joint(p)
        rolling[t + horizon] = mi_t

    return MutualInformationResult(
        mi=global_mi,
        local_mi=local_mi,
        state_edges=s_edges,
        outcome_edges=o_edges,
        joint_hist=p_jk_global / (p_jk_global.sum() + 1e-12),
        rolling_mi=rolling,
        horizon=horizon,
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(23)

    T = 800
    # state: lagged volatility (AR(1) process)
    raw = rng.normal(0, 1, T)
    vol = np.zeros(T)
    vol[0] = abs(raw[0])
    for i in range(1, T):
        vol[i] = 0.9 * vol[i - 1] + 0.1 * abs(raw[i])

    # outcome: return with vol-state dependent signal (MI detectable 0–500)
    noise   = rng.normal(0, 1, T)
    outcome = np.zeros(T)
    outcome[:500] = -0.3 * np.sign(vol[:500] - vol[:500].mean()) * vol[:500] + 0.8 * noise[:500]
    outcome[500:] = rng.normal(0, 0.01, T - 500)   # signal breaks at t=500

    res = run(vol, outcome, horizon=1, window=120, step=5)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle("Technique 23 — Mutual Information Scanner", fontweight="bold")

    t = np.arange(T)
    valid = ~np.isnan(res.rolling_mi)
    axes[0].plot(t[valid], res.rolling_mi[valid], color="darkorange", lw=1.5, label="Rolling Î(X;Y)")
    axes[0].axvspan(500, T, alpha=0.12, color="red", label="Signal broken")
    axes[0].set_ylabel("nats"); axes[0].legend(fontsize=8)
    axes[0].set_title("Rolling mutual information — measures state→outcome predictability")

    axes[1].bar(range(len(res.local_mi)), res.local_mi, color="steelblue", width=0.8)
    axes[1].set_xlabel("State bin j"); axes[1].set_ylabel("ℓ̂(A_j)")
    axes[1].set_title("Local MI per state region — where is the predictive edge concentrated?")

    im = axes[2].imshow(res.joint_hist.T, origin="lower", aspect="auto",
                        cmap="Blues",
                        extent=[0, res.joint_hist.shape[0], 0, res.joint_hist.shape[1]])
    axes[2].set_xlabel("State bin j"); axes[2].set_ylabel("Outcome bin k")
    axes[2].set_title("Joint distribution p̂(X,Y) — deviation from independence = edge")
    plt.colorbar(im, ax=axes[2])

    plt.tight_layout()
    plt.savefig("demos/output_23_mutual_information.png", dpi=120)
    plt.close()
    print("[23] Mutual Information — saved demos/output_23_mutual_information.png")
    print(f"     Global MI: {res.mi:.5f} nats")
    print(f"     Peak local MI: bin {res.local_mi.argmax()} = {res.local_mi.max():.5f}")


if __name__ == "__main__":
    demo()
