"""
Technique 12 — Partial Correlations / Conditional Dependence Network

Partial correlation ρ_{ij|rest} = -Θ_ij / √(Θ_ii Θ_jj) where Θ = Σ⁻¹.
Captures conditional dependence: two assets are directly linked only if their
partial correlation is non-zero (after removing all other assets' influence).
Build a sparse network by thresholding: edges = direct conditional links.
Use for network hedges, clustering, and relative-value overlays.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class PartialCorrResult:
    partial_corr: np.ndarray        # (n, n) partial correlation matrix
    adjacency: np.ndarray           # (n, n) binary adjacency (thresholded)
    edge_weights: np.ndarray        # (n, n) signed edge weights
    degree: np.ndarray              # (n,) node degree (connectivity count)
    connectedness: np.ndarray       # (n,) c(i) = Σ |W_ij| (weighted degree)
    theta: np.ndarray               # precision matrix used
    tau: float                      # threshold used
    asset_names: list


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def _precision_from_cov(S: np.ndarray, lam: float = 0.01) -> np.ndarray:
    """Ridge-regularised precision matrix."""
    n = S.shape[0]
    return np.linalg.inv(S + lam * np.eye(n))


def _partial_corr_from_precision(Theta: np.ndarray) -> np.ndarray:
    """ρ_{ij|rest} = -Θ_ij / √(Θ_ii * Θ_jj)"""
    n = Theta.shape[0]
    P = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                P[i, j] = 1.0
            else:
                denom = np.sqrt(Theta[i, i] * Theta[j, j])
                P[i, j] = -Theta[i, j] / max(denom, 1e-10)
    return P


def run(returns: np.ndarray,
        asset_names: list,
        tau: float = 0.1,
        lam: float = 0.01) -> PartialCorrResult:
    """
    Parameters
    ----------
    returns      : (T, n) daily returns
    asset_names  : list of n strings
    tau          : edge threshold — only show partial correlations above this
    lam          : ridge regularisation for precision

    Returns
    -------
    PartialCorrResult
    """
    S = np.cov(returns.T, ddof=1)
    Theta = _precision_from_cov(S, lam)
    P = _partial_corr_from_precision(Theta)

    n = len(asset_names)
    adj = (np.abs(P) >= tau).astype(float)
    np.fill_diagonal(adj, 0)

    W = P * adj   # signed edge weights
    degree = adj.sum(axis=1)
    conn = np.abs(W).sum(axis=1)

    return PartialCorrResult(
        partial_corr=P,
        adjacency=adj,
        edge_weights=W,
        degree=degree,
        connectedness=conn,
        theta=Theta,
        tau=tau,
        asset_names=asset_names,
    )


def network_hedge(res: PartialCorrResult, asset_idx: int) -> dict:
    """
    Build a network hedge basket for asset i.
    Neighbours are assets with direct conditional links.
    Hedge weights proportional to -W_ij (oppose the direct link).
    """
    W = res.edge_weights[asset_idx]
    neighbours = np.where(res.adjacency[asset_idx] > 0)[0]
    if len(neighbours) == 0:
        return {}
    raw = {res.asset_names[j]: -W[j] for j in neighbours}
    total = sum(abs(v) for v in raw.values())
    return {k: v / total for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(202)

    T, n = 400, 10
    names = ["SPY","QQQ","IWM","TLT","GLD","XLE","XLF","XLK","HYG","VIX-inv"]

    # true structure: equity cluster, rates, commodities
    eq_factor  = rng.normal(0, 1, T)
    rate_factor = rng.normal(0, 1, T)
    commod     = rng.normal(0, 1, T)

    returns = np.column_stack([
        eq_factor + rng.normal(0, 0.3, T),          # SPY
        eq_factor + rng.normal(0, 0.4, T),          # QQQ
        eq_factor + rng.normal(0, 0.5, T),          # IWM
        -0.6*eq_factor + rate_factor + rng.normal(0, 0.3, T),  # TLT
        0.2*eq_factor + commod + rng.normal(0, 0.3, T),        # GLD
        0.4*eq_factor + commod + rng.normal(0, 0.4, T),        # XLE
        eq_factor + rate_factor*0.3 + rng.normal(0, 0.3, T),  # XLF
        eq_factor + rng.normal(0, 0.2, T),          # XLK
        eq_factor*0.7 + rate_factor*0.3 + rng.normal(0, 0.4, T),  # HYG
        -eq_factor + rng.normal(0, 0.3, T),         # VIX-inv
    ])

    res = run(returns, names, tau=0.08, lam=0.02)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Technique 12 — Partial Correlations / Conditional Dependence Network",
                 fontweight="bold")

    kw = dict(vmin=-1, vmax=1, cmap="RdBu_r")

    ax = axes[0]
    linear_corr = np.corrcoef(returns.T)
    im = ax.imshow(linear_corr, **kw)
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=45, fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=7)
    ax.set_title("Linear correlation\n(includes indirect links)")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1]
    im2 = ax.imshow(res.partial_corr, **kw)
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=45, fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=7)
    ax.set_title(f"Partial correlation (τ={res.tau})\n(direct conditional links only)")
    plt.colorbar(im2, ax=ax, fraction=0.046)

    # network graph
    ax = axes[2]
    try:
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i+1, n):
                if res.adjacency[i, j]:
                    G.add_edge(i, j, weight=res.edge_weights[i, j])

        pos = nx.spring_layout(G, seed=42)
        node_labels = {i: names[i] for i in range(n)}
        edge_colors = ["red" if G[u][v]["weight"] < 0 else "steelblue"
                       for u, v in G.edges()]
        edge_widths = [abs(G[u][v]["weight"]) * 5 for u, v in G.edges()]
        nx.draw_networkx(G, pos=pos, ax=ax, labels=node_labels,
                         node_color="lightyellow", node_size=800,
                         edge_color=edge_colors, width=edge_widths,
                         font_size=7)
        ax.set_title("Conditional dependence network\nblue=positive, red=negative link")
    except ImportError:
        ax.imshow(res.adjacency, cmap="Blues")
        ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=45, fontsize=7)
        ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=7)
        ax.set_title("Adjacency matrix (install networkx for graph viz)")
    ax.axis("off") if ax.get_images() == [] else None

    plt.tight_layout()
    plt.savefig("demos/output_12_partial_correlation.png", dpi=120)
    plt.close()
    print("[12] Partial Correlation — saved demos/output_12_partial_correlation.png")
    print(f"     Edges: {int(res.adjacency.sum()//2)}, most connected: "
          f"{names[res.connectedness.argmax()]} (c={res.connectedness.max():.2f})")
    hedge = network_hedge(res, 0)
    print(f"     Network hedge for SPY: {hedge}")


if __name__ == "__main__":
    demo()
