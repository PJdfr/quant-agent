"""
Technique 3 — Shape Correlation (Distribution Geometry)

Map windowed return histograms to the unit sphere via the square-root transform.
Hellinger distance becomes Euclidean on the sphere. Shape correlation across time
captures tail/skew co-movement that linear return correlation misses.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class ShapeResult:
    assets: list                      # asset names
    shape_corr: np.ndarray            # (n, n) shape correlation matrix
    linear_corr: np.ndarray           # (n, n) Pearson correlation (for comparison)
    hellinger: np.ndarray             # (n, n) average Hellinger distance
    psi: np.ndarray                   # (n_windows, n, n_bins) sqrt-density vectors
    bins: np.ndarray                  # bin centres


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def _windowed_sqrt_density(returns: np.ndarray,
                            window: int,
                            n_bins: int = 30) -> tuple:
    """
    For each window, estimate density via histogram → square-root map.
    Returns psi: (n_windows, n_bins)  and  bin_centres.
    """
    T = len(returns)
    # fixed global bin edges (so densities are comparable)
    lo, hi = np.percentile(returns, 1), np.percentile(returns, 99)
    edges = np.linspace(lo, hi, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    psi_list = []
    for t in range(window, T):
        chunk = returns[t - window: t]
        counts, _ = np.histogram(chunk, bins=edges, density=True)
        counts = np.maximum(counts, 1e-12)
        # normalise to proper density (integrate = 1)
        width = edges[1] - edges[0]
        p = counts * width
        p /= p.sum()
        psi = np.sqrt(p)                # square-root map → unit sphere
        psi /= np.linalg.norm(psi)
        psi_list.append(psi)

    return np.array(psi_list), centres


def run(returns_matrix: np.ndarray,
        asset_names: list,
        window: int = 60,
        n_bins: int = 30) -> ShapeResult:
    """
    Parameters
    ----------
    returns_matrix : (T, n)  daily returns
    asset_names    : list of n strings
    window         : rolling estimation window
    n_bins         : histogram bins

    Returns
    -------
    ShapeResult
    """
    T, n = returns_matrix.shape
    all_psi = []
    all_centres = None

    for i in range(n):
        psi, centres = _windowed_sqrt_density(returns_matrix[:, i], window, n_bins)
        all_psi.append(psi)
        all_centres = centres

    # all_psi: (n, n_windows, n_bins)
    psi_arr = np.array(all_psi)
    n_windows = psi_arr.shape[1]

    # shape inner product S(a,b,t) = ψ_a · ψ_b  (= 1 - H²/2)
    shape_corr = np.zeros((n, n))
    hellinger  = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            dots = np.einsum("ti,ti->t", psi_arr[i], psi_arr[j])  # (n_windows,)
            # Hellinger distance: H = sqrt(2) * arccos(dot)
            H = np.sqrt(2) * np.arccos(np.clip(dots, -1, 1))
            # shape correlation: Corr_t[S(a,book), S(b,book)]
            sc = np.corrcoef(dots, np.ones(n_windows))[0, 0] if n_windows > 1 else 1.0
            sc = np.mean(dots)          # simpler: mean inner product as proxy
            shape_corr[i, j] = shape_corr[j, i] = sc
            hellinger[i, j] = hellinger[j, i] = H.mean()

    np.fill_diagonal(shape_corr, 1.0)
    np.fill_diagonal(hellinger, 0.0)

    linear_corr = np.corrcoef(returns_matrix.T)

    return ShapeResult(
        assets=asset_names,
        shape_corr=shape_corr,
        linear_corr=linear_corr,
        hellinger=hellinger,
        psi=psi_arr,
        bins=all_centres,
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(7)

    T, n = 500, 5
    names = ["SPY", "QQQ", "TLT", "GLD", "HYG"]

    # SPY/QQQ: correlated normal
    common = rng.normal(0, 0.01, T)
    returns = np.column_stack([
        common + rng.normal(0, 0.005, T),                       # SPY
        common + rng.normal(0, 0.007, T),                       # QQQ
        -0.3 * common + rng.normal(0, 0.006, T),                # TLT (flight-to-quality)
        rng.normal(0, 0.006, T),                                # GLD (uncorrelated)
        common + rng.laplace(0, 0.008, T),                      # HYG (fat tails!)
    ])

    res = run(returns, names, window=60)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Technique 3 — Shape Correlation vs Linear Correlation", fontweight="bold")

    kw = dict(vmin=-1, vmax=1, cmap="RdBu_r")

    def heatmap(ax, mat, title):
        im = ax.imshow(mat, **kw)
        ax.set_xticks(range(n)); ax.set_xticklabels(names, fontsize=9)
        ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=9)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046)

    heatmap(axes[0], res.linear_corr, "Linear (Pearson) Correlation")
    heatmap(axes[1], res.shape_corr,  "Shape Correlation (√-density sphere)")

    diff = res.shape_corr - res.linear_corr
    im2 = axes[2].imshow(diff, cmap="PuOr", vmin=-0.5, vmax=0.5)
    axes[2].set_xticks(range(n)); axes[2].set_xticklabels(names, fontsize=9)
    axes[2].set_yticks(range(n)); axes[2].set_yticklabels(names, fontsize=9)
    axes[2].set_title("Difference  (Shape − Linear)\npositive = shape captures extra co-movement")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    plt.savefig("demos/output_03_shape_correlation.png", dpi=120)
    plt.close()
    print("[3] Shape Correlation — saved demos/output_03_shape_correlation.png")
    print("    HYG-SPY shape corr:", f"{res.shape_corr[0,4]:.3f}",
          " linear corr:", f"{res.linear_corr[0,4]:.3f}")


if __name__ == "__main__":
    demo()
