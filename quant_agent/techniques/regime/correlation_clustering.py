"""
Technique 8 — Correlation Manifold Clustering

Map rolling correlation matrices into the SPD manifold using the
log-Euclidean metric. Cluster in that geometric space to discover
correlation regimes (risk-on, risk-off, sector-coupled, fragmentation).
Avoids artifacts of naive flatten-and-kmeans on raw correlations.
"""

import numpy as np
from dataclasses import dataclass
from sklearn.cluster import KMeans


@dataclass
class CorrelationClusterResult:
    times: np.ndarray
    labels: np.ndarray              # regime label ∈ {0,...,K-1}
    centroids_log: np.ndarray       # cluster centroids in log space
    centroids_corr: np.ndarray      # centroids back in correlation space
    embeddings: np.ndarray          # (T_windows, d) log-Euclidean coords
    inertia: float
    K: int


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def _matrix_log(C: np.ndarray) -> np.ndarray:
    """Matrix logarithm of SPD matrix via eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.maximum(eigvals, 1e-8)
    return eigvecs @ np.diag(np.log(eigvals)) @ eigvecs.T


def _matrix_exp(L: np.ndarray) -> np.ndarray:
    """Matrix exponential."""
    eigvals, eigvecs = np.linalg.eigh(L)
    return eigvecs @ np.diag(np.exp(eigvals)) @ eigvecs.T


def _log_euclidean_distance(Ca: np.ndarray, Cb: np.ndarray) -> float:
    """d(Ca, Cb) = ‖log Ca − log Cb‖_F"""
    return float(np.linalg.norm(_matrix_log(Ca) - _matrix_log(Cb), "fro"))


def _vech(L: np.ndarray) -> np.ndarray:
    """Lower-triangular vectorisation of symmetric matrix."""
    n = L.shape[0]
    idx = np.tril_indices(n)
    return L[idx]


def run(returns: np.ndarray,
        window: int = 60,
        step: int = 5,
        K: int = 3,
        random_state: int = 0) -> CorrelationClusterResult:
    """
    Parameters
    ----------
    returns : (T, n) daily returns
    window  : rolling correlation window
    step    : stride between windows
    K       : number of clusters (regimes)
    """
    T, n = returns.shape
    corr_matrices, times = [], []

    for t in range(window, T, step):
        chunk = returns[t - window: t]
        C = np.corrcoef(chunk.T)
        # regularise: push eigenvalues slightly above 0
        C = C + 1e-4 * np.eye(n)
        d = np.sqrt(np.diag(C))
        C = C / np.outer(d, d)
        corr_matrices.append(C)
        times.append(t)

    # embed in log-Euclidean space
    log_mats = [_matrix_log(C) for C in corr_matrices]
    embeddings = np.array([_vech(L) for L in log_mats])   # (W, n*(n+1)/2)

    km = KMeans(n_clusters=K, random_state=random_state, n_init=10)
    labels = km.fit_predict(embeddings)

    # recover centroid correlation matrices
    centroids_log = []
    centroids_corr = []
    for k in range(K):
        mask = labels == k
        if mask.any():
            mean_log = embeddings[mask].mean(axis=0)
        else:
            mean_log = embeddings.mean(axis=0)
        # reconstruct symmetric matrix from vech
        L_full = np.zeros((n, n))
        idx = np.tril_indices(n)
        L_full[idx] = mean_log
        L_full = L_full + L_full.T - np.diag(np.diag(L_full))
        centroids_log.append(L_full)
        centroids_corr.append(_matrix_exp(L_full))

    return CorrelationClusterResult(
        times=np.array(times),
        labels=labels,
        centroids_log=np.array(centroids_log),
        centroids_corr=np.array(centroids_corr),
        embeddings=embeddings,
        inertia=km.inertia_,
        K=K,
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(22)

    T, n = 700, 8
    names = [f"A{i}" for i in range(n)]

    def corr_block(T, rho_within, rho_cross):
        """n assets with block correlation structure."""
        cov = np.full((n, n), rho_cross)
        cov[:n//2, :n//2] = rho_within
        cov[n//2:, n//2:] = rho_within
        np.fill_diagonal(cov, 1.0)
        # Cholesky
        try:
            L = np.linalg.cholesky(cov + 1e-3 * np.eye(n))
        except np.linalg.LinAlgError:
            L = np.eye(n)
        return rng.normal(0, 0.01, (T, n)) @ L.T

    r1 = corr_block(250, rho_within=0.7, rho_cross=0.1)   # regime: sector-coupled
    r2 = corr_block(200, rho_within=0.9, rho_cross=0.8)   # regime: risk-off (all corr)
    r3 = corr_block(250, rho_within=0.2, rho_cross=-0.1)  # regime: fragmentation
    returns = np.vstack([r1, r2, r3])

    res = run(returns, window=60, step=5, K=3)

    colors = ["steelblue", "crimson", "seagreen"]
    regime_names = ["Sector-coupled", "Risk-off", "Fragmentation"]
    # heuristic: assign names by intra-cluster correlation level
    c_levels = [res.centroids_corr[k][np.triu_indices(n, 1)].mean() for k in range(res.K)]
    order = np.argsort(c_levels)[::-1]
    label_map = {order[0]: "Risk-off", order[1]: "Sector-coupled", order[2]: "Fragmentation"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Technique 8 — Correlation Manifold Clustering", fontweight="bold")

    # timeline
    ax = axes[0, 0]
    for k in range(res.K):
        mask = res.labels == k
        ax.scatter(res.times[mask],
                   [k] * mask.sum(),
                   c=colors[k], s=20, label=f"Regime {k}: {label_map.get(k,'')}")
    ax.axvline(250, color="gray", linestyle=":", label="True shifts")
    ax.axvline(450, color="gray", linestyle=":")
    ax.set_title("Correlation regime labels over time")
    ax.set_xlabel("Day"); ax.set_yticks([0,1,2]); ax.legend(fontsize=7)

    # 2D PCA of embeddings
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    emb2d = pca.fit_transform(res.embeddings)

    ax = axes[0, 1]
    for k in range(res.K):
        mask = res.labels == k
        ax.scatter(emb2d[mask, 0], emb2d[mask, 1], c=colors[k], s=20, alpha=0.7,
                   label=f"Regime {k}")
    ax.set_title("Log-Euclidean embedding (PCA 2D)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(fontsize=8)

    # centroid correlation matrices
    for ki, ax in enumerate([axes[1, 0], axes[1, 1]]):
        if ki >= res.K:
            ax.axis("off")
            continue
        im = ax.imshow(res.centroids_corr[ki], vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_title(f"Centroid — Regime {ki}: {label_map.get(ki,'')}")
        ax.set_xticks(range(n)); ax.set_xticklabels(names, fontsize=7)
        ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=7)
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig("demos/output_08_correlation_clustering.png", dpi=120)
    plt.close()
    print("[8] Correlation Clustering — saved demos/output_08_correlation_clustering.png")
    print(f"    Regimes found: {res.K}, inertia = {res.inertia:.2f}")


if __name__ == "__main__":
    demo()
