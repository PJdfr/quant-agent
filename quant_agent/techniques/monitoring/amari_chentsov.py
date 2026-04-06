"""
Technique 21 — Amari–Chentsov Cubic Tensor

Third-order information geometry. The Fisher information matrix I_ij(θ) = E[s_i s_j]
captures quadratic (second-order) risk. The Amari–Chentsov tensor

    T_ijk(θ) = E[s_i(x) s_j(x) s_k(x)]

captures cubic asymmetry of the likelihood surface — the geometry that linear
approximations miss.

For a direction u in parameter space, the cubic correction is:

    C(θ; u) = Σ_{i,j,k} T_ijk(θ) u_i u_j u_k

Hidden-tail detector: u^T I(θ) u small AND |C(θ; u)| large
→ second-order risk looks benign but the likelihood surface curves hard into tails.

Operational throttle:
    size_t = size_max / (1 + c |C(θ_t; u_t)|)
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class AmariChentsovResult:
    times:       np.ndarray   # (T,) time index
    cubic_score: np.ndarray   # (T,) |C(θ_t; u_t)| — cubic asymmetry magnitude
    fisher_quad: np.ndarray   # (T,) u^T I(θ) u — second-order risk in direction u
    size:        np.ndarray   # (T,) throttled position size
    hidden_tail: np.ndarray   # (T,) bool: low Fisher quad but high cubic
    theta:       np.ndarray   # (T, 2) fitted (μ, σ) path


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def _scores_gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Score vector for N(μ, σ²): s = (∂_μ log p, ∂_σ log p), shape (N, 2)."""
    s_mu    = (x - mu) / sigma ** 2
    s_sigma = (x - mu) ** 2 / sigma ** 3 - 1.0 / sigma
    return np.stack([s_mu, s_sigma], axis=1)


def _fisher_gaussian(sigma: float) -> np.ndarray:
    """Analytical 2×2 Fisher information matrix for N(μ, σ²)."""
    return np.array([[1.0 / sigma ** 2, 0.0],
                     [0.0,              2.0 / sigma ** 2]])


def _cubic_tensor(scores: np.ndarray) -> np.ndarray:
    """T_ijk = (1/N) Σ_n s_i(n) s_j(n) s_k(n), shape (2, 2, 2)."""
    return np.einsum("ni,nj,nk->ijk", scores, scores, scores) / len(scores)


def _cubic_correction(T: np.ndarray, u: np.ndarray) -> float:
    """C(θ; u) = Σ_{i,j,k} T_ijk u_i u_j u_k."""
    return float(np.einsum("ijk,i,j,k->", T, u, u, u))


def _min_fisher_direction(I: np.ndarray) -> np.ndarray:
    """Unit vector in the direction of minimum Fisher information (most hidden)."""
    eigenvalues, eigenvectors = np.linalg.eigh(I)
    return eigenvectors[:, np.argmin(eigenvalues)]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(returns:    np.ndarray,
        window:     int   = 60,
        c:          float = 1.0,
        size_max:   float = 1.0,
        tau_fisher: float = 0.5,
        tau_cubic:  float = 0.1) -> AmariChentsovResult:
    """
    Parameters
    ----------
    returns    : (T,) daily returns
    window     : rolling calibration window
    c          : throttle steepness — higher = more aggressive size reduction
    size_max   : maximum position size
    tau_fisher : threshold below which Fisher quad is considered 'small'
    tau_cubic  : threshold above which cubic score is considered 'large'
    """
    T = len(returns)
    cubic_scores = np.full(T, np.nan)
    fisher_quads = np.full(T, np.nan)
    sizes        = np.full(T, np.nan)
    thetas       = np.full((T, 2), np.nan)

    for t in range(window, T):
        chunk = returns[t - window: t]
        mu    = chunk.mean()
        sigma = max(chunk.std(ddof=1), 1e-8)
        thetas[t] = [mu, sigma]

        scores = _scores_gaussian(chunk, mu, sigma)
        I_mat  = _fisher_gaussian(sigma)
        T_ijk  = _cubic_tensor(scores)

        u     = _min_fisher_direction(I_mat)
        quad  = float(u @ I_mat @ u)
        cubic = abs(_cubic_correction(T_ijk, u))

        cubic_scores[t] = cubic
        fisher_quads[t] = quad
        sizes[t]        = size_max / (1.0 + c * cubic)

    hidden_tail = (fisher_quads < tau_fisher) & (cubic_scores > tau_cubic)

    return AmariChentsovResult(
        times=np.arange(T),
        cubic_score=cubic_scores,
        fisher_quad=fisher_quads,
        size=sizes,
        hidden_tail=hidden_tail,
        theta=thetas,
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(21)

    T = 600
    r1 = rng.normal(0.0002, 0.010, 200)                    # symmetric — low cubic
    r2 = rng.standard_t(df=3, size=150) * 0.012 - 0.001    # fat tails — high cubic
    r3 = rng.normal(0.0001, 0.009, 150)                     # symmetric again
    r4 = -np.abs(rng.normal(0, 0.015, 100))                 # one-sided negative
    returns = np.concatenate([r1, r2, r3, r4])

    res = run(returns, window=60, c=2.0, tau_fisher=0.8, tau_cubic=0.15)

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    fig.suptitle("Technique 21 — Amari–Chentsov Cubic Tensor", fontweight="bold")

    axes[0].plot(returns, lw=0.7, color="steelblue")
    axes[0].axvspan(200, 350, alpha=0.12, color="orange", label="Fat-tail regime")
    axes[0].axvspan(500, 600, alpha=0.12, color="red", label="One-sided tail")
    axes[0].set_ylabel("Return"); axes[0].legend(fontsize=8)
    axes[0].set_title("Returns")

    t = np.arange(T)
    valid = ~np.isnan(res.cubic_score)
    axes[1].plot(t[valid], res.cubic_score[valid], color="darkorange", lw=1.5, label="|C(θ;u)|")
    axes[1].axhline(0.15, color="red", linestyle="--", label="τ_cubic")
    axes[1].fill_between(t[valid], 0, res.cubic_score[valid], where=res.hidden_tail[valid],
                         alpha=0.3, color="red", label="Hidden-tail alarm")
    axes[1].set_ylabel("|C(θ;u)|"); axes[1].legend(fontsize=8)
    axes[1].set_title("Cubic correction — tail exposure beyond quadratic risk")

    axes[2].plot(t[valid], res.fisher_quad[valid], color="navy", lw=1.5, label="u^T I(θ) u")
    axes[2].axhline(0.8, color="red", linestyle="--", label="τ_fisher")
    axes[2].set_ylabel("Fisher quad"); axes[2].legend(fontsize=8)
    axes[2].set_title("Second-order risk in minimum-Fisher direction")

    axes[3].plot(t[valid], res.size[valid], color="seagreen", lw=1.5, label="size_t")
    axes[3].fill_between(t[valid], 0, res.size[valid], where=res.hidden_tail[valid],
                         alpha=0.3, color="red", label="Throttled (hidden tail)")
    axes[3].set_ylabel("size"); axes[3].set_xlabel("Day"); axes[3].legend(fontsize=8)
    axes[3].set_title("Position size throttled by cubic tail sensitivity")

    plt.tight_layout()
    plt.savefig("demos/output_21_amari_chentsov.png", dpi=120)
    plt.close()
    print("[21] Amari–Chentsov — saved demos/output_21_amari_chentsov.png")
    print(f"     Hidden-tail days: {res.hidden_tail.sum()}")


if __name__ == "__main__":
    demo()
