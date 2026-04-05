"""
Technique 20 — Volume Profile Manifold (Execution Shape)

Normalise intraday volume into a probability distribution over time-of-day bins.
Each day is a point on the density manifold; shape differences are measured by
Fisher-Rao geometry (arccos of inner product of square-root densities).

Compare today's partial-day shape to a historical library to infer the closest
execution regime (open-heavy, lunch-dip, close-heavy). Adapt child-order weights
to match the inferred shape: front-load, neutral, or back-load.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class VolumeProfileResult:
    times:            np.ndarray    # day index
    fr_distance:      np.ndarray    # FR dist from today to nearest historical
    nearest_day:      np.ndarray    # index of nearest historical day
    regime_label:     np.ndarray    # string label per day
    exec_weights:     np.ndarray    # (T, m) recommended child-order weights
    profiles:         np.ndarray    # (T, m) normalised volume profiles
    centroids:        dict          # {label: centroid profile}
    bin_labels:       list          # time-of-day bin names


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def _normalise(vol: np.ndarray) -> np.ndarray:
    total = vol.sum()
    if total < 1e-10:
        return np.ones(len(vol)) / len(vol)
    return vol / total


def _sqrt_density(p: np.ndarray) -> np.ndarray:
    p = np.maximum(p, 1e-12)
    psi = np.sqrt(p)
    return psi / np.linalg.norm(psi)


def _fr_dist(p: np.ndarray, q: np.ndarray) -> float:
    inner = np.dot(_sqrt_density(p), _sqrt_density(q))
    return float(2 * np.arccos(np.clip(inner, -1, 1)))


def _classify_regime(profile: np.ndarray) -> str:
    """Simple heuristic regime label from volume profile shape."""
    m = len(profile)
    open_wt  = profile[:m//4].sum()
    close_wt = profile[-m//4:].sum()
    mid_wt   = profile[m//4:-m//4].sum()
    if open_wt > 0.40:
        return "open-heavy"
    elif close_wt > 0.40:
        return "close-heavy"
    elif mid_wt > 0.50:
        return "midday-active"
    else:
        return "lunch-dip"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(volume_matrix:  np.ndarray,
        bin_labels:     list | None = None,
        history_size:   int   = 30,
        tau_fr:         float = 0.30) -> VolumeProfileResult:
    """
    Parameters
    ----------
    volume_matrix : (T, m) — raw intraday volumes per bin per day
    bin_labels    : list of m bin name strings (e.g. '09:30', '10:00', ...)
    history_size  : library size for nearest-neighbour comparison
    tau_fr        : FR distance threshold for "unusual" shape
    """
    T, m = volume_matrix.shape
    bin_labels = bin_labels or [f"B{i}" for i in range(m)]

    # Normalise each day to a probability distribution
    profiles = np.array([_normalise(volume_matrix[t]) for t in range(T)])

    fr_dists   = np.full(T, np.nan)
    nearest    = np.full(T, -1, dtype=int)
    regimes    = np.full(T, "", dtype=object)
    exec_w     = np.zeros((T, m))

    for t in range(history_size, T):
        lib = profiles[t - history_size: t]
        today = profiles[t]
        dists = [_fr_dist(today, lib[i]) for i in range(history_size)]
        j_star = int(np.argmin(dists))
        fr_dists[t] = dists[j_star]
        nearest[t]  = t - history_size + j_star

        # Infer regime from nearest historical match
        regime = _classify_regime(today)
        regimes[t] = regime

        # Execution weights: match child orders to inferred future shape
        # Use the nearest historical day's full-day profile as the template
        template = profiles[nearest[t]]
        # Remaining-day weights: re-normalise template from current bin forward
        current_bin = min(int((t % T) / T * m), m - 1)   # proxy: uniform position
        remaining = template.copy()
        remaining[:current_bin] = 0
        exec_w[t] = _normalise(remaining)

    # Centroids per regime (geometric mean in Fisher-Rao space)
    centroids = {}
    for label in ["open-heavy", "close-heavy", "midday-active", "lunch-dip"]:
        mask = regimes == label
        if mask.any():
            psi_stack = np.array([_sqrt_density(profiles[t]) for t in np.where(mask)[0]])
            mean_psi  = psi_stack.mean(axis=0)
            mean_psi /= np.linalg.norm(mean_psi)
            centroids[label] = mean_psi ** 2   # back to density

    return VolumeProfileResult(
        times=np.arange(T),
        fr_distance=fr_dists,
        nearest_day=nearest,
        regime_label=regimes,
        exec_weights=exec_w,
        profiles=profiles,
        centroids=centroids,
        bin_labels=bin_labels,
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(20)

    T = 200
    m = 13   # 30-min bins 9:30–16:00

    bin_labels = ["9:30","10:00","10:30","11:00","11:30","12:00",
                  "12:30","13:00","13:30","14:00","14:30","15:00","15:30"]

    def make_profile(regime, noise=0.3):
        base = np.ones(m)
        if regime == "open-heavy":
            base[:3] *= 4.0; base[-2:] *= 2.0
        elif regime == "close-heavy":
            base[-3:] *= 4.5; base[:2] *= 1.5
        elif regime == "lunch-dip":
            base[4:8] *= 0.3; base[:2] *= 2.0; base[-2:] *= 2.5
        else:  # midday-active
            base[4:9] *= 2.5
        return np.maximum(base + rng.normal(0, noise, m), 0.01)

    # Regime sequence
    regime_seq = (["open-heavy"]*60 + ["lunch-dip"]*60 +
                  ["close-heavy"]*50 + ["open-heavy"]*30)
    vol_matrix = np.array([make_profile(regime_seq[t]) for t in range(T)])

    res = run(vol_matrix, bin_labels, history_size=30)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle("Technique 20 — Volume Profile Manifold (Execution Shape)", fontweight="bold")

    # top-left: FR distance over time
    valid = ~np.isnan(res.fr_distance)
    t_valid = res.times[valid]
    axes[0, 0].plot(t_valid, res.fr_distance[valid], color="darkorange", lw=1.5,
                    label="FR dist to nearest historical")
    axes[0, 0].axhline(0.30, color="red", linestyle="--", label="τ=0.30 (unusual shape)")
    axes[0, 0].set_title("Fisher-Rao distance to nearest historical profile")
    axes[0, 0].set_xlabel("Day"); axes[0, 0].set_ylabel("FR distance")
    axes[0, 0].legend(fontsize=8)

    # top-right: regime labels over time
    colors_map = {"open-heavy":"steelblue","close-heavy":"crimson",
                  "lunch-dip":"seagreen","midday-active":"purple","":"gray"}
    for label, color in colors_map.items():
        mask = res.regime_label == label
        if mask.any():
            axes[0, 1].scatter(res.times[mask], np.ones(mask.sum()),
                               c=color, s=30, label=label)
    axes[0, 1].set_title("Inferred execution regime per day")
    axes[0, 1].set_xlabel("Day"); axes[0, 1].set_yticks([])
    axes[0, 1].legend(fontsize=8, loc="upper right")

    # bottom-left: centroid profiles
    colors_c = ["steelblue","crimson","seagreen","purple"]
    for i, (label, centroid) in enumerate(res.centroids.items()):
        axes[1, 0].plot(centroid, label=label, color=colors_c[i % 4], lw=2)
    axes[1, 0].set_xticks(range(m)); axes[1, 0].set_xticklabels(bin_labels, rotation=45, fontsize=7)
    axes[1, 0].set_ylabel("Volume fraction"); axes[1, 0].set_title("Regime centroid profiles")
    axes[1, 0].legend(fontsize=8)

    # bottom-right: exec weights for 3 example days
    for day, color, label in [(35,"steelblue","Day 35 (open-heavy)"),
                               (90,"seagreen","Day 90 (lunch-dip)"),
                               (140,"crimson","Day 140 (close-heavy)")]:
        axes[1, 1].plot(res.exec_weights[day], label=label, color=color, lw=2)
    axes[1, 1].set_xticks(range(m)); axes[1, 1].set_xticklabels(bin_labels, rotation=45, fontsize=7)
    axes[1, 1].set_ylabel("Child-order weight"); axes[1, 1].set_title("Recommended execution weights by regime")
    axes[1, 1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("demos/output_20_volume_profile_manifold.png", dpi=120)
    plt.close()
    print("[20] Volume Profile Manifold — saved demos/output_20_volume_profile_manifold.png")
    unique, counts = np.unique(res.regime_label[res.regime_label != ""], return_counts=True)
    print(f"     Regime distribution: { {u:c for u,c in zip(unique,counts)} }")


if __name__ == "__main__":
    demo()
