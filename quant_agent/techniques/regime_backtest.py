"""
Technique 15 — Regime-Local Backtesting Protocol

8-step protocol for testing a signal's predictive value within a specific regime,
avoiding the trap of global backtests that mix regimes and overstate performance.

Steps:
  1. Declare regime S (define by observables: vol, session, liquidity, ridge, ...)
  2. Fix the prediction object (density, event probability, or quantile)
  3. Match the proper score (log score, Brier, pinball)
  4. Construct a regime-local baseline (built only inside S)
  5. Evaluate on held-out timestamps inside S
  6. Check recurrence stability (score across repeated regime visits)
  7. Verify local calibration (probabilities match realized frequencies inside S)
  8. Assign evidential weight (only if improvement persists across folds + checks)
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable


@dataclass
class RegimeBacktestResult:
    # regime
    regime_mask: np.ndarray         # bool (T,)
    regime_visits: list             # list of (start, end) tuples per visit

    # scores
    signal_scores: np.ndarray       # score series inside regime (held-out)
    baseline_scores: np.ndarray     # regime-local baseline scores
    skill_score: float              # (mean_signal - mean_baseline) / |mean_baseline|
    recurrence_scores: list         # score per regime visit (stability check)

    # calibration
    calibration_error: float        # mean |predicted prob - realized freq| in bins
    is_calibrated: bool             # calibration error below threshold

    # evidential weight
    evidential_weight: float        # ∈ [0,1]; 1 = fully credible
    passes: dict                    # which checks passed

    prediction_type: str
    score_name: str


# ---------------------------------------------------------------------------
# Scoring rules
# ---------------------------------------------------------------------------

def _log_score(pred_probs: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """Log score for density predictions. outcomes are bin indices."""
    eps = 1e-10
    return np.log(np.maximum(pred_probs[np.arange(len(outcomes)), outcomes.astype(int)], eps))


def _brier_score(pred_probs: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """Brier score for binary event predictions. outcomes ∈ {0,1}."""
    return -(pred_probs - outcomes) ** 2   # negated so higher = better


def _pinball_loss(pred_quantile: np.ndarray, outcomes: np.ndarray,
                  alpha: float = 0.5) -> np.ndarray:
    """Pinball loss for quantile predictions. Negated so higher = better."""
    err = outcomes - pred_quantile
    return -(np.where(err >= 0, alpha * err, (alpha - 1) * err))


# ---------------------------------------------------------------------------
# Regime identification
# ---------------------------------------------------------------------------

def _find_visits(mask: np.ndarray) -> list:
    """Find contiguous True segments in boolean mask."""
    visits = []
    in_regime = False
    start = 0
    for t, m in enumerate(mask):
        if m and not in_regime:
            start = t; in_regime = True
        elif not m and in_regime:
            visits.append((start, t)); in_regime = False
    if in_regime:
        visits.append((start, len(mask)))
    return visits


def _calibration_error(pred_probs: np.ndarray,
                        outcomes_binary: np.ndarray,
                        n_bins: int = 5) -> float:
    """ECE-style calibration error: mean |predicted - realized| across probability bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    errors = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (pred_probs >= lo) & (pred_probs < hi)
        if mask.sum() < 3:
            continue
        errors.append(abs(pred_probs[mask].mean() - outcomes_binary[mask].mean()))
    return float(np.mean(errors)) if errors else 0.0


# ---------------------------------------------------------------------------
# Main protocol
# ---------------------------------------------------------------------------

def run(returns: np.ndarray,
        signal_pred: np.ndarray,
        regime_fn: Callable[[np.ndarray, int], bool],
        prediction_type: str = "quantile",
        alpha_quantile: float = 0.1,
        n_folds: int = 5,
        cal_threshold: float = 0.10) -> RegimeBacktestResult:
    """
    Parameters
    ----------
    returns         : (T,) realised return series
    signal_pred     : (T,) signal predictions
                      - if prediction_type='quantile': predicted quantile level
                      - if prediction_type='binary':   predicted event probability
    regime_fn       : callable(returns, t) → bool — is t inside regime S?
    prediction_type : 'quantile' | 'binary'
    alpha_quantile  : quantile level for pinball (e.g. 0.1 = 10th percentile)
    n_folds         : number of time-based CV folds for recurrence check
    cal_threshold   : ECE threshold to call "calibrated"
    """
    T = len(returns)

    # Step 1: Declare regime
    regime_mask = np.array([regime_fn(returns, t) for t in range(T)])
    visits = _find_visits(regime_mask)

    # Step 2+3: score inside regime
    idx_in = np.where(regime_mask)[0]
    if len(idx_in) < 10:
        raise ValueError("Too few observations inside regime to evaluate.")

    sig = signal_pred[idx_in]
    real = returns[idx_in]

    # Step 3: scoring rule
    if prediction_type == "quantile":
        signal_scores = _pinball_loss(sig, real, alpha_quantile)
        score_name = f"Pinball (α={alpha_quantile})"
    else:
        signal_scores = _brier_score(sig, (real < 0).astype(float))
        score_name = "Brier score"

    # Step 4: regime-local baseline = predict with historical quantile inside S
    if prediction_type == "quantile":
        baseline_pred = np.full(len(idx_in), np.quantile(real, alpha_quantile))
        baseline_scores = _pinball_loss(baseline_pred, real, alpha_quantile)
    else:
        p_base = (real < 0).mean()
        baseline_pred = np.full(len(idx_in), p_base)
        baseline_scores = _brier_score(baseline_pred, (real < 0).astype(float))

    # Skill score
    ms = signal_scores.mean(); mb = baseline_scores.mean()
    skill = (ms - mb) / max(abs(mb), 1e-10)

    # Step 6: recurrence stability — score per visit
    recurrence_scores = []
    for start, end in visits:
        local_idx = idx_in[(idx_in >= start) & (idx_in < end)]
        if len(local_idx) < 3:
            continue
        s = signal_pred[local_idx]; r = returns[local_idx]
        if prediction_type == "quantile":
            sc = _pinball_loss(s, r, alpha_quantile).mean()
        else:
            sc = _brier_score(s, (r < 0).astype(float)).mean()
        recurrence_scores.append(float(sc))

    # Step 7: calibration check (binary predictions)
    if prediction_type == "binary":
        cal_err = _calibration_error(sig, (real < 0).astype(float))
    else:
        # for quantile: check crossing rate = α
        crossing_rate = (real < sig).mean()
        cal_err = abs(crossing_rate - alpha_quantile)
    is_cal = cal_err < cal_threshold

    # Step 8: evidential weight
    passes = {
        "skill_positive":    skill > 0,
        "calibrated":        is_cal,
        "recurrence_stable": (np.std(recurrence_scores) / (abs(np.mean(recurrence_scores)) + 1e-10)) < 1.0
                              if len(recurrence_scores) > 1 else False,
        "enough_visits":     len(visits) >= 3,
    }
    ev_weight = sum(passes.values()) / len(passes)

    return RegimeBacktestResult(
        regime_mask=regime_mask,
        regime_visits=visits,
        signal_scores=signal_scores,
        baseline_scores=baseline_scores,
        skill_score=float(skill),
        recurrence_scores=recurrence_scores,
        calibration_error=float(cal_err),
        is_calibrated=is_cal,
        evidential_weight=float(ev_weight),
        passes=passes,
        prediction_type=prediction_type,
        score_name=score_name,
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(505)

    T = 1000
    # Realised returns with time-varying vol
    vol = np.where(np.arange(T) % 200 < 60, 0.025, 0.008)    # high-vol episodes
    returns = rng.normal(0, vol)

    # Signal: predicts the 10th percentile (downside quantile)
    # Good in high-vol regime, noisy in low-vol
    signal = np.where(vol > 0.01,
                      returns * 0.8 + rng.normal(0, 0.005, T),  # informative
                      rng.normal(0, 0.01, T))                    # noise

    # Regime: high volatility (vol > threshold, estimated from rolling std)
    window = 30
    rolling_vol = np.array([returns[max(0, t-window):t].std() if t > 0 else 0.01
                             for t in range(T)])

    def high_vol_regime(returns, t):
        if t < window:
            return False
        rv = returns[t - window: t].std()
        return rv > 0.015

    res = run(returns, signal,
              regime_fn=high_vol_regime,
              prediction_type="quantile",
              alpha_quantile=0.1,
              cal_threshold=0.08)

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    fig.suptitle("Technique 15 — Regime-Local Backtesting Protocol", fontweight="bold")

    t = np.arange(T)
    axes[0].plot(t, returns, color="steelblue", lw=0.7, alpha=0.8, label="Returns")
    axes[0].fill_between(t, -0.1, 0.1, where=res.regime_mask,
                         alpha=0.2, color="orange", label=f"High-vol regime ({res.regime_mask.sum()} days)")
    axes[0].set_ylabel("Return"); axes[0].legend(fontsize=8)
    axes[0].set_title(f"Regime definition: high-vol periods  ({len(res.regime_visits)} visits)")

    # rolling score inside regime
    in_idx = np.where(res.regime_mask)[0]
    axes[1].plot(in_idx, res.signal_scores, color="seagreen", lw=1, alpha=0.7, label="Signal score")
    axes[1].plot(in_idx, res.baseline_scores, color="gray", lw=1, alpha=0.7, label="Baseline score")
    axes[1].axhline(res.signal_scores.mean(), color="seagreen", linestyle="--", lw=2,
                    label=f"Signal mean = {res.signal_scores.mean():.4f}")
    axes[1].axhline(res.baseline_scores.mean(), color="gray", linestyle="--", lw=2,
                    label=f"Baseline mean = {res.baseline_scores.mean():.4f}")
    axes[1].set_ylabel(res.score_name); axes[1].legend(fontsize=7)
    axes[1].set_title(f"Regime-local scores  |  Skill = {res.skill_score*100:.1f}%")

    # recurrence stability
    if res.recurrence_scores:
        visit_n = list(range(1, len(res.recurrence_scores)+1))
        colors = ["seagreen" if s > res.baseline_scores.mean() else "red"
                  for s in res.recurrence_scores]
        axes[2].bar(visit_n, res.recurrence_scores, color=colors, alpha=0.7)
        axes[2].axhline(res.baseline_scores.mean(), color="gray", linestyle="--",
                        label="Baseline mean")
        axes[2].set_xlabel("Regime visit #"); axes[2].set_ylabel("Mean score")
        axes[2].set_title(f"Recurrence stability — score per regime visit  "
                          f"|  Cal error={res.calibration_error:.3f}  "
                          f"|  EW={res.evidential_weight:.2f}")
        axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("demos/output_15_regime_backtest.png", dpi=120)
    plt.close()
    print("[15] Regime Backtest — saved demos/output_15_regime_backtest.png")
    print(f"     Skill score: {res.skill_score*100:.1f}%  |  Calibrated: {res.is_calibrated}")
    print(f"     Evidential weight: {res.evidential_weight:.2f}  |  Checks: {res.passes}")


if __name__ == "__main__":
    demo()
