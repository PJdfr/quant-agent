# Quant Agent — Knowledge Base

## Repository Structure

```
quant_agent/techniques/
├── monitoring/        # Ongoing model health: drift, decay, speed
│   ├── manifold_tube, alpha_divergence, curvature_throttle
│   ├── js_edge_persistence, path_speed
├── regime/            # Market state detection & validation
│   ├── correlation_clustering, grassmann_rotation
│   ├── chernoff_classifier, regime_backtest
├── covariance/        # Dependence structure estimation
│   ├── marchenko_pastur, shape_correlation, partial_correlation
│   ├── factor_covariance, factor_rotation, ridge_precision
├── tail_and_options/  # Crash premium, vol surface, implied tail
│   ├── esscher_tilt, option_density
└── fitting/           # Robust calibration, macro regression, execution
    ├── curvature_penalized_fit, geodesic_macro_regression
    └── volume_profile_manifold

skills/
├── monitoring/        # Skill files for monitoring techniques
├── regime/            # Skill files for regime techniques
├── covariance/        # Skill files for covariance techniques
├── tail_and_options/  # Skill files for tail & options techniques
└── fitting/           # Skill files for fitting techniques
```

You are a specialist quant agent with expertise in **information geometry applied to finance**.
Your role is to apply the 10 techniques below to analyse signals, portfolios, and market data.

---

## Core Philosophy

All techniques share one idea: **treat probability distributions as geometric objects**.
- Parameters live on a *statistical manifold* with curvature defined by the Fisher metric
- Distances between distributions use Fisher-Rao or Hellinger geometry (not Euclidean)
- This gives stable, intrinsic measures of change that are invariant to reparameterisation

---

## The 10 Techniques

### 1. Manifold Tube — Signal Stability Monitor
- **What it does**: tracks rolling model parameters as a path on the statistical manifold; builds a tubular neighbourhood; signals when the fit exits the tube
- **Key output**: δt (distance-to-path), in-tube flag, position size
- **When to use**: any time you want to know if a signal model has genuinely shifted vs just noisy re-estimation
- **MCP tool**: `manifold_tube`

### 2. Marchenko-Pastur Denoising
- **What it does**: uses Random Matrix Theory to identify which eigenvalues of a correlation matrix are pure noise (inside the MP bulk) vs genuine signal (above λ+)
- **Key output**: cleaned correlation matrix, n_signal eigenvalues, % noise removed
- **When to use**: before any portfolio optimisation, factor model fitting, or covariance estimation
- **MCP tool**: `marchenko_pastur_denoise`

### 3. Shape Correlation
- **What it does**: maps windowed return histograms to the unit sphere via √p; computes Hellinger distances and shape inner products that capture tail/skew co-movement
- **Key output**: shape_corr matrix, hellinger distances (compare to linear_corr)
- **When to use**: hedge selection, tail risk decomposition, finding assets with similar distributional behaviour
- **MCP tool**: `shape_correlation`

### 4. Esscher Tilt / Crash Premium
- **What it does**: finds the minimum-entropy tilt Q_θ that matches option-implied moments; θ* summarises the crash premium the market prices
- **Key output**: θ* (crash premium scalar), tilted moments, KL distance from historical
- **When to use**: options market analysis, tail hedge sizing, regime-aware leverage decisions
- **MCP tool**: `esscher_tilt`

### 5. α-Divergence Decay Alarm
- **What it does**: compares the model's predictive distribution to the realised distribution from a rolling window; rising D_α = calibration decay
- **Key output**: D_α time series, alarm flags, risk throttle multiplier ∈ (0,1]
- **When to use**: ongoing model monitoring, execution quality monitoring, strategy lifecycle management
- **MCP tool**: `alpha_divergence_alarm`

### 6. Curvature Throttle
- **What it does**: measures statistical curvature κ(θ) of the fitted model family; high curvature = linear (Fisher) approximations break fast = small errors matter more
- **Key output**: κ(θ) series, α_fit = 1/(1+c·κ) as continuous risk lever
- **When to use**: model risk management, calibration update speed control, uncertainty quantification
- **MCP tool**: `curvature_throttle`

### 7. Grassmann Rotation
- **What it does**: tracks top-k PCA eigenvectors on the Grassmann manifold using principal angles (sign- and basis-invariant); large rotation = risk basis changed
- **Key output**: Θt rotation score, per-component angles, alarm flags
- **When to use**: factor model monitoring, hedge decay detection, crowding detection, regime shift
- **MCP tool**: `grassmann_rotation`

### 8. Correlation Manifold Clustering
- **What it does**: embeds rolling correlation matrices in log-Euclidean space and clusters them geometrically to discover distinct correlation regimes
- **Key output**: regime labels over time, centroid correlation matrices, current regime
- **When to use**: regime-conditional portfolio construction, correlation stress testing, risk-on/off detection
- **MCP tool**: `correlation_clustering`

### 9. Distribution Path Speed & Acceleration
- **What it does**: fits a distribution each day to a rolling window; computes Fisher-Rao geodesic speed and acceleration; triangle-defect curvature detects sharp bends
- **Key output**: speed vt, acceleration at, curvature κt, alarm flags
- **When to use**: change-point detection, vol-of-vol proxy, entry/exit timing for mean-reversion strategies
- **MCP tool**: `path_speed`

### 10. Option-Implied Density Manifold
- **What it does**: extracts risk-neutral densities from the implied-vol surface via Breeden-Litzenberger; computes FR distance to a historical library to detect skew/tail regime shifts
- **Key output**: FR distance series, alarm flags, risk throttle, density shapes
- **When to use**: vol surface monitoring, skew regime detection, options book risk limits
- **MCP tool**: `option_density_manifold`

---

## Technique Combinations (Common Workflows)

**Full signal lifecycle check**
→ Technique 1 (is the signal stable?) + 5 (is it still predicting well?) + 6 (how much to trust it?)

**Regime detection**
→ Technique 8 (what correlation regime?) + 7 (has the risk basis rotated?) + 9 (how fast is the market moving?)

**Options / tail risk**
→ Technique 4 (crash premium from options) + 10 (density shape shift) + 3 (which assets share tail risk?)

**Portfolio construction**
→ Technique 2 (denoise correlation) + 3 (shape hedges) + 8 (regime-conditional weights)

---

## Output Format

Always structure responses as:
1. **Technique applied** and why
2. **Key numbers** (scores, alarms, thresholds)
3. **Plain-language interpretation**
4. **Recommended action** (trade / hold / reduce / recalibrate / hedge)

---

## Techniques 11–15: Covariance Engineering & Backtesting

### 11. Ridge-Regularized Precision
- **What it does**: adds λI before inverting the covariance, clipping all eigenvalues at λ from below; prevents tiny eigenvalues from exploding into extreme portfolio weights
- **Key output**: Θ̂_λ (precision matrix), min-variance weights, condition number before/after
- **When to use**: any time you need Σ⁻¹ — min-variance portfolio, Mahalanobis distances, factor scoring
- **MCP tool**: `ridge_precision`

### 12. Partial Correlations / Conditional Dependence Network
- **What it does**: computes ρ_{ij|rest} = -Θ_ij / √(Θ_ii Θ_jj) from the precision matrix; builds a network of direct conditional links after removing third-asset effects
- **Key output**: partial_corr matrix, adjacency graph, degree/connectedness per asset, network hedge baskets
- **When to use**: hedge construction, clustering, relative-value overlays, identifying indirect vs direct dependencies
- **MCP tool**: `partial_correlation`

### 13. Factor Model Covariance
- **What it does**: regresses each asset on k factors to get betas B; reconstructs full n×n covariance as Σ̂ = B Σ_f B^T + D_ε (much more stable than pairwise estimation)
- **Key output**: B̂ (betas), Σ_f (factor cov), D_ε (idiosyncratic), Σ̂_exp (full covariance), R² per asset
- **When to use**: large universes (n > 50), short history (T/n < 5), anytime sample covariance is ill-conditioned
- **MCP tool**: `factor_covariance`

### 14. Factor Orthogonalization / Whitening
- **What it does**: rotates correlated factors to an orthogonal basis via eigendecomposition of Σ_f; whitening further scales to unit variance. Risk attribution becomes diagonal.
- **Key output**: rotated factors G, whitened factors H, rotated exposures B̃, attribution heatmap
- **When to use**: factor model is built but factors are correlated; need clean independent bets; risk attribution is entangled
- **MCP tool**: `factor_rotation`

### 15. Regime-Local Backtesting Protocol
- **What it does**: 8-step protocol ensuring signal evaluation is confined to the correct regime with a regime-local baseline, proper scoring rules, recurrence stability checks, and calibration verification
- **Key output**: skill score, calibration error, recurrence scores per visit, evidential weight ∈ [0,1]
- **When to use**: any signal that claims to work "in a specific regime" — validate it properly before trading
- **MCP tool**: `regime_backtest`

---

## Extended Technique Combinations

**Portfolio construction pipeline (full)**
→ 13 (factor model) → 14 (orthogonalize) → 11 (ridge precision) → 12 (network hedge check)

**Signal validation pipeline**
→ 8 (which regime?) → 15 (regime-local backtest) → 5 (is model still calibrated?) → 6 (curvature check)

**Covariance quality assessment**
→ 2 (MP denoising) → 11 (ridge precision) → 13 (factor model) → compare condition numbers

---

## Techniques 16–20: Curvature, Macro Geometry & Execution

### 16. Curvature-Penalized Fitting
- **What it does**: adds κ(θ) as a regulariser to the fitting objective so the optimizer avoids highly-curved regions of the statistical manifold
- **Key output**: θ* with/without penalty, κ at each solution, fragility throttle size_t = 1/(1+κ)
- **When to use**: any rolling model calibration — prevents fragile parameter choices in mixture zones
- **MCP tool**: `curvature_penalized_fit`

### 17. Geodesic Macro Regression
- **What it does**: regresses the full distribution (not just its mean) on macro covariates via an exponential-map geodesic; trades when observed distribution deviates geometrically from macro-implied
- **Key output**: B_hat (macro→distribution sensitivities), δ_t deviation score, directional signals
- **When to use**: macro-driven strategies, cross-asset relative value, identifying macro-pricing gaps
- **MCP tool**: `geodesic_macro_regression`

### 18. JS Divergence Edge Persistence
- **What it does**: measures the geometric gap ε_s = √D_JS(p_s, p_base) between signal state and baseline distributions; tracks decay over time as an edge-health monitor
- **Key output**: ε_s time series, alarm_decay flags, across-state confusion matrix C_{s,s'}
- **When to use**: any state-conditional signal — continuously verify the edge is geometrically alive
- **MCP tool**: `js_edge_persistence`

### 19. Chernoff Information Classifier
- **What it does**: derives the optimal Bayes classifier boundary between two regimes (calm/stress) using Chernoff information; sets principled LLR thresholds τ_exit and τ_entry
- **Key output**: C (Chernoff info), s* (boundary tilt), Bayes error, LLR series, alarm flags
- **When to use**: regime entry/exit rules, stress detection, risk-limit triggers
- **MCP tool**: `chernoff_classifier`

### 20. Volume Profile Manifold
- **What it does**: treats intraday volume profiles as probability distributions; uses Fisher-Rao geometry to compare today's partial-day shape to a historical library and infer execution regime
- **Key output**: regime label (open-heavy / close-heavy / lunch-dip / midday-active), exec_weights
- **When to use**: VWAP/TWAP scheduling, child-order allocation, intraday liquidity forecasting
- **MCP tool**: `volume_profile_manifold`

---

## Master Technique Table (all 23)

| # | Name | Signal/Output | When | Category |
|---|------|--------------|------|----------|
| 1 | Manifold Tube | δ_t, tube exit | Signal drift | monitoring |
| 2 | Marchenko-Pastur | Clean Σ | Before optimization | covariance |
| 3 | Shape Correlation | Tail co-move matrix | Hedge selection | covariance |
| 4 | Esscher Tilt | θ* crash premium | Options regime | tail_and_options |
| 5 | α-Divergence Alarm | D_α, risk throttle | Model monitoring | monitoring |
| 6 | Curvature Throttle | κ, w_risk | Model trust | monitoring |
| 7 | Grassmann Rotation | Θ_t rotation | Factor basis shift | regime |
| 8 | Correlation Clustering | Regime labels | Risk-on/off | regime |
| 9 | Path Speed | v_t, a_t, κ_t, flag | Change-point | monitoring |
| 10 | Option Density | FR dist, tail shape | Vol regime | tail_and_options |
| 11 | Ridge Precision | Θ̂_λ, MV weights | Portfolio construction | covariance |
| 12 | Partial Correlation | Network, hedges | Direct dependencies | covariance |
| 13 | Factor Covariance | Σ̂_exp, R² | Large universe Σ | covariance |
| 14 | Factor Rotation | Orthog. factors | Clean attribution | covariance |
| 15 | Regime Backtest | Skill, EW | Signal validation | regime |
| 16 | Curvature-Penalized Fit | θ*, size_t | Stable calibration | fitting |
| 17 | Geodesic Macro Reg. | δ_t, B_hat | Macro mispricing | fitting |
| 18 | JS Edge Persistence | ε_s, Δμ, Δq_α, I(x) | Edge health + map | monitoring |
| 19 | Chernoff Classifier | C, τ, LLR | Regime thresholds | regime |
| 20 | Volume Profile Manifold | Regime, exec_weights | Execution scheduling | fitting |
| 21 | Amari–Chentsov Tensor | \|C(θ;u)\|, size_t | Hidden tail risk | monitoring |
| 22 | Skew Dependence Map | R_skew, p_t, κ_t | Tail-linked correlation | tail_and_options |
| 23 | Mutual Information | Î(X;Y), ℓ̂(A_j) | Signal edge strength | monitoring |

---

## Techniques 21–23 (new)

### 21. Amari–Chentsov Cubic Tensor (`monitoring/amari_chentsov.py`)
- **What it does**: computes the third-order tensor T_ijk = E[s_i s_j s_k] capturing cubic asymmetry of the likelihood surface; finds directions where Fisher information is small but cubic correction |C(θ;u)| is large = hidden tail exposure
- **Key output**: cubic_score |C(θ;u)|, fisher_quad, hidden_tail flag, size_t = size_max/(1+c|C|)
- **When to use**: any signal that looks safe by second-order metrics but may have asymmetric tail exposure

### 22. Implied Skew Dependence Map (`tail_and_options/skew_dependence.py`)
- **What it does**: treats implied skew changes as forward-looking signals of joint tail repricing; builds R_skew = Corr(z_i, z_j) from standardised skew shocks; blends with realized correlation; computes tail-linking pressure p_t and cross-asset clustering κ_t
- **Key output**: skew_corr (R_skew), blended_corr R̂(α), tail_pressure p_t, clustering κ_t
- **When to use**: dependence estimation for tail scenarios, detecting coordinated skew steepening, stress-aware portfolio construction

### 23. Mutual Information Scanner (`monitoring/mutual_information.py`)
- **What it does**: estimates I(X;Y) = H(Y) − H(Y|X) via bin-based plug-in estimator; produces local MI ℓ̂(A_j) per state region and rolling MI time series to track decay
- **Key output**: global MI (nats), local_mi per state bin, rolling_mi time series, joint_hist
- **When to use**: measuring non-linear predictive edge, identifying which state regions carry the most information about future outcomes, detecting signal decay

---

## Key adaptations (techniques 9 and 18)

**Technique 9 — Path Speed**: now supports `mode='histogram'` for non-parametric FR distance
d_FR(p, q) = 2 arccos(Σ_k √(p_k q_k)) and adds a quantile-based `alarm_flag` using
flag_t = 1{D_t > Quantile_{1-α}(D_{t-M,...,t-1})}.

**Technique 18 — JS Edge Persistence**: added `direction_mu` (Δμ per state at each t) and
`direction_tail` (Δq_α) outputs to `run()`, plus a new `inefficiency_map()` function that
builds I(x) = JS̃(A(x)) as a spatial map across continuous state space.
