# Quant Agent

Information-geometric techniques for quantitative finance, packaged as a Claude Code agent.

The core idea: **treat probability distributions as geometric objects on a statistical manifold**.
Distances between distributions use Fisher-Rao or Hellinger geometry — not Euclidean —
giving intrinsic, reparameterisation-invariant measures of change.

---

## Technique catalogue (23 techniques across 5 categories)

### Monitoring — model health, drift detection, edge decay

| # | Technique | Key output | When to use |
|---|-----------|------------|-------------|
| 1 | **Manifold Tube** | δ_t, tube exit flag, size | Is my signal model still on-path? |
| 5 | **α-Divergence Alarm** | D_α series, risk throttle | Is my model still describing reality? |
| 6 | **Curvature Throttle** | κ(θ), w_risk ∈ (0,1] | How fragile is the current calibration? |
| 9 | **Path Speed** | speed v_t, acceleration, quantile flag | Is the market transitioning right now? |
| 18 | **JS Edge Persistence** | ε_s, decay alarm, Δμ, Δq_α, I(x) map | Is my signal edge geometrically alive? |
| 21 | **Amari–Chentsov Tensor** | \|C(θ;u)\|, hidden-tail flag, size | Tail risk hidden from second-order metrics |
| 23 | **Mutual Information** | Î(X;Y), local MI ℓ̂(A_j), rolling MI | How much does the state predict the future? |

### Regime — market state detection and validation

| # | Technique | Key output | When to use |
|---|-----------|------------|-------------|
| 7 | **Grassmann Rotation** | Θ_t rotation score, alarm | Has the risk factor basis changed? |
| 8 | **Correlation Clustering** | Regime labels, centroid matrices | Which correlation regime are we in? |
| 15 | **Regime Backtest** | Skill score, calibration error, EW | Is a regime-conditional signal actually valid? |
| 19 | **Chernoff Classifier** | C, s*, LLR series, Bayes error | Principled entry/exit thresholds between regimes |

### Covariance — dependence structure estimation

| # | Technique | Key output | When to use |
|---|-----------|------------|-------------|
| 2 | **Marchenko-Pastur** | Clean Σ, n_signal eigenvalues | Strip noise from correlation matrices |
| 3 | **Shape Correlation** | shape_corr, Hellinger distances | Assets that share tail/skew behaviour |
| 11 | **Ridge Precision** | Θ̂_λ, MV weights, condition number | Stable covariance inversion |
| 12 | **Partial Correlation** | Network, adjacency, degree | Direct vs indirect dependencies |
| 13 | **Factor Covariance** | B̂, Σ̂_exp, R² per asset | Covariance for large/short-history universes |
| 14 | **Factor Rotation** | Orthog. factors G, whitened H | Clean independent bets, diagonal attribution |

### Tail & Options — crash premium, vol surface, implied tail

| # | Technique | Key output | When to use |
|---|-----------|------------|-------------|
| 4 | **Esscher Tilt** | θ* crash premium, KL distance | What tail fear does the options market price? |
| 10 | **Option Density** | FR distance series, risk throttle | Has the implied density shape shifted regime? |
| 22 | **Skew Dependence Map** | R_skew, R̂(α), p_t, κ_t | Tail-linked correlation from skew co-movement |

### Fitting — robust calibration, macro regression, execution

| # | Technique | Key output | When to use |
|---|-----------|------------|-------------|
| 16 | **Curvature-Penalized Fit** | θ* penalised, fragility throttle | Avoid fragile parameter regions during calibration |
| 17 | **Geodesic Macro Regression** | δ_t deviation, B_hat, signal | Macro-implied distribution vs observed |
| 20 | **Volume Profile Manifold** | Regime label, exec_weights | VWAP/TWAP scheduling from intraday volume shape |

---

## Common workflows

**Full signal lifecycle check**
```
9 Path Speed       → is the market moving fast right now?
1 Manifold Tube    → is my signal still on its historical path?
5 α-Divergence     → is the model still predicting well?
6 Curvature Throttle → how much to scale down size?
21 Amari-Chentsov  → any hidden tail exposure I'm missing?
```

**Regime detection pipeline**
```
8 Correlation Clustering  → which correlation regime?
7 Grassmann Rotation      → has the factor basis rotated?
19 Chernoff Classifier    → set entry/exit LLR thresholds
15 Regime Backtest        → validate signal regime-locally
```

**Tail / options risk**
```
4 Esscher Tilt          → crash premium scalar θ*
10 Option Density        → density shape shift from vol surface
22 Skew Dependence Map   → which assets are jointly repriced in stress?
```

**Portfolio construction**
```
2 Marchenko-Pastur  → clean correlation matrix
13 Factor Covariance → stable Σ for large universe
14 Factor Rotation   → orthogonalise factors
11 Ridge Precision   → stable Σ⁻¹ for optimisation
12 Partial Corr      → check which hedge links are direct
```

**Signal edge health**
```
23 Mutual Information    → does state X still predict Y?
18 JS Edge Persistence   → is the distributional edge alive?
   inefficiency_map()    → spatial map of I(x) across state space
```

---

## Repo structure

```
quant-agent/
├── CLAUDE.md                          # Agent knowledge base (auto-loaded by Claude)
├── requirements.txt
│
├── quant_agent/
│   ├── mcp_server.py                  # MCP server — exposes techniques as Claude tools
│   └── techniques/
│       ├── monitoring/                # Ongoing health: drift, decay, speed
│       │   ├── manifold_tube.py       #  1
│       │   ├── alpha_divergence.py    #  5
│       │   ├── curvature_throttle.py  #  6
│       │   ├── path_speed.py          #  9  (+ histogram mode, quantile flag)
│       │   ├── js_edge_persistence.py # 18  (+ spatial map, directionality)
│       │   ├── amari_chentsov.py      # 21  NEW
│       │   └── mutual_information.py  # 23  NEW
│       │
│       ├── regime/                    # Market state detection & validation
│       │   ├── grassmann_rotation.py  #  7
│       │   ├── correlation_clustering.py #  8
│       │   ├── regime_backtest.py     # 15
│       │   └── chernoff_classifier.py # 19
│       │
│       ├── covariance/                # Dependence structure estimation
│       │   ├── marchenko_pastur.py    #  2
│       │   ├── shape_correlation.py   #  3
│       │   ├── ridge_precision.py     # 11
│       │   ├── partial_correlation.py # 12
│       │   ├── factor_covariance.py   # 13
│       │   └── factor_rotation.py     # 14
│       │
│       ├── tail_and_options/          # Crash premium, vol surface, implied tail
│       │   ├── esscher_tilt.py        #  4
│       │   ├── option_density.py      # 10
│       │   └── skew_dependence.py     # 22  NEW
│       │
│       └── fitting/                   # Robust calibration, macro, execution
│           ├── curvature_penalized_fit.py    # 16
│           ├── geodesic_macro_regression.py  # 17
│           └── volume_profile_manifold.py    # 20
│
├── skills/                            # Claude Code slash commands
│   ├── quant-agent.md                 # /quant-agent  (main entry point)
│   ├── monitoring/
│   ├── regime/
│   ├── covariance/
│   ├── tail_and_options/
│   └── fitting/
│
└── demos/
    └── run_all_demos.py               # Synthetic data demos for all techniques
```

---

## Install

```bash
git clone https://github.com/PJdfr/quant-agent
cd quant-agent
pip install -r requirements.txt
```

Register the MCP server in your Claude Code config:
```json
{
  "mcpServers": {
    "quant-agent": {
      "command": "python",
      "args": ["quant_agent/mcp_server.py"]
    }
  }
}
```

---

## Usage in Claude Code

**Main agent entry point:**
```
/quant-agent  analyse these returns and check if the signal is still stable
```

**Specific techniques:**
```
/manifold-tube
/marchenko-pastur
/alpha-divergence
/curvature-throttle
/path-speed
/grassmann-rotation
/correlation-clustering
/esscher-tilt
/option-density
/ridge-precision
/partial-correlation
/factor-covariance
/factor-rotation
/regime-backtest
/chernoff-classifier
/curvature-penalized-fit
/geodesic-macro-regression
/js-edge-persistence
/volume-profile-manifold
```

**Natural language:**
```
@quant-agent  is the equity factor still stable? our PnL looks noisy lately
@quant-agent  run a full regime check on these 10 assets
@quant-agent  what is the options market implying about crash risk?
@quant-agent  build a clean covariance matrix for this 50-asset universe
```

---

## Run demos

```bash
python demos/run_all_demos.py
```

Generates PNG output for each technique in `demos/` using synthetic but realistic data.

---

## Adding a new technique

1. Choose the right category (`monitoring`, `regime`, `covariance`, `tail_and_options`, `fitting`)
2. Create `quant_agent/techniques/<category>/your_technique.py` with a `run()` function and a `demo()`
3. Add it to `quant_agent/techniques/<category>/__init__.py`
4. Add a skill file to `skills/<category>/your-technique.md`
5. Optionally expose it as an MCP tool in `quant_agent/mcp_server.py`

---

## Dependencies

- `numpy >= 1.24`
- `scipy >= 1.10`
- `scikit-learn >= 1.3`
- `matplotlib >= 3.7`
- `mcp >= 1.0` (Anthropic MCP Python SDK)
