Apply **Technique 8: Correlation Manifold Clustering** to the provided multi-asset returns.

Steps:
1. Call `correlation_clustering` MCP tool with (T, n) returns, K regimes, window, step
2. Report regime label timeline and centroid correlation matrices
3. Characterise each regime: risk-on (high corr), risk-off (flight-to-quality), sector-coupled, fragmentation
4. State which regime we are currently in and what it implies for portfolio construction

Key interpretation:
- Regime label stable for long stretch → current correlation structure is persistent
- Frequent switching → unstable market → widen error bars on all correlation estimates
- Risk-off regime → diversification collapses → reduce gross exposure
- Fragmentation regime → low correlations → diversification is highest → scale up

Use K=3 as default (calm / stress / transition). Try K=4 for richer structure.
