Apply **Technique 6: Curvature Throttle** to the provided returns.

Steps:
1. Call `curvature_throttle` MCP tool with returns, window, and c parameters
2. Report κ(θ) time series and resulting risk weight w_risk
3. Identify periods of high curvature and what market events coincided
4. Recommend: use w_risk as a multiplier on all position sizes going forward

Key interpretation:
- κ low → model is well-specified, linear approximations hold → full conviction
- κ high → model family is bent → small fit errors cascade → reduce leverage
- c controls sensitivity: c=1 gentle throttle, c=5 aggressive throttle
- Use alongside manifold tube: tube exit + high curvature = double signal to be careful
