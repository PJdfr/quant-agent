Apply **Technique 16: Curvature-Penalized Fitting** to the provided returns.

Steps:
1. Call `curvature_penalized_fit` MCP tool with returns and penalty weight λ
2. Compare fitted parameters (μ, σ) with and without curvature penalty
3. Report κ(θ) at both solutions — penalised solution should be in a flatter zone
4. Use the fragility throttle size_t = 1/(1+κ) for position sizing

Key interpretation:
- κ_plain >> κ_penalised → plain MLE landed in a fragile, curved zone
- Penalised fit sacrifices a small amount of data fit for much better stability
- Use rolling_fit to track κ over time and auto-throttle updates
- λ too small → near plain MLE (no benefit); λ too large → over-regularised toward prior
- Start with λ=0.5 and tune by cross-validated forecast error
