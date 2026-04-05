Apply **Technique 17: Geodesic Macro Regression** to the provided returns and macro data.

Steps:
1. Call `geodesic_macro_regression` MCP tool with returns, macro matrix, epsilon
2. Inspect B_hat — which macro variables drive distribution mean and vol?
3. Plot observed vs macro-implied parameters over time
4. Identify periods of large geometric deviation δ_t > ε — these are trade signals

Key interpretation:
- δ_t large → market distribution deviates from what macro state implies → mispricing
- Signal direction: obs mean above implied → positive (trend following macro)
- B_hat[0,:] = macro sensitivity of μ (drift channel)
- B_hat[1,:] = macro sensitivity of σ (vol channel)
- Good covariates: VIX, credit spreads, rate levels, yield curve slope
- Use δ_t as a standalone signal or as a filter for other strategies
