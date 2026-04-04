Apply **Technique 13: Factor Model Covariance** to the provided returns and factors.

Steps:
1. Call `factor_covariance` MCP tool with (T, n) returns and (T, k) factor returns
2. Report R² per asset — which assets are well-explained by the factors?
3. Compare condition number of factor-model covariance vs sample covariance
4. Use Σ̂_exp = B Σ_f B^T + D_ε for all downstream portfolio construction

Key interpretation:
- R² > 70% → asset well captured by factor model → use model covariance confidently
- R² < 30% → idiosyncratic dominated → need more factors or asset-specific treatment
- Factor cov condition << sample cov condition → model is dramatically more stable
- If no observable factors: use PCA factors (call pca_factors utility first)
- D_ε diagonal → idiosyncratic risks are independent → add them back cleanly
