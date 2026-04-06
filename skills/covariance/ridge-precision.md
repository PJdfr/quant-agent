Apply **Technique 11: Ridge-Regularized Precision** to the provided return matrix.

Steps:
1. Call `ridge_precision` MCP tool with the (T, n) returns and optionally a λ value
2. Report the condition number before/after regularisation and the λ selected
3. Compare min-variance weights: raw inversion vs ridge-stabilised
4. Recommend a λ range based on the eigenvalue spectrum

Key interpretation:
- High condition number (>1000) → raw inversion is dangerous, weights will be extreme
- After ridge: condition drops dramatically, weights become more diversified
- λ too large → over-shrinks toward equal weight (loses signal)
- λ too small → near-raw inversion (unstable)
- Auto-λ rule: set λ = 10th percentile of eigenvalue spectrum (clips noise floor)
- Use Θ̂_λ as a drop-in replacement for Σ⁻¹ everywhere (MVopt, Mahalanobis, scoring)
