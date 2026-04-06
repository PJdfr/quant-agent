Apply **Technique 14: Factor Orthogonalization / Whitening** to the provided factor model.

Steps:
1. Call `factor_rotation` MCP tool with exposure matrix B (n×k) and factor returns (T×k)
2. Report pre/post rotation factor correlation matrix (should go near-diagonal)
3. Show variance contribution per rotated factor (scree in orthogonal basis)
4. Show per-asset, per-factor attribution heatmap (now diagonal = independent)

Key interpretation:
- Off-diagonal factor corr drops to ~0 → factors are now clean, independent bets
- Use whitened factors h_t = Λ^{-1/2} U^T f_t for all signal generation
- Use B̃ = BU for rotated exposures, B̃̃ = BUΛ^{1/2} for whitened exposures
- Risk attribution: Var(w^T r) ≈ β_g^T Λ β_g + w^T D_ε w → each factor term is independent
- With whitening: Var ≈ ‖β_h‖² + w^T D_ε w → even simpler, pure L2 norm
- Size each factor exposure independently based on signal strength per rotated factor
