Apply **Technique 7: Grassmann Rotation** to the provided multi-asset returns.

Steps:
1. Call `grassmann_rotation` MCP tool with (T, n) returns, k, window, tau
2. Report rotation score Θt time series and alarm events
3. Identify which principal components rotated most (by individual angles)
4. Interpret: factor crowding shift, correlation regime break, or hedge decay?

Key interpretation:
- Θt small → risk basis stable → current hedges and factors still valid
- Θt spike → subspace rotated → re-examine all factor exposures
- PC1 angle large → dominant risk factor changed direction (e.g. equity factor shifted)
- PC2/3 angles large → secondary factors reshuffled (sector rotation, crowding)
- Use to trigger: re-run factor decomposition, rebalance hedges, review position limits
