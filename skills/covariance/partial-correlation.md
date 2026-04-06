Apply **Technique 12: Partial Correlations / Conditional Dependence Network** to the provided returns.

Steps:
1. Call `partial_correlation` MCP tool with (T, n) returns, asset names, and threshold τ
2. Compare partial correlation matrix to linear correlation matrix
3. Identify pairs where linear corr is high but partial corr is low (indirect link via third asset)
4. Report the network: most connected nodes, strongest direct edges
5. Build a network hedge basket for a specified asset using direct conditional links

Key interpretation:
- High linear, low partial → spurious correlation (remove from hedge basket)
- High partial → direct dependence → genuine hedge candidate
- Degree(i) high → central node in risk network → systemic asset
- Negative partial corr → conditional inverse relationship → natural hedge
- τ too low → dense noisy network; τ too high → too sparse, misses true structure
