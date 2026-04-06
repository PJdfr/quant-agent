Apply **Technique 3: Shape Correlation** to the provided multi-asset returns.

Steps:
1. Call `shape_correlation` MCP tool with the (T, n) returns matrix and asset names
2. Compare shape_correlation vs linear_correlation matrices
3. Highlight pairs where shape >> linear (tail co-movement missed by Pearson)
4. Use Hellinger distances to rank asset pairs by distributional dissimilarity

Key interpretation:
- Shape corr > linear corr → assets share tail/skew behaviour beyond their average co-movement
- Large Hellinger distance → distributions are structurally different → weak hedge
- Use shape correlation to select hedges that track book tail risk, not just beta
