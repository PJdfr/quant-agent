Apply **Technique 1: Manifold Tube** to the provided returns data.

Steps:
1. Call `manifold_tube` MCP tool with the returns, window, and epsilon parameters
2. Report the distance-to-path score δt over time
3. Identify tube-exit events (δt > ε) and their timing
4. State the directional signal and suggested position size
5. Interpret: is this estimation noise or a genuine structural departure?

Key thresholds:
- δt ≤ ε → inside tube → hold / no trade
- δt > ε → tube exit → trade, size = min(1, (δt−ε)/ε) × size_max

Always explain what drove the exit: parameter drift in μ, σ, or both.
