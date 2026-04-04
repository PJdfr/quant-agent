Apply **Technique 9: Distribution Path Speed & Acceleration** to the provided returns.

Steps:
1. Call `path_speed` MCP tool with returns, window, tau_speed, tau_curve
2. Report speed, acceleration, and triangle-defect curvature time series
3. Identify speed spikes (fast transition) and sharp bends (structural snap)
4. Correlate with known market events if available

Key interpretation:
- Speed spike → distribution is moving fast on the manifold → regime transition underway
- High acceleration after speed spike → transition is still accelerating → do not fade
- Triangle defect κt very negative → path is bending sharply → model is entering a new region
- Calm period (low speed, low accel) → good conditions for signal extraction
- Use speed as a vol-of-vol proxy and throttle position sizing accordingly
