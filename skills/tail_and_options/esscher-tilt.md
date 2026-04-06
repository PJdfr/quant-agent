Apply **Technique 4: Esscher Tilt / Crash Premium** to the provided returns (and optional option targets).

Steps:
1. Call `esscher_tilt` MCP tool (set rolling=true for time series, false for snapshot)
2. Report θ* (crash premium), change in skew from P to Q_θ*, KL distance
3. Compare to historical distribution of θ* (is it elevated?)
4. State the implication for tail hedging and leverage

Key interpretation:
- θ* > 0 and large → market pricing significant left-tail risk
- Rising rolling θ* → crash premium increasing → reduce gross exposure or buy puts
- KL small → market's implied view is close to history → normal regime
- KL large → market see a very different world than history → caution
