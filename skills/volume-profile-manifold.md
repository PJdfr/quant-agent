Apply **Technique 20: Volume Profile Manifold** to infer execution regimes and adapt schedules.

Steps:
1. Call `volume_profile_manifold` MCP tool with (T, m) intraday volume matrix
2. Report today's inferred regime: open-heavy / close-heavy / lunch-dip / midday-active
3. Compare today's FR distance to historical library — large distance = unusual day
4. Use exec_weights to schedule child orders matching inferred volume shape

Key interpretation:
- FR distance low → familiar volume shape → use regime centroid as execution template
- FR distance high → unusual day (earnings? macro event?) → use safer neutral schedule
- open-heavy → front-load child orders to first 90 minutes
- close-heavy → back-load child orders to last 90 minutes
- lunch-dip → avoid midday; split front/back
- midday-active → spread evenly, opportunistic at midday
- Update historical library daily; re-cluster centroids weekly
