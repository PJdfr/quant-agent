Apply **Technique 15: Regime-Local Backtesting Protocol** to the provided signal and returns.

Steps:
1. Define the regime S by observable state variables (rolling vol, session, liquidity, etc.)
2. Call `regime_backtest` MCP tool with returns, signal predictions, and regime definition
3. Report skill score, calibration error, recurrence stability across visits
4. Evaluate evidential weight: how many of the 8 checks pass?

Key interpretation:
- Skill > 0 → signal beats regime-local baseline → genuine predictive value inside S
- Calibrated → stated probabilities/quantiles match realized rates → not overfit
- Recurrence stable → score consistent across repeated regime visits → not lucky
- Evidential weight = 1.0 → all 4 checks pass → high-conviction signal within this regime
- Evidential weight < 0.5 → fewer than 2 checks pass → do not rely on signal in this regime

8-step checklist:
  ✓ 1. Regime declared by observables (no look-ahead)
  ✓ 2. Prediction object fixed (quantile / density / event prob)
  ✓ 3. Proper scoring rule matched to prediction type
  ✓ 4. Baseline built regime-locally (not globally)
  ✓ 5. Evaluated only on held-out timestamps inside S
  ✓ 6. Recurrence stability checked across visits
  ✓ 7. Local calibration verified
  ✓ 8. Evidential weight assigned conditional on all checks
