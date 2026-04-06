Apply **Technique 5: α-Divergence Decay Alarm** to the provided predictions and outcomes.

Steps:
1. Call `alpha_divergence_alarm` MCP tool with pred_mu, pred_sigma, outcomes arrays
2. Report divergence series, alarm day count, and minimum risk multiplier reached
3. Identify when the alarm was triggered and what happened to returns around that time
4. Recommend: recalibrate model, reduce risk, or pause strategy

Key interpretation:
- D_α rising steadily → model slowly drifting from reality (recalibrate)
- D_α spike suddenly → regime change or data anomaly (investigate)
- Risk multiplier < 0.5 → cut position size in half automatically
- Choose α: higher α = more sensitive to model's high-probability region errors
           lower α = more sensitive to tail misses
