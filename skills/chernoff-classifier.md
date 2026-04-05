Apply **Technique 19: Chernoff Information Classifier** to set principled regime thresholds.

Steps:
1. Define calm (H0) and stress (H1) distributions from historical data
2. Call `chernoff_classifier` MCP tool with observations, p0_params, p1_params
3. Report C (Chernoff information), s* (optimal tilt), Bayes error rate
4. Use tau_exit and tau_entry thresholds for entry/exit rules

Key interpretation:
- C large (>0.5): distributions well-separated, classifier reliable
- C small (<0.1): distributions overlap, classification unreliable
- Bayes error approx exp(-C): theoretical floor on misclassification rate
- LLR > tau_exit: in stress, exit / de-lever
- LLR < tau_entry: clearly calm, allow full exposure
- s* != 0.5: optimal boundary is asymmetric
- Update p0/p1 params periodically using rolling window estimates
