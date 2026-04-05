Apply **Technique 18: JS Divergence Edge Persistence** to monitor signal edge decay.

Steps:
1. Call `js_edge_persistence` MCP tool with returns, state_labels, tau_edge
2. Track ε_s(t) = √D_JS(p_s, p_base) over time for each signal state
3. Alarm when ε_s < τ_edge → conditional distribution collapsing toward baseline → edge dying
4. Check across-state confusion C_{s,s'} → are states becoming indistinguishable?

Key interpretation:
- ε_s high and stable → strong, persistent edge — keep trading
- ε_s declining → edge decaying → reduce size or stop trading this signal
- ε_s < τ_edge → edge geometrically dead → halt signal
- C_{s,s'} low → states indistinguishable → regime model is breaking down
- Use in combination with Technique 5 (α-divergence alarm) for full signal health check
