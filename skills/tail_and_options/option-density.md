Apply **Technique 10: Option-Implied Density Manifold** to the provided implied-vol surface data.

Steps:
1. Call `option_density_manifold` MCP tool with atm_vols, skews, kurt_adjs arrays
2. Report FR distance to historical library, alarm dates, and risk multiplier
3. Compare today's risk-neutral density shape to the nearest historical date
4. State implications for vol positioning, skew trading, and overall risk limits

Key interpretation:
- FR distance low → current implied density is similar to historical norms → normal regime
- FR distance spike → skew/tail shape changed structurally → investigate options positioning
- Alarm triggered → consider buying puts, widening VaR limits, or reducing delta exposure
- Risk multiplier low → auto-throttle vol positions
- Use in combination with Esscher tilt (θ*) for a complete options risk picture:
  θ* gives the magnitude, FR distance gives the structural shape change
