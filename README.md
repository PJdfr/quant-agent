# Quant Agent

Information-geometric techniques for quantitative finance, packaged as a Claude Code agent.

## Techniques

| # | Technique | MCP Tool | One-liner |
|---|-----------|----------|-----------|
| 1 | Manifold Tube | `manifold_tube` | Is my signal model still on-path? |
| 2 | Marchenko-Pastur | `marchenko_pastur_denoise` | Which correlations are real? |
| 3 | Shape Correlation | `shape_correlation` | Do assets share tail/skew behaviour? |
| 4 | Esscher Tilt | `esscher_tilt` | What crash premium does the options market price? |
| 5 | α-Divergence Alarm | `alpha_divergence_alarm` | Is my model still describing reality? |
| 6 | Curvature Throttle | `curvature_throttle` | How much should I trust my model? |
| 7 | Grassmann Rotation | `grassmann_rotation` | Has the risk basis rotated? |
| 8 | Correlation Clustering | `correlation_clustering` | Which correlation regime are we in? |
| 9 | Path Speed | `path_speed` | Is the market transitioning or stable? |
| 10 | Option Density | `option_density_manifold` | Has the implied skew/tail structure shifted? |

## Install (one command)

```bash
git clone https://github.com/yourorg/quant-agent
cd quant-agent
chmod +x install.sh && ./install.sh
```

This installs Python dependencies, copies skills to `~/.claude/skills/`, and registers the MCP server.

## Usage in Claude Code

**Invoke the agent:**
```
/quant-agent  analyse these returns using the manifold tube approach
```

**Use a specific technique:**
```
/manifold-tube
/marchenko-pastur
/shape-correlation
/esscher-tilt
/alpha-divergence
/curvature-throttle
/grassmann-rotation
/correlation-clustering
/path-speed
/option-density
```

**Or just describe what you want:**
```
@quant-agent  is the equity factor still stable? our PnL looks noisy lately
@quant-agent  run a full regime check on these 10 assets
@quant-agent  what is the options market implying about crash risk right now?
```

## Run demos

```bash
cd quant-agent
python demos/run_all_demos.py
```

Generates `demos/output_01_*.png` through `demos/output_10_*.png` with synthetic but realistic data.

## Repo structure

```
quant-agent/
├── CLAUDE.md                        # Agent knowledge base (auto-loaded)
├── install.sh                       # One-command team setup
├── requirements.txt
├── quant_agent/
│   ├── mcp_server.py                # MCP server (exposes tools to Claude)
│   └── techniques/
│       ├── manifold_tube.py         # Technique 1
│       ├── marchenko_pastur.py      # Technique 2
│       ├── shape_correlation.py     # Technique 3
│       ├── esscher_tilt.py          # Technique 4
│       ├── alpha_divergence.py      # Technique 5
│       ├── curvature_throttle.py    # Technique 6
│       ├── grassmann_rotation.py    # Technique 7
│       ├── correlation_clustering.py # Technique 8
│       ├── path_speed.py            # Technique 9
│       └── option_density.py        # Technique 10
├── skills/                          # Claude Code slash commands
│   ├── quant-agent.md               # /quant-agent
│   ├── manifold-tube.md             # /manifold-tube
│   └── ...
└── demos/
    └── run_all_demos.py             # Synthetic data demonstrations
```

## Dependencies

- `numpy`, `scipy`, `scikit-learn`, `matplotlib`
- `mcp` (Anthropic MCP Python SDK)
