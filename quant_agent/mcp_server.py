"""
Quant Agent — MCP Server

Exposes all 10 information-geometric techniques as callable tools
for Claude Code. Run via: python quant_agent/mcp_server.py
"""

import json
import numpy as np
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from quant_agent.techniques import (
    manifold_tube,
    marchenko_pastur,
    shape_correlation,
    esscher_tilt,
    alpha_divergence,
    curvature_throttle,
    grassmann_rotation,
    correlation_clustering,
    path_speed,
    option_density,
)

app = Server("quant-agent")


def _parse_returns(payload: str | list) -> np.ndarray:
    if isinstance(payload, str):
        return np.array(json.loads(payload), dtype=float)
    return np.array(payload, dtype=float)


def _parse_matrix(payload: str | list) -> np.ndarray:
    if isinstance(payload, str):
        return np.array(json.loads(payload), dtype=float)
    return np.array(payload, dtype=float)


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="manifold_tube",
            description=(
                "Technique 1: Manifold Tube — Signal Stability Monitor. "
                "Treats rolling-calibrated Gaussian parameters as a path on the statistical manifold. "
                "Returns δt (distance-to-path), in-tube flag, trade signal, and position size for each day. "
                "Use to detect when a signal model has structurally drifted."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "returns": {"type": "array", "items": {"type": "number"},
                                "description": "1-D array of daily returns"},
                    "window":  {"type": "integer", "default": 60,
                                "description": "Rolling calibration window (days)"},
                    "epsilon": {"type": "number", "default": 0.15,
                                "description": "Tube radius in Fisher-Rao units"},
                    "size_max": {"type": "number", "default": 1.0,
                                 "description": "Maximum position size"},
                },
                "required": ["returns"],
            },
        ),
        Tool(
            name="marchenko_pastur_denoise",
            description=(
                "Technique 2: Marchenko-Pastur Denoising. "
                "Separates genuine correlation signal from sampling noise using Random Matrix Theory. "
                "Returns cleaned correlation matrix, noise fraction removed, and number of signal eigenvalues. "
                "Use before portfolio optimisation or factor analysis."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "returns": {"type": "array",
                                "description": "2-D array (T, n) of daily returns — list of lists"},
                },
                "required": ["returns"],
            },
        ),
        Tool(
            name="shape_correlation",
            description=(
                "Technique 3: Shape Correlation. "
                "Maps windowed return histograms to the unit sphere via square-root transform. "
                "Computes Hellinger distances and shape correlations that capture tail/skew co-movement "
                "missed by linear correlation. Returns shape_corr and linear_corr matrices."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "returns":      {"type": "array", "description": "2-D array (T, n) of daily returns"},
                    "asset_names":  {"type": "array", "items": {"type": "string"}},
                    "window":       {"type": "integer", "default": 60},
                    "n_bins":       {"type": "integer", "default": 30},
                },
                "required": ["returns", "asset_names"],
            },
        ),
        Tool(
            name="esscher_tilt",
            description=(
                "Technique 4: Esscher Tilt / Crash Premium. "
                "Finds the minimum-entropy change of measure Q_θ that matches option-implied moments. "
                "Returns θ* (crash premium), tilted moments, KL divergence from historical law. "
                "Use to quantify how much tail fear is embedded in the options market."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "returns":      {"type": "array", "description": "1-D array of historical daily returns"},
                    "target_skew":  {"type": "number", "default": -1.5,
                                     "description": "Option-implied target skew (negative = crash fear)"},
                    "target_var":   {"type": "number", "description": "Option-implied variance (optional)"},
                    "rolling":      {"type": "boolean", "default": False,
                                     "description": "If true, return rolling crash premium time series"},
                    "window":       {"type": "integer", "default": 60},
                },
                "required": ["returns"],
            },
        ),
        Tool(
            name="alpha_divergence_alarm",
            description=(
                "Technique 5: α-Divergence Decay Alarm. "
                "Compares model predictive distributions to realised return distributions. "
                "Rising D_α signals calibration drift, regime change, or execution slippage. "
                "Returns divergence series, alarm flags, and risk throttle multiplier."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "pred_mu":    {"type": "array", "description": "1-D array of predicted means"},
                    "pred_sigma": {"type": "array", "description": "1-D array of predicted std devs"},
                    "outcomes":   {"type": "array", "description": "1-D array of realised returns"},
                    "alpha":      {"type": "number", "default": 0.5,
                                   "description": "α ∈ (0,1); closer to 0 = KL(q‖p), closer to 1 = KL(p‖q)"},
                    "tau":        {"type": "number", "default": 0.05, "description": "Alarm threshold"},
                    "window":     {"type": "integer", "default": 40},
                    "c":          {"type": "number", "default": 10.0, "description": "Throttle steepness"},
                },
                "required": ["pred_mu", "pred_sigma", "outcomes"],
            },
        ),
        Tool(
            name="curvature_throttle",
            description=(
                "Technique 6: Statistical Curvature Throttle. "
                "Computes Fisher curvature κ(θ) of the fitted Gaussian model. "
                "High curvature means small misspecification causes large prediction errors. "
                "Returns κ per day and risk weight α_fit = 1/(1+c·κ) as a continuous lever."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "returns": {"type": "array", "description": "1-D daily returns"},
                    "window":  {"type": "integer", "default": 60},
                    "c":       {"type": "number", "default": 2.0, "description": "Throttle steepness"},
                },
                "required": ["returns"],
            },
        ),
        Tool(
            name="grassmann_rotation",
            description=(
                "Technique 7: Grassmann Rotation — Risk Subspace Detector. "
                "Tracks top-k PCA eigenvectors on the Grassmann manifold using principal angles. "
                "Large rotation Θt = ‖principal angles‖ signals factor crowding shifts or regime breaks. "
                "Returns rotation score, per-component angles, and alarm flags."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "returns": {"type": "array", "description": "2-D array (T, n) of daily returns"},
                    "k":       {"type": "integer", "default": 3,
                                "description": "Number of principal components to track"},
                    "window":  {"type": "integer", "default": 60},
                    "tau":     {"type": "number", "default": 0.3,
                                "description": "Rotation alarm threshold (radians)"},
                },
                "required": ["returns"],
            },
        ),
        Tool(
            name="correlation_clustering",
            description=(
                "Technique 8: Correlation Manifold Clustering. "
                "Maps rolling correlation matrices into log-Euclidean space and clusters them. "
                "Discovers correlation regimes (risk-on, risk-off, sector-coupled, fragmentation) "
                "without distorting SPD geometry. Returns regime labels and centroid matrices."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "returns": {"type": "array", "description": "2-D array (T, n) of daily returns"},
                    "K":       {"type": "integer", "default": 3, "description": "Number of regimes"},
                    "window":  {"type": "integer", "default": 60},
                    "step":    {"type": "integer", "default": 5,
                                "description": "Stride between windows"},
                },
                "required": ["returns"],
            },
        ),
        Tool(
            name="path_speed",
            description=(
                "Technique 9: Distribution Path Speed & Acceleration. "
                "Treats rolling Gaussian fits as a path on the statistical manifold. "
                "Speed (Fisher-Rao distance per step) and acceleration detect regime transitions. "
                "Triangle-defect curvature detects sharp bends / structural snaps in the path."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "returns":    {"type": "array", "description": "1-D daily returns"},
                    "window":     {"type": "integer", "default": 40},
                    "tau_speed":  {"type": "number", "default": 0.15},
                    "tau_curve":  {"type": "number", "default": -0.05},
                },
                "required": ["returns"],
            },
        ),
        Tool(
            name="option_density_manifold",
            description=(
                "Technique 10: Option-Implied Density Manifold. "
                "Extracts risk-neutral densities from the implied-vol surface via Breeden-Litzenberger. "
                "Computes Fisher-Rao distance to a historical density library to detect skew/tail regime shifts. "
                "Returns FR distances, alarm flags, and risk throttle."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "atm_vols":   {"type": "array", "description": "1-D array of ATM implied vols"},
                    "skews":      {"type": "array", "description": "1-D array of smile skew parameters"},
                    "kurt_adjs":  {"type": "array", "description": "1-D array of smile curvature params"},
                    "F":          {"type": "number", "default": 100.0, "description": "Forward price"},
                    "tau":        {"type": "number", "default": 0.15, "description": "FR alarm threshold"},
                    "c":          {"type": "number", "default": 8.0, "description": "Throttle steepness"},
                },
                "required": ["atm_vols", "skews", "kurt_adjs"],
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:

    def _txt(obj) -> list[TextContent]:
        return [TextContent(type="text", text=json.dumps(obj, indent=2))]

    # ---- 1. Manifold Tube ----
    if name == "manifold_tube":
        r = _parse_returns(arguments["returns"])
        res = manifold_tube.run(
            r,
            window=arguments.get("window", 60),
            epsilon=arguments.get("epsilon", 0.15),
            size_max=arguments.get("size_max", 1.0),
        )
        return _txt({
            "tube_exits":    int((res.signal != 0).sum()),
            "total_days":    len(res.signal),
            "mean_delta":    float(res.delta.mean()),
            "max_delta":     float(res.delta.max()),
            "signals":       res.signal.tolist(),
            "sizes":         res.size.tolist(),
            "delta":         res.delta.tolist(),
        })

    # ---- 2. Marchenko-Pastur ----
    elif name == "marchenko_pastur_denoise":
        r = _parse_matrix(arguments["returns"])
        res = marchenko_pastur.denoise(r)
        return _txt({
            "clean_correlation":   res.clean_corr.tolist(),
            "lambda_plus":         res.lambda_plus,
            "lambda_minus":        res.lambda_minus,
            "n_signal_eigenvalues": res.n_signal,
            "noise_removed_pct":   round(res.explained_noise_pct, 2),
            "sigma_sq":            res.sigma_sq,
        })

    # ---- 3. Shape Correlation ----
    elif name == "shape_correlation":
        r = _parse_matrix(arguments["returns"])
        names = arguments["asset_names"]
        res = shape_correlation.run(
            r, names,
            window=arguments.get("window", 60),
            n_bins=arguments.get("n_bins", 30),
        )
        return _txt({
            "shape_correlation":  res.shape_corr.tolist(),
            "linear_correlation": res.linear_corr.tolist(),
            "hellinger_distance": res.hellinger.tolist(),
            "assets":             names,
        })

    # ---- 4. Esscher Tilt ----
    elif name == "esscher_tilt":
        r = _parse_returns(arguments["returns"])
        if arguments.get("rolling", False):
            prem = esscher_tilt.rolling_crash_premium(
                r,
                window=arguments.get("window", 60),
                target_skew=arguments.get("target_skew", -1.5),
            )
            return _txt({"rolling_crash_premium": prem.tolist()})
        res = esscher_tilt.calibrate(
            r,
            target_skew=arguments.get("target_skew", -1.5),
            target_var=arguments.get("target_var", None),
        )
        return _txt({
            "theta_star":       res.theta_star,
            "crash_premium":    res.crash_premium,
            "tilted_mean":      res.tilted_mean,
            "tilted_var":       res.tilted_var,
            "tilted_skew":      res.tilted_skew,
            "historical_skew":  res.historical_skew,
            "kl_distance":      res.kl_distance,
        })

    # ---- 5. α-Divergence ----
    elif name == "alpha_divergence_alarm":
        mu    = _parse_returns(arguments["pred_mu"])
        sigma = _parse_returns(arguments["pred_sigma"])
        out   = _parse_returns(arguments["outcomes"])
        pred_params = np.column_stack([mu, sigma])
        res = alpha_divergence.run(
            pred_params, out,
            window=arguments.get("window", 40),
            alpha=arguments.get("alpha", 0.5),
            tau=arguments.get("tau", 0.05),
            c=arguments.get("c", 10.0),
        )
        return _txt({
            "alarm_days":       int(res.alarm.sum()),
            "total_days":       len(res.alarm),
            "divergence":       res.divergence.tolist(),
            "risk_multiplier":  res.risk_multiplier.tolist(),
            "alarm":            res.alarm.tolist(),
        })

    # ---- 6. Curvature Throttle ----
    elif name == "curvature_throttle":
        r = _parse_returns(arguments["returns"])
        res = curvature_throttle.run(
            r,
            window=arguments.get("window", 60),
            c=arguments.get("c", 2.0),
        )
        return _txt({
            "kappa":      res.kappa.tolist(),
            "alpha_fit":  res.alpha_fit.tolist(),
            "w_risk":     res.w_risk.tolist(),
            "peak_kappa": float(res.kappa.max()),
            "min_w_risk": float(res.w_risk.min()),
        })

    # ---- 7. Grassmann Rotation ----
    elif name == "grassmann_rotation":
        r = _parse_matrix(arguments["returns"])
        res = grassmann_rotation.run(
            r,
            k=arguments.get("k", 3),
            window=arguments.get("window", 60),
            tau=arguments.get("tau", 0.3),
        )
        return _txt({
            "rotation_score": res.rotation_score.tolist(),
            "alarm_days":     int(res.alarm.sum()),
            "peak_rotation":  float(res.rotation_score.max()),
            "alarm":          res.alarm.tolist(),
        })

    # ---- 8. Correlation Clustering ----
    elif name == "correlation_clustering":
        r = _parse_matrix(arguments["returns"])
        res = correlation_clustering.run(
            r,
            window=arguments.get("window", 60),
            step=arguments.get("step", 5),
            K=arguments.get("K", 3),
        )
        return _txt({
            "labels":          res.labels.tolist(),
            "times":           res.times.tolist(),
            "centroids_corr":  res.centroids_corr.tolist(),
            "K":               res.K,
            "inertia":         res.inertia,
        })

    # ---- 9. Path Speed ----
    elif name == "path_speed":
        r = _parse_returns(arguments["returns"])
        res = path_speed.run(
            r,
            window=arguments.get("window", 40),
            tau_speed=arguments.get("tau_speed", 0.15),
            tau_curve=arguments.get("tau_curve", -0.05),
        )
        return _txt({
            "speed":            res.speed.tolist(),
            "acceleration":     res.acceleration.tolist(),
            "curvature":        res.curvature.tolist(),
            "speed_alarms":     int(res.alarm_speed.sum()),
            "curvature_alarms": int(res.alarm_curve.sum()),
        })

    # ---- 10. Option Density ----
    elif name == "option_density_manifold":
        atm   = _parse_returns(arguments["atm_vols"])
        skews = _parse_returns(arguments["skews"])
        kurts = _parse_returns(arguments["kurt_adjs"])
        vol_params = np.column_stack([atm, skews, kurts])
        res = option_density.run(
            vol_params,
            F=arguments.get("F", 100.0),
            tau=arguments.get("tau", 0.15),
            c=arguments.get("c", 8.0),
        )
        return _txt({
            "fr_distance":     res.fr_distance.tolist(),
            "alarm":           res.alarm.tolist(),
            "alarm_days":      int(res.alarm.sum()),
            "risk_multiplier": res.risk_multiplier.tolist(),
            "peak_fr_dist":    float(res.fr_distance.max()),
        })

    else:
        return _txt({"error": f"Unknown tool: {name}"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
