"""Run all 20 quant-agent technique demos.
Usage: cd quant-agent && python demos/run_all_demos.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant_agent.techniques import (
    manifold_tube, marchenko_pastur, shape_correlation,
    esscher_tilt, alpha_divergence, curvature_throttle,
    grassmann_rotation, correlation_clustering, path_speed,
    option_density, ridge_precision, partial_correlation,
    factor_covariance, factor_rotation, regime_backtest,
    curvature_penalized_fit, geodesic_macro_regression,
    js_edge_persistence, chernoff_classifier, volume_profile_manifold,
)
print("=" * 60)
print("  Quant Agent — Running all 20 technique demos")
print("=" * 60)
manifold_tube.demo()
marchenko_pastur.demo()
shape_correlation.demo()
esscher_tilt.demo()
alpha_divergence.demo()
curvature_throttle.demo()
grassmann_rotation.demo()
correlation_clustering.demo()
path_speed.demo()
option_density.demo()
ridge_precision.demo()
partial_correlation.demo()
factor_covariance.demo()
factor_rotation.demo()
regime_backtest.demo()
curvature_penalized_fit.demo()
geodesic_macro_regression.demo()
js_edge_persistence.demo()
chernoff_classifier.demo()
volume_profile_manifold.demo()
print("=" * 60)
print("  All demos complete. Check demos/output_*.png")
print("=" * 60)
