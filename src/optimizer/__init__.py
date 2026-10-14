"""
Drilling Optimization & Parameter Road Mapping Package
=======================================================
Autonomous real-time and post-well drilling parameter optimization:
- Founder Point detection via piecewise linear regression & curvature analysis
- Safe operating parameter envelope (WOB, RPM, Flow)
- Stick-slip mitigation constraint (SSI < 0.5)
- Business ROI and financial cost savings calculator

Author: Tiago Meilan
Project: DrillOpt-Engine
"""

from .founder_point import FounderPointAnalyzer, FounderPointResult
from .road_map import RoadMapOptimizer, DrillingEnvelope, OptimizationResult

__all__ = [
    "FounderPointAnalyzer",
    "FounderPointResult",
    "RoadMapOptimizer",
    "DrillingEnvelope",
    "OptimizationResult",
]
