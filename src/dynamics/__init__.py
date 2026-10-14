"""Drilling dynamics modules for DrillOpt-Engine.

Submodules:
    - stick_slip: Stick-Slip Severity Index (SSI), torsional resonance f0 = c_s / (4L),
      spectral FFT and Welch PSD analysis.
    - bit_bounce: Axial vibration index, rolling dispersion of WOB, and shock detection.
    - wear_model: Mechanical energy dissipation, PDC cutter wear grading (0 to 8 IADC scale).
"""

from .stick_slip import (
    SpectralAnalysisResult,
    analyze_torsional_fft,
    analyze_torsional_welch,
    check_resonance_risk,
    classify_stick_slip_severity,
    compute_rolling_stick_slip_index,
    compute_stick_slip_index,
    drillstring_torsional_natural_frequency,
)
from .bit_bounce import (
    AxialShockSummary,
    classify_bit_bounce_severity,
    compute_axial_vibration_index,
    detect_axial_shocks,
)
from .wear_model import (
    CutterWearState,
    compute_cumulative_dissipated_energy,
    estimate_cutter_wear_grade,
    evaluate_bit_wear_state,
)

__all__ = [
    # Stick-Slip & Torsional
    "compute_stick_slip_index",
    "compute_rolling_stick_slip_index",
    "classify_stick_slip_severity",
    "drillstring_torsional_natural_frequency",
    "analyze_torsional_fft",
    "analyze_torsional_welch",
    "check_resonance_risk",
    "SpectralAnalysisResult",
    # Bit Bounce & Axial
    "compute_axial_vibration_index",
    "classify_bit_bounce_severity",
    "detect_axial_shocks",
    "AxialShockSummary",
    # Wear Model
    "compute_cumulative_dissipated_energy",
    "estimate_cutter_wear_grade",
    "evaluate_bit_wear_state",
    "CutterWearState",
]
