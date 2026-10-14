"""Drilling physics modules for DrillOpt-Engine.

Submodules:
    - mse: Mechanical Specific Energy (Teale, Dupriest) and mechanical efficiency.
    - d_exponent: D-exponent, corrected Dc-exponent, and pore pressure detection.
    - hydraulics: Bit TFA, jet velocity, bit pressure drop, HSI, and annular velocity.
"""

from .mse import (
    dupriest_mse,
    mechanical_efficiency,
    mpa_to_psi,
    over_energy_ratio,
    psi_to_mpa,
    teale_mse,
)
from .d_exponent import (
    detect_pore_pressure_anomaly,
    eaton_pore_pressure,
    fit_normal_compaction_trend,
    jordan_shirley_d_exponent,
    rehm_mcclendon_dc_exponent,
)
from .hydraulics import (
    annular_velocity,
    bit_pressure_drop,
    hydraulic_horsepower_per_sq_in,
    jet_impact_force,
    nozzle_tfa,
    nozzle_velocity,
)

__all__ = [
    # MSE
    "teale_mse",
    "dupriest_mse",
    "mechanical_efficiency",
    "over_energy_ratio",
    "psi_to_mpa",
    "mpa_to_psi",
    # D-exponent
    "jordan_shirley_d_exponent",
    "rehm_mcclendon_dc_exponent",
    "fit_normal_compaction_trend",
    "detect_pore_pressure_anomaly",
    "eaton_pore_pressure",
    # Hydraulics
    "nozzle_tfa",
    "nozzle_velocity",
    "bit_pressure_drop",
    "hydraulic_horsepower_per_sq_in",
    "jet_impact_force",
    "annular_velocity",
]
