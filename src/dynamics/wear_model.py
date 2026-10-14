"""PDC cutter wear modeling based on mechanical energy dissipation.

Formulations:
- Cumulative dissipated frictional energy:
    E_dissipated = integral( max(0, MSE - UCS) * dV )
  where dV = A_b * dDepth.
- PDC cutter dull grading estimation on the standard IADC 0 to 8 scale.
- Remaining Useful Life (RUL) estimation in footage and dissipated energy.

References:
    - Glowka, D. A. (1989). Use of Single-Cutter Data in the Analysis of PDC Bit Designs:
      Part 1—Development of a PDC Cutting Force Model. JPT, 41(08), 850-859.
    - Pessier, R. C., & Damschen, M. S. (1993). Hybrid Bits Offer Extended Life in
      Abrasive Formations. SPE/IADC 25734.
    - IADC Dull Grading System for Fixed Cutter (PDC) Bits.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd

FT_LBF_TO_JOULES = 1.3558179483314004
FT_LBF_TO_MJ = 1.3558179483314004e-6


@dataclass
class CutterWearState:
    """State of PDC bit cutter wear."""
    cumulative_energy_dissipated_ft_lbf: float
    cumulative_energy_dissipated_mj: float
    wear_grade_continuous: float  # 0.0 to 8.0
    iadc_wear_grade_integer: int  # 0 to 8
    remaining_energy_capacity_mj: float
    remaining_footage_estimate: Optional[float]
    wear_condition: str  # 'New', 'Light', 'Moderate', 'Worn', 'Dull / Pulled'


def compute_cumulative_dissipated_energy(
    mse: Union[np.ndarray, pd.Series],
    ucs: Union[np.ndarray, pd.Series, float, int],
    depth: Union[np.ndarray, pd.Series],
    bit_diameter: Optional[float] = None,
    bit_area: Optional[float] = None,
    energy_unit: str = "ft-lbf",
) -> Union[np.ndarray, pd.Series]:
    """Calculate cumulative mechanical energy dissipated above rock UCS.

    Energy required for cutting is UCS * Volume.
    Any specific energy spent above UCS represents frictional dissipation,
    vibration losses, and cutter abrasive wear:
        Delta_E_i = max(0, MSE_i - UCS_i) * A_b * Delta_depth_i

    When MSE and UCS are in psi (lbf/in^2), bit_area A_b in in^2,
    and Delta_depth in feet:
        Force = (MSE - UCS) * A_b  [lbf]
        Work  = Force * Delta_depth [ft-lbf]

    Args:
        mse: Mechanical specific energy series (psi).
        ucs: Rock unconfined compressive strength series or scalar (psi).
        depth: Measured well depth series (feet).
        bit_diameter: Bit outer diameter in inches (used if bit_area is None).
        bit_area: Bit cross-sectional area in in^2.
        energy_unit: Desired energy unit ('ft-lbf', 'MJ', or 'J'). Default is 'ft-lbf'.

    Returns:
        Cumulative dissipated energy series (starting at 0).
    """
    if bit_area is None:
        if bit_diameter is None:
            raise ValueError("Either 'bit_diameter' or 'bit_area' must be provided.")
        area = (np.pi / 4.0) * (bit_diameter ** 2)
    else:
        area = float(bit_area)

    is_series = isinstance(mse, pd.Series) or isinstance(depth, pd.Series)
    idx_mse = getattr(mse, "index", None)
    idx_depth = getattr(depth, "index", None)
    series_index = idx_mse if idx_mse is not None else idx_depth

    mse_arr = np.asarray(mse, dtype=float)
    ucs_arr = np.asarray(ucs, dtype=float)
    z_arr = np.asarray(depth, dtype=float)

    n = len(mse_arr)
    if n == 0:
        return pd.Series([], dtype=float) if is_series else np.array([], dtype=float)

    # Compute delta depth
    delta_z = np.zeros(n, dtype=float)
    if n > 1:
        diffs = np.diff(z_arr)
        # Avoid negative depth increments if pipe is pulled
        diffs = np.maximum(diffs, 0.0)
        delta_z[1:] = diffs

    # Over-energy: max(0, MSE - UCS)
    excess_energy_density = np.maximum(0.0, mse_arr - ucs_arr)

    # Incremental dissipated energy in ft-lbf: (psi * in^2 * ft = lbf * ft = ft-lbf)
    delta_energy_ft_lbf = excess_energy_density * area * delta_z
    cum_energy_ft_lbf = np.cumsum(delta_energy_ft_lbf)

    if energy_unit.lower() == "ft-lbf":
        result = cum_energy_ft_lbf
    elif energy_unit.lower() == "mj":
        result = cum_energy_ft_lbf * FT_LBF_TO_MJ
    elif energy_unit.lower() == "j":
        result = cum_energy_ft_lbf * FT_LBF_TO_JOULES
    else:
        raise ValueError(f"Unsupported energy_unit '{energy_unit}'. Use 'ft-lbf', 'MJ', or 'J'.")

    if is_series:
        return pd.Series(result, index=series_index)
    return result


def estimate_cutter_wear_grade(
    cumulative_dissipated_energy: Union[float, np.ndarray, pd.Series],
    energy_capacity: float = 1.0e8,
    exponent: float = 1.0,
    energy_unit: str = "ft-lbf",
) -> Union[Tuple[float, int], Tuple[np.ndarray, np.ndarray]]:
    """Estimate PDC cutter wear grade on the standard IADC 0 to 8 scale.

    Formulation:
        wear_continuous = 8.0 * min(1.0, (E_dissipated / E_capacity)^exponent)
        wear_iadc       = round(wear_continuous)   in {0, 1, ..., 8}

    IADC Dull Grading:
        - 0: No wear (pristine cutters)
        - 1-2: Light wear (chamfer worn / minor diamond table flat)
        - 3-5: Moderate wear (normal progression)
        - 6-7: Severe wear (diamond table delamination / severe flat)
        - 8: 100% dull / total diamond loss

    Args:
        cumulative_dissipated_energy: Dissipated energy accumulated by bit.
        energy_capacity: Total dissipated energy required to reach terminal wear (grade 8).
            Default is 1.0e8 ft-lbf (~135.6 MJ).
        exponent: Wear progression shape factor (1.0 = linear Archard wear).
        energy_unit: 'ft-lbf' or 'MJ'. If 'MJ', energy_capacity should also be in MJ.

    Returns:
        For scalar input: Tuple of (continuous_wear_grade, integer_iadc_grade).
        For array/Series input: Tuple of (continuous_wear_array, integer_iadc_array).
    """
    if energy_capacity <= 0:
        raise ValueError("energy_capacity must be strictly positive.")

    is_series = isinstance(cumulative_dissipated_energy, pd.Series)
    series_index = getattr(cumulative_dissipated_energy, "index", None)
    is_scalar = not isinstance(
        cumulative_dissipated_energy, (np.ndarray, pd.Series, list, tuple)
    )

    e_arr = np.asarray(cumulative_dissipated_energy, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.maximum(0.0, e_arr) / float(energy_capacity)
        progression = np.power(np.minimum(1.0, ratio), exponent)
        wear_cont = np.where(np.isnan(e_arr), np.nan, 8.0 * progression)
        valid = np.isfinite(wear_cont)
        safe_round = np.round(np.where(valid, wear_cont, 0.0)).astype(int)
        wear_int = np.where(valid, np.clip(safe_round, 0, 8), -1)

    if is_scalar:
        if np.isnan(wear_cont):
            return float("nan"), -1
        return float(wear_cont), int(wear_int)
    if is_series:
        return pd.Series(wear_cont, index=series_index), pd.Series(wear_int, index=series_index)
    return wear_cont, wear_int


def evaluate_bit_wear_state(
    cumulative_energy_ft_lbf: float,
    energy_capacity_ft_lbf: float = 1.0e8,
    drilled_depth_ft: Optional[float] = None,
) -> CutterWearState:
    """Evaluate full cutter wear state and estimate remaining footage life.

    Args:
        cumulative_energy_ft_lbf: Total energy dissipated to date in ft-lbf.
        energy_capacity_ft_lbf: Total bit capacity in ft-lbf before Grade 8.
        drilled_depth_ft: Total interval footage drilled by this bit so far.

    Returns:
        CutterWearState dataclass.
    """
    w_cont, w_int = estimate_cutter_wear_grade(
        cumulative_dissipated_energy=cumulative_energy_ft_lbf,
        energy_capacity=energy_capacity_ft_lbf,
        energy_unit="ft-lbf",
    )

    e_mj = cumulative_energy_ft_lbf * FT_LBF_TO_MJ
    cap_mj = energy_capacity_ft_lbf * FT_LBF_TO_MJ
    rem_mj = max(0.0, cap_mj - e_mj)

    rem_footage = None
    if drilled_depth_ft is not None and drilled_depth_ft > 0 and cumulative_energy_ft_lbf > 0:
        energy_per_ft = cumulative_energy_ft_lbf / drilled_depth_ft
        rem_energy_ft_lbf = max(0.0, energy_capacity_ft_lbf - cumulative_energy_ft_lbf)
        rem_footage = rem_energy_ft_lbf / energy_per_ft if energy_per_ft > 0 else None

    # Condition string
    if w_int == 0:
        cond = "New"
    elif w_int <= 2:
        cond = "Light"
    elif w_int <= 5:
        cond = "Moderate"
    elif w_int <= 7:
        cond = "Worn"
    else:
        cond = "Dull / Pulled"

    return CutterWearState(
        cumulative_energy_dissipated_ft_lbf=cumulative_energy_ft_lbf,
        cumulative_energy_dissipated_mj=e_mj,
        wear_grade_continuous=float(w_cont),
        iadc_wear_grade_integer=int(w_int),
        remaining_energy_capacity_mj=rem_mj,
        remaining_footage_estimate=rem_footage,
        wear_condition=cond,
    )
