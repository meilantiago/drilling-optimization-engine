"""Drilling hydraulics calculations.

Formulations:
- Bit nozzle Total Flow Area (TFA) from 32nds of an inch sizes.
- Nozzle jet exit velocity (ft/s).
- Pressure drop across drill bit nozzles delta_P_bit (psi).
- Hydraulic Horsepower per square inch (HSI, hp/in^2).
- Jet impact force F_j (lbf).
- Annular fluid velocity (ft/min and ft/s).

References:
    - Bourgoyne, A. T., Millheim, K. K., Chenevert, M. E., & Young, F. S. (1986).
      Applied Drilling Engineering. SPE Textbook Series, Vol. 2.
    - API Recommended Practice 13D: Rheology and Hydraulics of Oil-well Drilling Fluids.
"""

from typing import Iterable, Optional, Sequence, Union
import numpy as np
import pandas as pd


def nozzle_tfa(
    nozzle_sizes: Union[Sequence[Union[float, int]], np.ndarray],
    in_32nds: bool = True,
) -> float:
    """Calculate Total Flow Area (TFA) in square inches from nozzle sizes.

    In standard drilling practice, nozzle orifice diameters are specified in
    32nds of an inch (e.g., [12, 12, 12] designates three 12/32" nozzles).

    Formula (in 32nds):
        TFA = sum( (pi / 4) * (size_i / 32)^2 ) = (pi / 4096) * sum(size_i^2)  [in^2]

    Formula (in inches):
        TFA = sum( (pi / 4) * d_i^2 )  [in^2]

    Args:
        nozzle_sizes: Collection of nozzle orifice sizes.
        in_32nds: If True (default), sizes are in 32nds of an inch.
            If False, sizes are in inches.

    Returns:
        Total Flow Area (TFA) in square inches (in^2).

    Raises:
        ValueError: If nozzle_sizes is empty or contains non-positive values.
    """
    sizes = np.asarray(nozzle_sizes, dtype=float)
    if len(sizes) == 0:
        raise ValueError("nozzle_sizes cannot be empty.")
    if np.any(sizes <= 0):
        raise ValueError("All nozzle sizes must be strictly positive.")

    if in_32nds:
        diameters = sizes / 32.0
    else:
        diameters = sizes

    areas = (np.pi / 4.0) * (diameters ** 2)
    return float(np.sum(areas))


def nozzle_velocity(
    flow_rate: Union[float, int, np.ndarray, pd.Series],
    tfa: Union[float, int, np.ndarray, pd.Series],
) -> Union[float, np.ndarray, pd.Series]:
    """Calculate nozzle jet exit velocity v_n (ft/s).

    Formula:
        v_n = (Q * 0.32086) / TFA   [ft/s]
            = Q / (3.117 * TFA)

    Standard Units:
        - flow_rate (Q): Mud circulation rate in gallons per minute (gpm).
        - tfa: Bit Total Flow Area in square inches (in^2).

    Args:
        flow_rate: Flow rate in gpm.
        tfa: Total Flow Area in in^2.

    Returns:
        Jet nozzle exit velocity in ft/s. Returns np.nan where TFA <= 0 or Q < 0.
    """
    is_series = isinstance(flow_rate, pd.Series) or isinstance(tfa, pd.Series)
    series_index = getattr(flow_rate, "index", None) or getattr(tfa, "index", None)
    is_scalar = not isinstance(flow_rate, (np.ndarray, pd.Series, list, tuple)) and not isinstance(
        tfa, (np.ndarray, pd.Series, list, tuple)
    )

    q_arr = np.asarray(flow_rate, dtype=float)
    tfa_arr = np.asarray(tfa, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        valid = (tfa_arr > 0) & (q_arr >= 0)
        # 1 gpm = 231 in^3 / min = (231 / 720) ft*in^2/s = 0.3208333 ft*in^2/s
        v_n = np.where(valid, (0.32086 * q_arr) / tfa_arr, np.nan)

    if is_scalar:
        return float(v_n) if not np.isnan(v_n) else float("nan")
    if is_series:
        return pd.Series(v_n, index=series_index)
    return v_n


def bit_pressure_drop(
    mud_weight: Union[float, int, np.ndarray, pd.Series],
    flow_rate: Union[float, int, np.ndarray, pd.Series],
    tfa: Union[float, int, np.ndarray, pd.Series],
    cd: float = 0.95,
) -> Union[float, np.ndarray, pd.Series]:
    """Calculate pressure drop across bit nozzles delta_P_bit (psi).

    Standard Orifice Formula (Bourgoyne et al., 1986):
        delta_P_bit = (mud_weight * Q^2) / (12032 * Cd^2 * TFA^2)  [psi]
        Note: When Cd = 0.95, 12032 * 0.95^2 = 10858.

    Standard Units:
        - mud_weight: Fluid density in pounds per gallon (ppg).
        - flow_rate (Q): Flow rate in gpm.
        - tfa: Total Flow Area in in^2.
        - cd: Discharge coefficient (dimensionless, typically 0.95 for PDC/roller cone).

    Args:
        mud_weight: Mud weight in ppg.
        flow_rate: Mud flow rate in gpm.
        tfa: Total Flow Area in in^2.
        cd: Orifice nozzle discharge coefficient (default 0.95).

    Returns:
        Bit nozzle pressure loss in psi.
    """
    is_series = any(isinstance(x, pd.Series) for x in (mud_weight, flow_rate, tfa))
    series_index = None
    for x in (mud_weight, flow_rate, tfa):
        if isinstance(x, pd.Series):
            series_index = x.index
            break

    is_scalar = not any(
        isinstance(x, (np.ndarray, pd.Series, list, tuple))
        for x in (mud_weight, flow_rate, tfa)
    )

    mw_arr = np.asarray(mud_weight, dtype=float)
    q_arr = np.asarray(flow_rate, dtype=float)
    tfa_arr = np.asarray(tfa, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        valid = (tfa_arr > 0) & (mw_arr > 0) & (q_arr >= 0) & (cd > 0)
        constant = 12032.0 * (cd ** 2)
        dp = np.where(valid, (mw_arr * (q_arr ** 2)) / (constant * (tfa_arr ** 2)), np.nan)

    if is_scalar:
        return float(dp) if not np.isnan(dp) else float("nan")
    if is_series:
        return pd.Series(dp, index=series_index)
    return dp


def hydraulic_horsepower_per_sq_in(
    delta_p_bit: Union[float, int, np.ndarray, pd.Series],
    flow_rate: Union[float, int, np.ndarray, pd.Series],
    bit_diameter: Optional[Union[float, int, np.ndarray, pd.Series]] = None,
    bit_area: Optional[Union[float, int, np.ndarray, pd.Series]] = None,
) -> Union[float, np.ndarray, pd.Series]:
    """Calculate Hydraulic Horsepower per square inch (HSI, hp/in^2).

    Measures the hydraulic power density delivered at the bottom of the hole
    to clean cuttings beneath the bit cutters.

    Formula:
        HHP_bit = (delta_P_bit * Q) / 1714       [hp]
        HSI     = HHP_bit / A_b                 [hp/in^2]
                = (delta_P_bit * Q) / (1714 * A_b)

    Typical PDC design optimum is between 2.5 and 5.0 HSI.

    Args:
        delta_p_bit: Bit pressure drop in psi.
        flow_rate: Circulation rate in gpm.
        bit_diameter: Bit diameter in inches (used to compute A_b = pi/4 * Db^2).
        bit_area: Bit cross-sectional area in in^2 (alternative to bit_diameter).

    Returns:
        HSI in hp/in^2.
    """
    if bit_area is None:
        if bit_diameter is None:
            raise ValueError("Either 'bit_diameter' or 'bit_area' must be provided.")
        area = (np.pi / 4.0) * (np.asarray(bit_diameter, dtype=float) ** 2)
    else:
        area = np.asarray(bit_area, dtype=float)

    is_series = any(isinstance(x, pd.Series) for x in (delta_p_bit, flow_rate))
    series_index = None
    for x in (delta_p_bit, flow_rate):
        if isinstance(x, pd.Series):
            series_index = x.index
            break

    is_scalar = not any(
        isinstance(x, (np.ndarray, pd.Series, list, tuple))
        for x in (delta_p_bit, flow_rate, area)
    )

    dp_arr = np.asarray(delta_p_bit, dtype=float)
    q_arr = np.asarray(flow_rate, dtype=float)
    area_arr = np.asarray(area, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        valid = (area_arr > 0) & (dp_arr >= 0) & (q_arr >= 0)
        hsi = np.where(valid, (dp_arr * q_arr) / (1714.0 * area_arr), np.nan)

    if is_scalar:
        return float(hsi) if not np.isnan(hsi) else float("nan")
    if is_series:
        return pd.Series(hsi, index=series_index)
    return hsi


def jet_impact_force(
    mud_weight: Union[float, int, np.ndarray, pd.Series],
    flow_rate: Union[float, int, np.ndarray, pd.Series],
    nozzle_vel: Optional[Union[float, int, np.ndarray, pd.Series]] = None,
    tfa: Optional[Union[float, int, np.ndarray, pd.Series]] = None,
) -> Union[float, np.ndarray, pd.Series]:
    """Calculate the total nozzle jet impact force F_j (lbf).

    Derived from momentum conservation:
        F_j = mass_flow_rate * v_n / g_c
            = (mud_weight * Q * v_n) / 1930.4   [lbf]
            = 0.000518 * mud_weight * Q * v_n

    Standard Units:
        - mud_weight: Fluid density in ppg.
        - flow_rate (Q): Flow rate in gpm.
        - nozzle_vel (v_n): Nozzle exit velocity in ft/s.
        - tfa: Total Flow Area in in^2 (used to compute v_n if nozzle_vel is None).

    Args:
        mud_weight: Mud weight in ppg.
        flow_rate: Circulation rate in gpm.
        nozzle_vel: Nozzle velocity in ft/s. If None, computed from flow_rate and tfa.
        tfa: Total Flow Area in in^2 (required if nozzle_vel is None).

    Returns:
        Jet impact force in pounds-force (lbf).
    """
    if nozzle_vel is None:
        if tfa is None:
            raise ValueError("Either 'nozzle_vel' or 'tfa' must be provided.")
        nozzle_vel = nozzle_velocity(flow_rate, tfa)

    is_series = any(isinstance(x, pd.Series) for x in (mud_weight, flow_rate, nozzle_vel))
    series_index = None
    for x in (mud_weight, flow_rate, nozzle_vel):
        if isinstance(x, pd.Series):
            series_index = x.index
            break

    is_scalar = not any(
        isinstance(x, (np.ndarray, pd.Series, list, tuple))
        for x in (mud_weight, flow_rate, nozzle_vel)
    )

    mw_arr = np.asarray(mud_weight, dtype=float)
    q_arr = np.asarray(flow_rate, dtype=float)
    vn_arr = np.asarray(nozzle_vel, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        valid = (mw_arr > 0) & (q_arr >= 0) & (vn_arr >= 0)
        # F_j = (rho * Q * v_n) / 1930.4
        fj = np.where(valid, (mw_arr * q_arr * vn_arr) / 1930.4, np.nan)

    if is_scalar:
        return float(fj) if not np.isnan(fj) else float("nan")
    if is_series:
        return pd.Series(fj, index=series_index)
    return fj


def annular_velocity(
    flow_rate: Union[float, int, np.ndarray, pd.Series],
    hole_diameter: Union[float, int, np.ndarray, pd.Series],
    pipe_outer_diameter: Union[float, int, np.ndarray, pd.Series],
    in_feet_per_minute: bool = True,
) -> Union[float, np.ndarray, pd.Series]:
    """Calculate drilling fluid annular velocity (ft/min or ft/s).

    Formula:
        v_a = (24.5 * Q) / (D_hole^2 - D_pipe^2)   [ft/min]
        v_a = Q / (2.448 * (D_hole^2 - D_pipe^2))  [ft/s]

    Args:
        flow_rate: Mud flow rate in gpm.
        hole_diameter: Wellbore or casing inner diameter in inches.
        pipe_outer_diameter: Drill pipe outer diameter in inches.
        in_feet_per_minute: If True (default), returns ft/min; if False, returns ft/s.

    Returns:
        Annular velocity in ft/min or ft/s.
    """
    is_series = any(
        isinstance(x, pd.Series)
        for x in (flow_rate, hole_diameter, pipe_outer_diameter)
    )
    series_index = None
    for x in (flow_rate, hole_diameter, pipe_outer_diameter):
        if isinstance(x, pd.Series):
            series_index = x.index
            break

    is_scalar = not any(
        isinstance(x, (np.ndarray, pd.Series, list, tuple))
        for x in (flow_rate, hole_diameter, pipe_outer_diameter)
    )

    q_arr = np.asarray(flow_rate, dtype=float)
    dh_arr = np.asarray(hole_diameter, dtype=float)
    dp_arr = np.asarray(pipe_outer_diameter, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        diff_sq = (dh_arr ** 2) - (dp_arr ** 2)
        valid = (diff_sq > 0) & (q_arr >= 0)
        v_ft_min = np.where(valid, (24.5 * q_arr) / diff_sq, np.nan)
        res = v_ft_min if in_feet_per_minute else v_ft_min / 60.0

    if is_scalar:
        return float(res) if not np.isnan(res) else float("nan")
    if is_series:
        return pd.Series(res, index=series_index)
    return res
