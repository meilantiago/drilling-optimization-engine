"""D-exponent and corrected Dc-exponent pore pressure calculations.

Formulations:
- Jordan & Shirley (1966): Dimensionless drilling exponent (D-exponent) normalized
  for ROP, rotary speed, bit diameter, and weight on bit.
- Rehm & McClendon (1971): Corrected Dc-exponent accounting for mud weight differences
  relative to normal hydrostatic formation pressure.
- Eaton (1975): Quantitative pore pressure gradient prediction from Dc departure
  from the normal compaction trend (NCT).

References:
    - Jordan, J. R., & Shirley, O. J. (1966). Application of Drillability Concepts
      to Overburden and Pore Pressure Detection. Journal of Petroleum Technology, 18(05).
    - Rehm, W. A., & McClendon, R. (1971). Measurement of Formation Pressure from
      Drilling Data. SPE 3601.
    - Eaton, B. A. (1975). The Equation for Geopressure Prediction from Well Logs.
      SPE 5544.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd


def jordan_shirley_d_exponent(
    rop: Union[float, int, np.ndarray, pd.Series],
    rpm: Union[float, int, np.ndarray, pd.Series],
    wob: Union[float, int, np.ndarray, pd.Series],
    bit_diameter: Union[float, int, np.ndarray, pd.Series],
) -> Union[float, np.ndarray, pd.Series]:
    """Calculate the Jordan & Shirley (1966) D-exponent.

    Formula:
        D = log10(ROP / (60 * RPM)) / log10((12 * WOB) / (1e6 * D_b))

    Standard Units:
        - ROP: Rate of penetration in ft/hr.
        - RPM: Rotary speed in revolutions per minute.
        - WOB: Weight on bit in pounds-force (lbf).
        - bit_diameter (D_b): Bit diameter in inches.

    Handles zero, non-positive values, or invalid ratios by returning np.nan.

    Args:
        rop: Rate of penetration in ft/hr.
        rpm: Drillstring rotary speed in RPM.
        wob: Weight on bit in lbf.
        bit_diameter: Bit outer diameter in inches.

    Returns:
        D-exponent (dimensionless, typically 1.0 - 2.5 for normal shales).
    """
    is_series = any(isinstance(x, pd.Series) for x in (rop, rpm, wob, bit_diameter))
    series_index = None
    for x in (rop, rpm, wob, bit_diameter):
        if isinstance(x, pd.Series):
            series_index = x.index
            break

    is_scalar = not any(
        isinstance(x, (np.ndarray, pd.Series, list, tuple))
        for x in (rop, rpm, wob, bit_diameter)
    )

    rop_arr = np.asarray(rop, dtype=float)
    rpm_arr = np.asarray(rpm, dtype=float)
    wob_arr = np.asarray(wob, dtype=float)
    db_arr = np.asarray(bit_diameter, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        num_arg = rop_arr / (60.0 * rpm_arr)
        den_arg = (12.0 * wob_arr) / (1.0e6 * db_arr)

        valid = (num_arg > 0) & (den_arg > 0) & (den_arg != 1.0)

        num = np.where(valid, np.log10(num_arg), np.nan)
        den = np.where(valid, np.log10(den_arg), np.nan)

        d_exp = np.where(valid & (den != 0), num / den, np.nan)

    if is_scalar:
        return float(d_exp) if not np.isnan(d_exp) else float("nan")
    if is_series:
        return pd.Series(d_exp, index=series_index)
    return d_exp


def rehm_mcclendon_dc_exponent(
    d_exponent: Union[float, int, np.ndarray, pd.Series],
    actual_mud_weight: Union[float, int, np.ndarray, pd.Series],
    normal_mud_weight: Union[float, int, np.ndarray, pd.Series] = 9.0,
) -> Union[float, np.ndarray, pd.Series]:
    """Calculate Rehm & McClendon (1971) corrected Dc-exponent.

    Normalizes the D-exponent for the effect of overbalanced hydrostatic mud weight.
    Formula:
        Dc = D * (normal_mud_weight / actual_mud_weight)

    Args:
        d_exponent: Jordan & Shirley D-exponent (dimensionless).
        actual_mud_weight: Actual drilling fluid density (e.g. ppg or SG).
        normal_mud_weight: Normal pore fluid equivalent density (default 9.0 ppg,
            or 8.33 ppg for pure fresh water).

    Returns:
        Corrected Dc-exponent (dimensionless).
    """
    is_series = any(
        isinstance(x, pd.Series)
        for x in (d_exponent, actual_mud_weight, normal_mud_weight)
    )
    series_index = None
    for x in (d_exponent, actual_mud_weight, normal_mud_weight):
        if isinstance(x, pd.Series):
            series_index = x.index
            break

    is_scalar = not any(
        isinstance(x, (np.ndarray, pd.Series, list, tuple))
        for x in (d_exponent, actual_mud_weight, normal_mud_weight)
    )

    d_arr = np.asarray(d_exponent, dtype=float)
    act_mw = np.asarray(actual_mud_weight, dtype=float)
    norm_mw = np.asarray(normal_mud_weight, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        valid = (act_mw > 0) & np.isfinite(d_arr)
        dc = np.where(valid, d_arr * (norm_mw / act_mw), np.nan)

    if is_scalar:
        return float(dc) if not np.isnan(dc) else float("nan")
    if is_series:
        return pd.Series(dc, index=series_index)
    return dc


def fit_normal_compaction_trend(
    depth: Union[np.ndarray, pd.Series],
    dc_exponent: Union[np.ndarray, pd.Series],
    depth_min: Optional[float] = None,
    depth_max: Optional[float] = None,
) -> Tuple[float, float]:
    """Fit a linear Normal Compaction Trend (NCT) to Dc-exponent data.

    In normally pressured shales, matrix compaction increases uniformly with depth,
    yielding a linear increase in Dc:
        Dc_normal(depth) = slope * depth + intercept

    Args:
        depth: Measured or true vertical depth array.
        dc_exponent: Corrected Dc-exponent array.
        depth_min: Optional minimum depth of normal compaction interval.
        depth_max: Optional maximum depth of normal compaction interval.

    Returns:
        Tuple of (slope, intercept).
    """
    z = np.asarray(depth, dtype=float)
    dc = np.asarray(dc_exponent, dtype=float)

    mask = np.isfinite(z) & np.isfinite(dc) & (dc > 0)
    if depth_min is not None:
        mask &= z >= depth_min
    if depth_max is not None:
        mask &= z <= depth_max

    z_valid = z[mask]
    dc_valid = dc[mask]

    if len(z_valid) < 2 or len(np.unique(z_valid)) < 2:
        raise ValueError("Insufficient distinct depth points to fit normal compaction trend.")

    # Linear regression: Dc = slope * depth + intercept
    slope, intercept = np.polyfit(z_valid, dc_valid, 1)
    return float(slope), float(intercept)


def detect_pore_pressure_anomaly(
    depth: Union[np.ndarray, pd.Series],
    dc_exponent: Union[np.ndarray, pd.Series],
    normal_slope: Optional[float] = None,
    normal_intercept: Optional[float] = None,
    threshold: float = 0.15,
) -> pd.DataFrame:
    """Detect abnormal pore pressure trend deviations from the Normal Compaction Trend.

    A significant drop of Dc below the normal compaction line (Dc_norm - Dc > threshold)
    is the primary real-time signature of entering under-compacted, overpressured shale.

    Args:
        depth: Well depth array.
        dc_exponent: Corrected Dc-exponent array.
        normal_slope: Slope of normal compaction line. If None, fitted from the top 40% of data.
        normal_intercept: Intercept of normal compaction line.
        threshold: Dc departure threshold to trigger overpressure alarm (default 0.15).

    Returns:
        pd.DataFrame containing:
            - 'depth': Depth
            - 'dc': Measured Dc
            - 'dc_norm': Normal compaction trendline value
            - 'delta_dc': Departure (dc_norm - dc)
            - 'is_overpressured': Boolean flag (True if delta_dc >= threshold)
            - 'anomaly_severity': Categorical severity ('normal', 'mild', 'severe')
    """
    z = np.asarray(depth, dtype=float)
    dc = np.asarray(dc_exponent, dtype=float)

    if normal_slope is None or normal_intercept is None:
        valid_idx = np.where(np.isfinite(z) & np.isfinite(dc) & (dc > 0))[0]
        if len(valid_idx) < 2 or len(np.unique(z[valid_idx])) < 2:
            raise ValueError("Cannot automatically fit NCT: insufficient distinct depth points.")
        # Fit on top 40% of depth by default
        split_z = np.percentile(z[valid_idx], 40)
        fit_mask = valid_idx[z[valid_idx] <= split_z]
        if len(fit_mask) < 2 or len(np.unique(z[fit_mask])) < 2:
            fit_mask = valid_idx[: max(2, len(valid_idx) // 2)]
        slope_fit, intercept_fit = np.polyfit(z[fit_mask], dc[fit_mask], 1)
        normal_slope, normal_intercept = float(slope_fit), float(intercept_fit)

    dc_norm = normal_slope * z + normal_intercept
    delta_dc = dc_norm - dc  # positive means under-compacted / overpressured

    is_overpressured = (delta_dc >= threshold) & np.isfinite(dc)

    # Classify severity
    severity = np.full(len(z), "normal", dtype=object)
    mild_mask = (delta_dc >= threshold) & (delta_dc < 2.0 * threshold) & np.isfinite(dc)
    severe_mask = (delta_dc >= 2.0 * threshold) & np.isfinite(dc)
    severity[mild_mask] = "mild"
    severity[severe_mask] = "severe"

    return pd.DataFrame({
        "depth": z,
        "dc": dc,
        "dc_norm": dc_norm,
        "delta_dc": delta_dc,
        "is_overpressured": is_overpressured,
        "anomaly_severity": severity,
    })


def eaton_pore_pressure(
    dc_exponent: Union[float, np.ndarray, pd.Series],
    dc_normal: Union[float, np.ndarray, pd.Series],
    overburden_gradient: Union[float, np.ndarray, pd.Series] = 1.0,
    normal_pore_gradient: Union[float, np.ndarray, pd.Series] = 0.465,
    exponent: float = 1.2,
) -> Union[float, np.ndarray, pd.Series]:
    """Calculate pore pressure gradient using Eaton's (1975) equation for Dc.

    Formula:
        PPG = OBG - (OBG - PPG_norm) * (Dc / Dc_norm)^1.2

    Args:
        dc_exponent: Observed Dc-exponent.
        dc_normal: Normal compaction trend Dc value at the same depth.
        overburden_gradient: Overburden stress gradient (psi/ft or ppg, default 1.0 psi/ft).
        normal_pore_gradient: Normal hydrostatic pressure gradient (psi/ft or ppg, default 0.465 psi/ft).
        exponent: Eaton Dc exponent (standard empirical value 1.2).

    Returns:
        Estimated pore pressure gradient in same units as OBG and normal_pore_gradient.
    """
    is_series = any(isinstance(x, pd.Series) for x in (dc_exponent, dc_normal))
    series_index = getattr(dc_exponent, "index", None) or getattr(dc_normal, "index", None)
    is_scalar = not any(
        isinstance(x, (np.ndarray, pd.Series, list, tuple))
        for x in (dc_exponent, dc_normal, overburden_gradient, normal_pore_gradient)
    )

    dc_arr = np.asarray(dc_exponent, dtype=float)
    dcn_arr = np.asarray(dc_normal, dtype=float)
    obg = np.asarray(overburden_gradient, dtype=float)
    ppgn = np.asarray(normal_pore_gradient, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = dc_arr / dcn_arr
        valid = (dcn_arr > 0) & (ratio > 0) & np.isfinite(ratio)
        ppg = np.where(valid, obg - (obg - ppgn) * (ratio ** exponent), np.nan)

    if is_scalar:
        return float(ppg) if not np.isnan(ppg) else float("nan")
    if is_series:
        return pd.Series(ppg, index=series_index)
    return ppg
