"""Mechanical Specific Energy (MSE) calculations.

Implements foundational and modern drilling mechanics formulations:
- Teale (1965): Fundamental Mechanical Specific Energy (axial and torsional components).
- Dupriest & Koederitz (2005): Mechanical efficiency and calibrated baseline MSE.
- Rock strength comparison: Mechanical efficiency (EFF_m) and Over-Energy Ratio (OER).

References:
    - Teale, R. (1965). The Concept of Specific Energy in Rock Drilling.
      International Journal of Rock Mechanics and Mining Sciences, 2(1), 57-73.
    - Dupriest, F. E., & Koederitz, W. L. (2005). Maximizing Drill Rates With
      Real-Time Surveillance of Mechanical Specific Energy. SPE/IADC 92194.
"""

from typing import Optional, Union
import math
import numpy as np
import pandas as pd

# Conversion constants
PSI_TO_MPA = 0.006894757293168361
MPA_TO_PSI = 1.0 / PSI_TO_MPA


def psi_to_mpa(
    pressure_psi: Union[float, np.ndarray, pd.Series]
) -> Union[float, np.ndarray, pd.Series]:
    """Convert pressure or specific energy from psi to MPa.

    Args:
        pressure_psi: Pressure in pounds per square inch (psi).

    Returns:
        Pressure in Megapascals (MPa).
    """
    return pressure_psi * PSI_TO_MPA


def mpa_to_psi(
    pressure_mpa: Union[float, np.ndarray, pd.Series]
) -> Union[float, np.ndarray, pd.Series]:
    """Convert pressure or specific energy from MPa to psi.

    Args:
        pressure_mpa: Pressure in Megapascals (MPa).

    Returns:
        Pressure in pounds per square inch (psi).
    """
    return pressure_mpa * MPA_TO_PSI


def _compute_bit_area(
    bit_diameter: Optional[Union[float, np.ndarray, pd.Series]] = None,
    bit_area: Optional[Union[float, np.ndarray, pd.Series]] = None,
) -> Union[float, np.ndarray, pd.Series]:
    """Calculate or validate the bit cross-sectional area.

    Args:
        bit_diameter: Bit outer diameter in inches (or meters if SI).
        bit_area: Bit cross-sectional area in square inches (or m^2 if SI).

    Returns:
        Bit cross-sectional area A_b.

    Raises:
        ValueError: If neither or both bit_diameter and bit_area are missing.
    """
    if bit_area is not None:
        return bit_area
    if bit_diameter is not None:
        return (np.pi / 4.0) * (bit_diameter ** 2)
    raise ValueError("Either 'bit_diameter' or 'bit_area' must be provided.")


def _to_numpy_or_scalar(val: Union[float, int, np.ndarray, pd.Series]):
    """Convert input to numpy array or scalar float, keeping track of pandas wrapper."""
    is_pandas = isinstance(val, (pd.Series, pd.DataFrame))
    index = getattr(val, "index", None)
    if is_pandas:
        arr = val.to_numpy(dtype=float)
    elif isinstance(val, (list, tuple, np.ndarray)):
        arr = np.asarray(val, dtype=float)
    else:
        arr = float(val) if val is not None else np.nan
    return arr, is_pandas, index


def teale_mse(
    wob: Union[float, int, np.ndarray, pd.Series],
    torque: Union[float, int, np.ndarray, pd.Series],
    rpm: Union[float, int, np.ndarray, pd.Series],
    rop: Union[float, int, np.ndarray, pd.Series],
    bit_diameter: Optional[Union[float, int, np.ndarray, pd.Series]] = None,
    bit_area: Optional[Union[float, int, np.ndarray, pd.Series]] = None,
    use_field_units: bool = True,
    output_unit: str = "psi",
) -> Union[float, np.ndarray, pd.Series]:
    """Compute Teale (1965) Mechanical Specific Energy (MSE).

    In field units (default):
        - WOB: Weight on bit in pounds-force (lbf).
        - Torque: Drilling torque in foot-pounds (ft-lbf).
        - RPM: Rotary speed in revolutions per minute (RPM).
        - ROP: Rate of penetration in feet per hour (ft/hr).
        - bit_diameter: Bit outer diameter in inches (in).
        - bit_area: Bit area in square inches (in^2).

    Formula (Field Units):
        MSE_axial = WOB / A_b   [psi]
        MSE_rot   = (120 * pi * RPM * Torque) / (A_b * ROP)   [psi]
                  = (480 * RPM * Torque) / (D_b^2 * ROP)
        MSE_total = MSE_axial + MSE_rot   [psi]

    In SI units (use_field_units=False):
        - WOB: Newtons (N).
        - Torque: Newton-meters (N-m).
        - RPM: Revolutions per minute (RPM).
        - ROP: Rate of penetration in meters per hour (m/hr).
        - bit_diameter: Bit outer diameter in meters (m).
        - bit_area: Bit area in square meters (m^2).
        Formula:
        MSE = WOB / A_b + (2 * pi * (RPM/60) * Torque) / (A_b * (ROP/3600))  [Pa]

    Division by zero (e.g. ROP <= 0, A_b <= 0) is handled safely by returning np.nan.

    Args:
        wob: Weight on Bit (lbf in field, N in SI).
        torque: Surface or downhole torque (ft-lbf in field, N-m in SI).
        rpm: Drillstring rotary speed (RPM).
        rop: Rate of penetration (ft/hr in field, m/hr in SI).
        bit_diameter: Bit diameter (inches in field, meters in SI).
        bit_area: Bit cross-sectional area (in^2 in field, m^2 in SI).
        use_field_units: If True, uses field units; if False, uses SI units.
        output_unit: Output unit ('psi' or 'MPa'). Default is 'psi'.

    Returns:
        Calculated MSE in requested units ('psi' or 'MPa') matching input type
        (scalar float, numpy array, or pandas Series).
    """
    # Track pandas series types for consistent return type
    is_series = any(isinstance(x, pd.Series) for x in (wob, torque, rpm, rop, bit_diameter, bit_area))
    series_index = None
    for x in (wob, torque, rpm, rop, bit_diameter, bit_area):
        if isinstance(x, pd.Series):
            series_index = x.index
            break

    area = _compute_bit_area(bit_diameter=bit_diameter, bit_area=bit_area)

    # Convert all inputs to numpy arrays (or keep scalars if all are scalars)
    is_scalar = not any(
        isinstance(x, (np.ndarray, pd.Series, list, tuple))
        for x in (wob, torque, rpm, rop, area)
    )

    wob_arr = np.asarray(wob, dtype=float)
    tor_arr = np.asarray(torque, dtype=float)
    rpm_arr = np.asarray(rpm, dtype=float)
    rop_arr = np.asarray(rop, dtype=float)
    area_arr = np.asarray(area, dtype=float)

    # Safe mask where ROP > 0 and area > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        valid_mask = (rop_arr > 0) & (area_arr > 0)

        if use_field_units:
            # Field: WOB in lbf, area in in^2 -> axial in psi
            # Torque in ft-lbf, ROP in ft/hr, RPM in rev/min
            # 1 ft-lbf = 12 in-lbf; 1 hr = 60 min.
            # Work = 2*pi*RPM*Torque*12 in-lbf/min
            # Volume = Area * (ROP*12/60) in^3/min = Area * ROP / 5 in^3/min
            # MSE_rot = (24*pi*RPM*Torque) / (Area*ROP/5) = (120*pi*RPM*Torque)/(Area*ROP)
            axial = np.where(area_arr > 0, wob_arr / area_arr, np.nan)
            rotational = np.where(
                valid_mask,
                (120.0 * np.pi * rpm_arr * tor_arr) / (area_arr * rop_arr),
                np.nan,
            )
            mse_psi = axial + rotational
        else:
            # SI units: WOB in N, area in m^2, torque in N-m, ROP in m/hr -> m/s is ROP/3600
            # RPM in rev/min -> rev/s is RPM/60
            axial = np.where(area_arr > 0, wob_arr / area_arr, np.nan)
            rot_work = 2.0 * np.pi * (rpm_arr / 60.0) * tor_arr  # W (J/s)
            vol_rate = area_arr * (rop_arr / 3600.0)             # m^3/s
            rotational = np.where(valid_mask, rot_work / vol_rate, np.nan)
            mse_pa = axial + rotational
            mse_psi = mse_pa / 6894.757293168361

        if output_unit.lower() == "mpa":
            result = psi_to_mpa(mse_psi)
        elif output_unit.lower() == "psi":
            result = mse_psi
        else:
            raise ValueError(f"Unsupported output_unit '{output_unit}'. Use 'psi' or 'MPa'.")

    if is_scalar:
        return float(result) if not np.isnan(result) else float("nan")
    if is_series:
        return pd.Series(result, index=series_index)
    return result


def dupriest_mse(
    mse_teale: Optional[Union[float, int, np.ndarray, pd.Series]] = None,
    wob: Optional[Union[float, int, np.ndarray, pd.Series]] = None,
    torque: Optional[Union[float, int, np.ndarray, pd.Series]] = None,
    rpm: Optional[Union[float, int, np.ndarray, pd.Series]] = None,
    rop: Optional[Union[float, int, np.ndarray, pd.Series]] = None,
    bit_diameter: Optional[Union[float, int, np.ndarray, pd.Series]] = None,
    bit_area: Optional[Union[float, int, np.ndarray, pd.Series]] = None,
    eff_m: float = 0.35,
    use_field_units: bool = True,
    output_unit: str = "psi",
) -> Union[float, np.ndarray, pd.Series]:
    """Compute Dupriest & Koederitz (2005) Mechanical Specific Energy.

    Dupriest recognized that minimum MSE achieved in a rock under peak efficiency
    is approximately equal to its UCS / EFF_m, with a baseline mechanical efficiency
    EFF_m typically around 0.35 for PDC bits (or 0.30 - 0.40).
    The calibrated effective energy is:
        MSE_dupriest = EFF_m * MSE_teale

    Args:
        mse_teale: Precomputed Teale MSE. If None, computed from drilling parameters.
        wob: Weight on Bit (required if mse_teale is None).
        torque: Torque (required if mse_teale is None).
        rpm: Rotary speed (required if mse_teale is None).
        rop: Rate of penetration (required if mse_teale is None).
        bit_diameter: Bit outer diameter.
        bit_area: Bit cross-sectional area.
        eff_m: Mechanical efficiency factor (default 0.35 for PDC).
        use_field_units: Use field units if True.
        output_unit: Output unit ('psi' or 'MPa').

    Returns:
        Dupriest MSE in requested units.
    """
    if mse_teale is None:
        if wob is None or torque is None or rpm is None or rop is None:
            raise ValueError("Must provide either 'mse_teale' or all drilling parameters.")
        mse_teale = teale_mse(
            wob=wob,
            torque=torque,
            rpm=rpm,
            rop=rop,
            bit_diameter=bit_diameter,
            bit_area=bit_area,
            use_field_units=use_field_units,
            output_unit=output_unit,
        )

    return mse_teale * eff_m


def mechanical_efficiency(
    mse_teale: Union[float, int, np.ndarray, pd.Series],
    ucs: Union[float, int, np.ndarray, pd.Series],
) -> Union[float, np.ndarray, pd.Series]:
    """Calculate mechanical drilling efficiency EFF_m = UCS / MSE_teale.

    Mechanical efficiency reflects the fraction of mechanical energy input that is
    actually utilized in rock destruction versus wasted in friction, vibration,
    or cutter wear.

    Args:
        mse_teale: Teale MSE in same unit as UCS (e.g. psi or MPa).
        ucs: Unconfined Compressive Strength of rock in same unit.

    Returns:
        Dimensionless efficiency EFF_m (typically 0.0 to 1.0; 0.30-0.40 optimal for PDC).
        Returns np.nan if MSE <= 0 or UCS <= 0.
    """
    is_series = isinstance(mse_teale, pd.Series) or isinstance(ucs, pd.Series)
    series_index = getattr(mse_teale, "index", None) or getattr(ucs, "index", None)
    is_scalar = not isinstance(mse_teale, (np.ndarray, pd.Series, list, tuple)) and not isinstance(
        ucs, (np.ndarray, pd.Series, list, tuple)
    )

    mse_arr = np.asarray(mse_teale, dtype=float)
    ucs_arr = np.asarray(ucs, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        valid = (mse_arr > 0) & (ucs_arr > 0)
        eff = np.where(valid, ucs_arr / mse_arr, np.nan)

    if is_scalar:
        return float(eff) if not np.isnan(eff) else float("nan")
    if is_series:
        return pd.Series(eff, index=series_index)
    return eff


def over_energy_ratio(
    mse_teale: Union[float, int, np.ndarray, pd.Series],
    ucs: Union[float, int, np.ndarray, pd.Series],
) -> Union[float, np.ndarray, pd.Series]:
    """Calculate the Over-Energy Ratio (OER) = MSE_teale / UCS.

    OER is the inverse of mechanical efficiency (OER = 1 / EFF_m).
    An OER of ~2.5 - 3.5 represents normal efficient PDC drilling.
    An OER > 5.0 - 10.0 signals severe energy dissipation from bit founder,
    balling, whirl, stick-slip, or extreme cutter wear.

    Args:
        mse_teale: Teale MSE in same units as UCS.
        ucs: Rock unconfined compressive strength in same units.

    Returns:
        Dimensionless over-energy ratio MSE / UCS.
    """
    is_series = isinstance(mse_teale, pd.Series) or isinstance(ucs, pd.Series)
    series_index = getattr(mse_teale, "index", None) or getattr(ucs, "index", None)
    is_scalar = not isinstance(mse_teale, (np.ndarray, pd.Series, list, tuple)) and not isinstance(
        ucs, (np.ndarray, pd.Series, list, tuple)
    )

    mse_arr = np.asarray(mse_teale, dtype=float)
    ucs_arr = np.asarray(ucs, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        valid = (ucs_arr > 0) & (mse_arr > 0)
        oer = np.where(valid, mse_arr / ucs_arr, np.nan)

    if is_scalar:
        return float(oer) if not np.isnan(oer) else float("nan")
    if is_series:
        return pd.Series(oer, index=series_index)
    return oer
