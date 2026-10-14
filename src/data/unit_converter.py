"""
Oilfield Unit Converter Module
==============================
Provides strict, bidirectional, physics-consistent conversion between Oilfield Imperial
units (lbf, klbf, psi, gpm, ft/hr, ft-lbf) and SI Metric units (N, kN, bar, kPa, lpm, m/hr, N-m).

Author: Tiago Meilan
Project: DrillOpt-Engine
"""

from typing import Union
import numpy as np
import pandas as pd

NumberOrArray = Union[float, int, np.ndarray, pd.Series]


class UnitConverter:
    """Rigorous bidirectional converter for standard drilling and petroleum engineering units."""

    # -------------------------------------------------------------
    # Length & Depth
    # -------------------------------------------------------------
    @staticmethod
    def ft_to_m(val: NumberOrArray) -> NumberOrArray:
        """Converts feet (ft) to meters (m)."""
        return val * 0.3048

    @staticmethod
    def m_to_ft(val: NumberOrArray) -> NumberOrArray:
        """Converts meters (m) to feet (ft)."""
        return val / 0.3048

    @staticmethod
    def in_to_mm(val: NumberOrArray) -> NumberOrArray:
        """Converts inches (in) to millimeters (mm)."""
        return val * 25.4

    @staticmethod
    def mm_to_in(val: NumberOrArray) -> NumberOrArray:
        """Converts millimeters (mm) to inches (in)."""
        return val / 25.4

    # -------------------------------------------------------------
    # Rate of Penetration (ROP)
    # -------------------------------------------------------------
    @staticmethod
    def ft_hr_to_m_hr(val: NumberOrArray) -> NumberOrArray:
        """Converts feet per hour (ft/hr) to meters per hour (m/hr)."""
        return val * 0.3048

    @staticmethod
    def m_hr_to_ft_hr(val: NumberOrArray) -> NumberOrArray:
        """Converts meters per hour (m/hr) to feet per hour (ft/hr)."""
        return val / 0.3048

    # -------------------------------------------------------------
    # Weight on Bit (WOB) & Axial Force
    # -------------------------------------------------------------
    @staticmethod
    def klbf_to_kn(val: NumberOrArray) -> NumberOrArray:
        """Converts thousands of pounds-force (klbf) to kiloNewtons (kN)."""
        return val * 4.4482216152605

    @staticmethod
    def kn_to_klbf(val: NumberOrArray) -> NumberOrArray:
        """Converts kiloNewtons (kN) to thousands of pounds-force (klbf)."""
        return val / 4.4482216152605

    @staticmethod
    def klbf_to_tonnes(val: NumberOrArray) -> NumberOrArray:
        """Converts thousands of pounds-force (klbf) to metric tonnes (t)."""
        return val * 0.45359237

    @staticmethod
    def tonnes_to_klbf(val: NumberOrArray) -> NumberOrArray:
        """Converts metric tonnes (t) to thousands of pounds-force (klbf)."""
        return val / 0.45359237

    # -------------------------------------------------------------
    # Torque
    # -------------------------------------------------------------
    @staticmethod
    def ft_lbf_to_nm(val: NumberOrArray) -> NumberOrArray:
        """Converts foot-pounds (ft-lbf) to Newton-meters (N-m)."""
        return val * 1.3558179483314

    @staticmethod
    def nm_to_ft_lbf(val: NumberOrArray) -> NumberOrArray:
        """Converts Newton-meters (N-m) to foot-pounds (ft-lbf)."""
        return val / 1.3558179483314

    @staticmethod
    def k_ft_lbf_to_knm(val: NumberOrArray) -> NumberOrArray:
        """Converts thousands of foot-pounds (k ft-lbf) to kiloNewton-meters (kN-m)."""
        return val * 1.3558179483314

    # -------------------------------------------------------------
    # Pressure & Stress / MSE
    # -------------------------------------------------------------
    @staticmethod
    def psi_to_bar(val: NumberOrArray) -> NumberOrArray:
        """Converts pounds per square inch (psi) to bar."""
        return val * 0.06894757293168

    @staticmethod
    def bar_to_psi(val: NumberOrArray) -> NumberOrArray:
        """Converts bar to pounds per square inch (psi)."""
        return val / 0.06894757293168

    @staticmethod
    def psi_to_mpa(val: NumberOrArray) -> NumberOrArray:
        """Converts pounds per square inch (psi) to MegaPascals (MPa)."""
        return val * 0.006894757293168

    @staticmethod
    def mpa_to_psi(val: NumberOrArray) -> NumberOrArray:
        """Converts MegaPascals (MPa) to pounds per square inch (psi)."""
        return val / 0.006894757293168

    # -------------------------------------------------------------
    # Flow Rate
    # -------------------------------------------------------------
    @staticmethod
    def gpm_to_lpm(val: NumberOrArray) -> NumberOrArray:
        """Converts gallons per minute (US gpm) to liters per minute (lpm)."""
        return val * 3.785411784

    @staticmethod
    def lpm_to_gpm(val: NumberOrArray) -> NumberOrArray:
        """Converts liters per minute (lpm) to gallons per minute (US gpm)."""
        return val / 3.785411784

    # -------------------------------------------------------------
    # Mud Weight / Density
    # -------------------------------------------------------------
    @staticmethod
    def ppg_to_sg(val: NumberOrArray) -> NumberOrArray:
        """Converts pounds per gallon (ppg) to specific gravity (sg = g/cm3)."""
        return val * 0.119826427

    @staticmethod
    def sg_to_ppg(val: NumberOrArray) -> NumberOrArray:
        """Converts specific gravity (sg) to pounds per gallon (ppg)."""
        return val / 0.119826427
