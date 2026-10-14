"""
Rig State Classification Module
================================
Identifies the operational state of the drilling rig from continuous telemetry channels.
Distinguishes between:
- DRILLING_ON_BOTTOM: Active rotary drilling into formation (WOB > 0, RPM > 0, Flow > 0, ROP > 0)
- DRILLING_SLIDING: Steerable mud motor drilling without surface drillstring rotation (RPM_surf ~ 0)
- REAMING_ROTATING: Circulating and rotating while not on bottom (hole conditioning)
- CIRCULATING_OFF_BOTTOM: Mud pumps active without drillstring rotation or bottom contact
- CONNECTION_STATIC: No pumping, no rotation, slips set, adding/removing pipe stand

Author: Tiago Meilan
Project: DrillOpt-Engine
"""

from enum import Enum
from typing import Optional
import numpy as np
import pandas as pd


class RigState(str, Enum):
    DRILLING_ROTARY = "DRILLING_ROTARY"
    DRILLING_SLIDING = "DRILLING_SLIDING"
    REAMING = "REAMING"
    CIRCULATING = "CIRCULATING"
    CONNECTION = "CONNECTION"
    TRIPPING_OR_STATIC = "TRIPPING_OR_STATIC"


class RigStateClassifier:
    """Classifies continuous telemetry intervals into discrete operational rig states."""

    def __init__(
        self,
        min_wob_klbf: float = 2.0,
        min_rpm: float = 15.0,
        min_flow_gpm: float = 100.0,
        min_rop_m_hr: float = 0.5,
    ):
        self.min_wob_klbf = min_wob_klbf
        self.min_rpm = min_rpm
        self.min_flow_gpm = min_flow_gpm
        self.min_rop_m_hr = min_rop_m_hr

    def classify_row(
        self,
        wob: float,
        rpm: float,
        flow: float,
        rop: float,
    ) -> RigState:
        """Determines the rig state for a single point of telemetry."""
        is_pumping = flow >= self.min_flow_gpm
        is_rotating = rpm >= self.min_rpm
        is_on_bottom = wob >= self.min_wob_klbf
        is_penetrating = rop >= self.min_rop_m_hr

        if is_on_bottom and is_pumping and is_penetrating:
            if is_rotating:
                return RigState.DRILLING_ROTARY
            else:
                return RigState.DRILLING_SLIDING
        elif is_pumping:
            if is_rotating:
                return RigState.REAMING
            else:
                return RigState.CIRCULATING
        elif not is_pumping and not is_rotating and not is_on_bottom:
            return RigState.CONNECTION
        else:
            return RigState.TRIPPING_OR_STATIC

    def classify_dataframe(
        self,
        df: pd.DataFrame,
        wob_col: str = "wob",
        rpm_col: str = "rpm",
        flow_col: str = "flow",
        rop_col: str = "rop",
        output_col: str = "rig_state",
    ) -> pd.DataFrame:
        """
        Applies rig state classification to an entire telemetry DataFrame.
        Adds `output_col` and a boolean column `is_on_bottom`.
        """
        df_out = df.copy()

        # Fill NaNs with 0 for thresholding
        wob_series = df_out[wob_col].fillna(0) if wob_col in df_out.columns else pd.Series(0, index=df_out.index)
        rpm_series = df_out[rpm_col].fillna(0) if rpm_col in df_out.columns else pd.Series(0, index=df_out.index)
        flow_series = df_out[flow_col].fillna(0) if flow_col in df_out.columns else pd.Series(0, index=df_out.index)
        rop_series = df_out[rop_col].fillna(0) if rop_col in df_out.columns else pd.Series(0, index=df_out.index)

        is_pumping = flow_series >= self.min_flow_gpm
        is_rotating = rpm_series >= self.min_rpm
        is_on_bottom = wob_series >= self.min_wob_klbf
        is_penetrating = rop_series >= self.min_rop_m_hr

        conditions = [
            is_on_bottom & is_pumping & is_penetrating & is_rotating,
            is_on_bottom & is_pumping & is_penetrating & (~is_rotating),
            is_pumping & is_rotating & (~is_on_bottom),
            is_pumping & (~is_rotating) & (~is_on_bottom),
            (~is_pumping) & (~is_rotating) & (~is_on_bottom),
        ]

        choices = [
            RigState.DRILLING_ROTARY.value,
            RigState.DRILLING_SLIDING.value,
            RigState.REAMING.value,
            RigState.CIRCULATING.value,
            RigState.CONNECTION.value,
        ]

        df_out[output_col] = np.select(conditions, choices, default=RigState.TRIPPING_OR_STATIC.value)
        df_out["is_on_bottom"] = df_out[output_col].isin([
            RigState.DRILLING_ROTARY.value,
            RigState.DRILLING_SLIDING.value,
        ])

        return df_out

    @staticmethod
    def filter_active_drilling(df: pd.DataFrame, state_col: str = "rig_state") -> pd.DataFrame:
        """Convenience filter that returns only intervals of active on-bottom drilling."""
        if state_col not in df.columns:
            return df
        return df[df[state_col] == RigState.DRILLING_ROTARY.value].copy()
