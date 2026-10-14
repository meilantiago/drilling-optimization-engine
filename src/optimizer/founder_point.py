"""
Founder Point Automated Detection Module
=========================================
Automated, robust detection of the drilling Founder Point (inflection point where
the rate of penetration d(ROP)/d(WOB) saturates and Mechanical Specific Energy
d(MSE)/d(WOB) shoots up).

Based on:
- Dupriest, F.E., & Koederitz, W.L. (2005). "Maximizing Drill Rates with Real-Time
  Surveillance of Mechanical Specific Energy." SPE/IADC 92194.
- Teale, R. (1965). "The Concept of Specific Energy in Rock Drilling."

Author: Tiago Meilan
Project: DrillOpt-Engine
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


@dataclass
class FounderPointResult:
    """Diagnostic outcome of founder point analysis on a telemetry interval."""
    wob_founder: float
    wob_recommended: float
    rop_at_founder: float
    rop_at_recommended: float
    mse_at_founder: float
    confidence_score: float       # R-squared or goodness of fit (0.0 to 1.0)
    regime: str                   # "founder_detected", "pre_founder", "post_founder", "insufficient_data"
    d_rop_d_wob: float            # Local cutting gradient at recommended operating point
    wob_tested_range: Tuple[float, float]
    depth_interval: Tuple[float, float]
    formation: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class FounderPointAnalyzer:
    """
    Automated detector of the drilling founder point using piecewise linear
    regression, polynomial curvature analysis, and MSE inflection tracking.
    """

    def __init__(
        self,
        bit_diameter_in: float = 8.5,
        safety_factor: float = 0.90,
        min_wob_span_klbf: float = 4.0,
        min_samples: int = 15,
    ):
        """
        Parameters
        ----------
        bit_diameter_in : float
            Bit diameter in inches (default: 8.5).
        safety_factor : float
            Fraction of WOB_founder recommended for safe continuous drilling (default: 0.90).
        min_wob_span_klbf : float
            Minimum span between max and min WOB required to infer curvature reliably.
        min_samples : int
            Minimum number of active drilling points required.
        """
        self.bit_diameter_in = bit_diameter_in
        self.bit_area_in2 = (np.pi / 4.0) * (bit_diameter_in ** 2)
        self.safety_factor = safety_factor
        self.min_wob_span_klbf = min_wob_span_klbf
        self.min_samples = min_samples

    def calculate_mse(
        self,
        df: pd.DataFrame,
        wob_col: str = "wob",
        rop_col: str = "rop",
        torque_col: str = "torque",
        rpm_col: str = "rpm",
    ) -> pd.Series:
        """
        Calculates Teale's Mechanical Specific Energy (psi):
        MSE = (WOB / A_bit) + (120 * pi * RPM * Torque) / (A_bit * ROP)

        Assumes:
        - wob in klbf
        - rop in m/hr (converted internally to ft/hr)
        - torque in ft-lbf
        - rpm in rev/min
        """
        wob_lbf = df[wob_col].fillna(0) * 1000.0
        # Convert m/hr to ft/hr: 1 m/hr = 3.28084 ft/hr
        rop_ft_hr = np.maximum(df[rop_col].fillna(0) / 0.3048, 0.1)
        torque = df[torque_col].fillna(0) if torque_col in df.columns else pd.Series(0, index=df.index)
        rpm = df[rpm_col].fillna(0) if rpm_col in df.columns else pd.Series(0, index=df.index)

        axial = wob_lbf / self.bit_area_in2
        torsional = (120.0 * np.pi * rpm * torque) / (self.bit_area_in2 * rop_ft_hr)
        return axial + torsional

    def filter_active_drilling(
        self,
        df: pd.DataFrame,
        wob_col: str = "wob",
        rop_col: str = "rop",
        rpm_col: str = "rpm",
        min_wob: float = 2.0,
        min_rop: float = 0.5,
        min_rpm: float = 20.0,
    ) -> pd.DataFrame:
        """Filters continuous telemetry to isolate valid on-bottom rotary drilling."""
        if df.empty:
            return df

        mask = (
            (df[wob_col] >= min_wob)
            & (df[rop_col] >= min_rop)
            & (df[rpm_col] >= min_rpm)
            & df[wob_col].notna()
            & df[rop_col].notna()
        )
        return df[mask].copy()

    def analyze(
        self,
        df: pd.DataFrame,
        depth_col: str = "depth",
        wob_col: str = "wob",
        rop_col: str = "rop",
        torque_col: str = "torque",
        rpm_col: str = "rpm",
        mse_col: Optional[str] = "mse_psi",
        formation_col: Optional[str] = "formation",
    ) -> FounderPointResult:
        """
        Analyzes a telemetry segment to detect the founder point and recommend optimal WOB.

        Parameters
        ----------
        df : pd.DataFrame
            Telemetry dataframe.
        depth_col : str
            Depth column name.
        wob_col : str
            WOB column name.
        rop_col : str
            ROP column name.
        torque_col : str
            Torque column name.
        rpm_col : str
            RPM column name.
        mse_col : Optional[str]
            MSE column name. If not present, calculated automatically via Teale's equation.
        formation_col : Optional[str]
            Formation name column if present.

        Returns
        -------
        FounderPointResult
            Identified founder WOB, recommended WOB, regime classification, and diagnostics.
        """
        active_df = self.filter_active_drilling(df, wob_col=wob_col, rop_col=rop_col, rpm_col=rpm_col)

        d_min = float(df[depth_col].min()) if (depth_col in df.columns and not df.empty) else 0.0
        d_max = float(df[depth_col].max()) if (depth_col in df.columns and not df.empty) else 0.0
        fm_name = str(df[formation_col].iloc[0]) if (formation_col and formation_col in df.columns and not df.empty) else None

        if len(active_df) < self.min_samples:
            return FounderPointResult(
                wob_founder=float("nan"),
                wob_recommended=float("nan"),
                rop_at_founder=float("nan"),
                rop_at_recommended=float("nan"),
                mse_at_founder=float("nan"),
                confidence_score=0.0,
                regime="insufficient_data",
                d_rop_d_wob=0.0,
                wob_tested_range=(0.0, 0.0),
                depth_interval=(d_min, d_max),
                formation=fm_name,
                details={"reason": f"Sample count ({len(active_df)}) < minimum required ({self.min_samples})"},
            )

        wob = active_df[wob_col].values
        rop = active_df[rop_col].values

        if mse_col and mse_col in active_df.columns:
            mse = active_df[mse_col].values
        else:
            mse = self.calculate_mse(active_df, wob_col=wob_col, rop_col=rop_col, torque_col=torque_col, rpm_col=rpm_col).values

        w_min = float(np.min(wob))
        w_max = float(np.max(wob))
        w_span = w_max - w_min

        if w_span < self.min_wob_span_klbf:
            # WOB was held steady without exploration
            mean_wob = float(np.mean(wob))
            mean_rop = float(np.mean(rop))
            mean_mse = float(np.mean(mse))
            return FounderPointResult(
                wob_founder=w_max,
                wob_recommended=mean_wob,
                rop_at_founder=mean_rop,
                rop_at_recommended=mean_rop,
                mse_at_founder=mean_mse,
                confidence_score=0.3,
                regime="pre_founder",
                d_rop_d_wob=1.0,
                wob_tested_range=(w_min, w_max),
                depth_interval=(d_min, d_max),
                formation=fm_name,
                details={"reason": f"WOB span ({w_span:.1f} klbf) < threshold ({self.min_wob_span_klbf:.1f} klbf)"},
            )

        # ------------------------------------------------------------------
        # Method 1: Quadratic Polynomial Curvature Fit: ROP(w) = a*w^2 + b*w + c
        # ------------------------------------------------------------------
        # Sort by WOB for monotonic analysis
        sort_idx = np.argsort(wob)
        w_sorted = wob[sort_idx]
        rop_sorted = rop[sort_idx]
        mse_sorted = mse[sort_idx]

        poly_coeffs = np.polyfit(w_sorted, rop_sorted, deg=2)
        a, b, c = poly_coeffs
        # R-squared of quadratic model
        rop_poly_pred = np.polyval(poly_coeffs, w_sorted)
        ss_tot = np.sum((rop_sorted - np.mean(rop_sorted)) ** 2)
        ss_res = np.sum((rop_sorted - rop_poly_pred) ** 2)
        r2_poly = max(0.0, float(1.0 - (ss_res / max(1e-6, ss_tot))))

        # ------------------------------------------------------------------
        # Method 2: Piecewise Two-Segment Linear Search (Broken-Stick Model)
        # ------------------------------------------------------------------
        # Evaluate candidate breakpoints wf from 25th to 85th percentiles of WOB
        q25, q85 = np.percentile(w_sorted, [20, 85])
        candidate_breakpoints = np.linspace(q25, q85, 30)

        best_wf = None
        best_sse = float("inf")
        best_slope_linear = 0.0
        best_slope_saturation = 0.0

        for wf in candidate_breakpoints:
            idx_left = w_sorted <= wf
            idx_right = w_sorted > wf

            if (
                np.sum(idx_left) >= 5
                and np.sum(idx_right) >= 5
                and len(np.unique(w_sorted[idx_left])) >= 2
                and len(np.unique(w_sorted[idx_right])) >= 2
            ):
                # Left segment: ROP = m1*w + k1
                p_left = np.polyfit(w_sorted[idx_left], rop_sorted[idx_left], deg=1)
                # Right segment: ROP = m2*w + k2
                p_right = np.polyfit(w_sorted[idx_right], rop_sorted[idx_right], deg=1)

                pred_left = np.polyval(p_left, w_sorted[idx_left])
                pred_right = np.polyval(p_right, w_sorted[idx_right])
                sse = np.sum((rop_sorted[idx_left] - pred_left) ** 2) + np.sum((rop_sorted[idx_right] - pred_right) ** 2)

                if sse < best_sse:
                    best_sse = sse
                    best_wf = wf
                    best_slope_linear = float(p_left[0])
                    best_slope_saturation = float(p_right[0])

        # ------------------------------------------------------------------
        # Method 3: MSE Inflection and Explosion
        # ------------------------------------------------------------------
        # Bin MSE by WOB intervals to find where MSE begins its steep climb
        n_bins = min(8, max(3, len(w_sorted) // 6))
        wob_bins = np.linspace(w_min, w_max, n_bins + 1)
        bin_wob_centers = []
        bin_mse_means = []

        for b_idx in range(n_bins):
            in_b = (w_sorted >= wob_bins[b_idx]) & (w_sorted < wob_bins[b_idx + 1])
            if np.sum(in_b) > 0:
                bin_wob_centers.append(float(np.mean(w_sorted[in_b])))
                bin_mse_means.append(float(np.mean(mse_sorted[in_b])))

        # ------------------------------------------------------------------
        # Synthesize Founder Point Detection
        # ------------------------------------------------------------------
        # Peak of quadratic model: dROP/dw = 2*a*w + b = 0 => w_peak = -b / (2*a)
        has_concavity = a < -1e-4

        detected_wob_founder: float
        regime: str

        if has_concavity:
            w_peak_poly = -b / (2.0 * a)
            if w_min <= w_peak_poly <= w_max * 1.05:
                # Clear parabolic peak within the tested range
                if best_wf is not None and best_slope_saturation < 0.3 * max(1e-4, best_slope_linear):
                    # Consensus between polynomial peak and broken-stick inflection
                    detected_wob_founder = float(0.6 * w_peak_poly + 0.4 * best_wf)
                else:
                    detected_wob_founder = float(w_peak_poly)
                regime = "founder_detected"
            elif w_peak_poly > w_max:
                # Founder point lies above current maximum tested WOB
                detected_wob_founder = float(w_max)
                regime = "pre_founder"
            else:
                # w_peak is lower than minimum WOB: data is mostly over-foundered
                detected_wob_founder = float(max(w_min, w_peak_poly))
                regime = "post_founder"
        else:
            # Upward or linear slope across range
            if best_wf is not None and best_slope_linear > 0 and best_slope_saturation <= 0.05 * best_slope_linear:
                detected_wob_founder = float(best_wf)
                regime = "founder_detected"
            elif np.mean(np.diff(rop_sorted)) > 0:
                detected_wob_founder = float(w_max)
                regime = "pre_founder"
            else:
                detected_wob_founder = float(w_min)
                regime = "post_founder"
        # Round detected founder WOB
        detected_wob_founder = round(detected_wob_founder, 2)
        # Safe Recommended WOB: typically 90% of founder WOB
        wob_recommended = round(detected_wob_founder * self.safety_factor, 2)

        # Evaluate ROP at founder and recommended WOB
        if has_concavity:
            rop_at_founder = float(np.polyval(poly_coeffs, detected_wob_founder))
            rop_at_rec = float(np.polyval(poly_coeffs, wob_recommended))
            d_rop = float(2.0 * a * wob_recommended + b)
        else:
            # Interpolate or extrapolate linearly
            rop_at_founder = float(np.interp(detected_wob_founder, w_sorted, rop_sorted))
            rop_at_rec = float(np.interp(wob_recommended, w_sorted, rop_sorted))
            d_rop = float(max(0.1, best_slope_linear if best_slope_linear > 0 else 1.0))

        # Evaluate MSE at founder
        mse_at_founder = float(np.interp(detected_wob_founder, w_sorted, mse_sorted))

        return FounderPointResult(
            wob_founder=round(detected_wob_founder, 2),
            wob_recommended=wob_recommended,
            rop_at_founder=round(max(0.1, rop_at_founder), 2),
            rop_at_recommended=round(max(0.1, rop_at_rec), 2),
            mse_at_founder=round(mse_at_founder, 1),
            confidence_score=round(r2_poly, 3),
            regime=regime,
            d_rop_d_wob=round(d_rop, 3),
            wob_tested_range=(round(w_min, 2), round(w_max, 2)),
            depth_interval=(round(d_min, 2), round(d_max, 2)),
            formation=fm_name,
            details={
                "poly_coeffs": [float(x) for x in poly_coeffs],
                "r2_poly": round(r2_poly, 3),
                "broken_stick_wf": round(best_wf, 2) if best_wf is not None else None,
                "broken_stick_slope_left": round(best_slope_linear, 3),
                "broken_stick_slope_right": round(best_slope_saturation, 3),
                "mse_bins_wob": [round(x, 1) for x in bin_wob_centers],
                "mse_bins_mean": [round(x, 1) for x in bin_mse_means],
            },
        )

    def analyze_by_formation(
        self,
        df: pd.DataFrame,
        formation_col: str = "formation",
        **kwargs,
    ) -> Dict[str, FounderPointResult]:
        """
        Segments telemetry by lithology/formation and detects founder point for each layer.

        Parameters
        ----------
        df : pd.DataFrame
            Drilling telemetry dataframe.
        formation_col : str
            Name of formation column (default: "formation").
        **kwargs : Any
            Additional arguments passed to `analyze()`.

        Returns
        -------
        Dict[str, FounderPointResult]
            Dictionary mapping formation names to their analysis results.
        """
        if formation_col not in df.columns:
            return {"all": self.analyze(df, **kwargs)}

        results = {}
        for fm, group in df.groupby(formation_col, sort=False):
            if len(group) >= self.min_samples:
                results[str(fm)] = self.analyze(group, formation_col=formation_col, **kwargs)
        return results

    def analyze_depth_windows(
        self,
        df: pd.DataFrame,
        window_size_m: float = 50.0,
        step_m: float = 25.0,
        depth_col: str = "depth",
        **kwargs,
    ) -> List[FounderPointResult]:
        """
        Performs a continuous sliding window analysis across the well trajectory.

        Parameters
        ----------
        df : pd.DataFrame
            Telemetry dataframe.
        window_size_m : float
            Window length in meters.
        step_m : float
            Step size in meters.
        depth_col : str
            Column representing measured depth.
        **kwargs : Any
            Additional arguments passed to `analyze()`.

        Returns
        -------
        List[FounderPointResult]
            Chronological list of founder point results along the wellbore.
        """
        if df.empty or depth_col not in df.columns:
            return []

        d_start = float(df[depth_col].min())
        d_end = float(df[depth_col].max())

        results = []
        cur_depth = d_start
        while cur_depth + window_size_m <= d_end + 1e-3:
            w_sub = df[(df[depth_col] >= cur_depth) & (df[depth_col] < cur_depth + window_size_m)]
            if len(w_sub) >= self.min_samples:
                res = self.analyze(w_sub, depth_col=depth_col, **kwargs)
                results.append(res)
            cur_depth += step_m

        return results
