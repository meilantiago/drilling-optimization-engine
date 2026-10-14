"""
Drilling Road Map & Operational Envelope Optimizer
===================================================
Recommends the optimal operating parameters (WOB, RPM, Flow Rate) to maximize
Rate of Penetration (ROP) while strictly respecting mechanical and hydraulic limits:
- WOB <= WOB_founder (safety factor to prevent bit balling and MSE flounder)
- Stick-Slip Severity Index (SSI) < 0.5 (eliminates torsional vibration)
- Surface Drillstring Torque <= Maximum Top Drive Continuous Rating
- Hydraulic Flow Rate within Annular Velocity and Standpipe Pressure (SPP) limits

Quantifies business ROI:
- Projected ROP improvement (%)
- Drilling time saved (hours)
- Direct rig cost savings in USD given a configurable Rig Day Rate ($/day).

Author: Tiago Meilan
Project: DrillOpt-Engine
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from .founder_point import FounderPointAnalyzer, FounderPointResult


@dataclass
class DrillingEnvelope:
    """Allowable and optimal operational windows for surface parameters."""
    wob_min_klbf: float
    wob_opt_klbf: float
    wob_max_klbf: float
    rpm_min: float
    rpm_opt: float
    rpm_max: float
    flow_min_gpm: float
    flow_opt_gpm: float
    flow_max_gpm: float


@dataclass
class OptimizationResult:
    """Holistic drilling optimization outcome with business ROI quantification."""
    baseline_rop_m_hr: float
    projected_rop_m_hr: float
    rop_gain_pct: float
    drilling_depth_m: float
    baseline_hours: float
    projected_hours: float
    hours_saved: float
    day_rate_usd: float
    cost_savings_usd: float
    envelope: DrillingEnvelope
    recommended_wob_klbf: float
    recommended_rpm: float
    recommended_flow_gpm: float
    wob_founder_limit_klbf: float
    initial_ssi: float
    projected_ssi: float
    estimated_torque_ft_lbf: float
    topdrive_torque_limit_ft_lbf: float
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def baseline_rop(self) -> float:
        """Alias for baseline_rop_m_hr."""
        return self.baseline_rop_m_hr

    @property
    def projected_rop(self) -> float:
        """Alias for projected_rop_m_hr."""
        return self.projected_rop_m_hr

    @property
    def recommended_wob(self) -> float:
        """Alias for recommended_wob_klbf."""
        return self.recommended_wob_klbf

    @property
    def recommended_flow(self) -> float:
        """Alias for recommended_flow_gpm."""
        return self.recommended_flow_gpm


class RoadMapOptimizer:
    """
    Generates actionable parameter road maps to maximize ROP safely,
    mitigate stick-slip, and calculate financial savings.
    """

    def __init__(
        self,
        founder_analyzer: Optional[FounderPointAnalyzer] = None,
        topdrive_max_torque_ft_lbf: float = 28000.0,
        max_rig_wob_klbf: float = 55.0,
        min_rig_wob_klbf: float = 8.0,
        max_rig_rpm: float = 180.0,
        min_rig_rpm: float = 60.0,
        max_rig_flow_gpm: float = 750.0,
        min_rig_flow_gpm: float = 380.0,
        max_spp_psi: float = 4500.0,
        default_day_rate_usd: float = 50000.0,
    ):
        """
        Parameters
        ----------
        founder_analyzer : Optional[FounderPointAnalyzer]
            Analyzer instance for founder point detection.
        topdrive_max_torque_ft_lbf : float
            Maximum continuous torque rating of the rig top drive (ft-lbf).
        max_rig_wob_klbf : float
            Maximum allowable Weight on Bit by drillstring mechanical rating (klbf).
        min_rig_wob_klbf : float
            Minimum functional Weight on Bit for cutting engagement (klbf).
        max_rig_rpm : float
            Maximum safe drillstring rotation speed (RPM).
        min_rig_rpm : float
            Minimum rotation speed to prevent differential sticking (RPM).
        max_rig_flow_gpm : float
            Mud pump flow rate ceiling (GPM).
        min_rig_flow_gpm : float
            Mud pump minimum flow rate for cuttings transport (GPM).
        max_spp_psi : float
            Maximum allowable standpipe pressure rating (psi).
        default_day_rate_usd : float
            Benchmark rig day rate in USD per 24-hour day ($50,000 default).
        """
        self.founder_analyzer = founder_analyzer or FounderPointAnalyzer()
        self.topdrive_max_torque_ft_lbf = topdrive_max_torque_ft_lbf
        self.max_rig_wob_klbf = max_rig_wob_klbf
        self.min_rig_wob_klbf = min_rig_wob_klbf
        self.max_rig_rpm = max_rig_rpm
        self.min_rig_rpm = min_rig_rpm
        self.max_rig_flow_gpm = max_rig_flow_gpm
        self.min_rig_flow_gpm = min_rig_flow_gpm
        self.max_spp_psi = max_spp_psi
        self.default_day_rate_usd = default_day_rate_usd

    def calculate_ssi_from_rpm(self, rpm_series: pd.Series) -> float:
        """
        Calculates Stick-Slip Severity Index from rotary speed variance:
        SSI = (RPM_max - RPM_min) / (2 * RPM_mean)
        """
        clean_rpm = rpm_series.dropna()
        if len(clean_rpm) < 2:
            return 0.1
        mean_rpm = float(np.mean(clean_rpm))
        if mean_rpm < 5.0:
            return 1.0
        # Robust 5th and 95th percentiles to avoid single-sample noise spikes
        p95 = float(np.percentile(clean_rpm, 95))
        p05 = float(np.percentile(clean_rpm, 5))
        return float(np.clip((p95 - p05) / (2.0 * mean_rpm), 0.0, 2.0))

    def optimize_interval(
        self,
        df: pd.DataFrame,
        founder_result: Optional[FounderPointResult] = None,
        day_rate_usd: Optional[float] = None,
        target_ssi_limit: float = 0.45,
        wob_col: str = "wob",
        rop_col: str = "rop",
        rpm_col: str = "rpm",
        torque_col: str = "torque",
        flow_col: str = "flow",
        depth_col: str = "depth",
        spp_col: str = "spp",
        ssi_col: Optional[str] = "ssi",
    ) -> OptimizationResult:
        """
        Generates the recommended operating envelope and quantifies business savings.

        Parameters
        ----------
        df : pd.DataFrame
            Drilling telemetry dataframe for the target interval.
        founder_result : Optional[FounderPointResult]
            Precomputed founder point result; if None, computed automatically.
        day_rate_usd : Optional[float]
            Rig day rate in USD (defaults to self.default_day_rate_usd).
        target_ssi_limit : float
            Upper limit for Stick-Slip Index (default: 0.45, strictly < 0.50).

        Returns
        -------
        OptimizationResult
            Full engineering envelope, parameter setpoint recommendations, and financial ROI.
        """
        rig_day_rate = day_rate_usd if day_rate_usd is not None else self.default_day_rate_usd

        # Filter active drilling rows
        active_df = self.founder_analyzer.filter_active_drilling(
            df, wob_col=wob_col, rop_col=rop_col, rpm_col=rpm_col
        )
        if active_df.empty:
            active_df = df.copy()

        # Baseline telemetry metrics
        base_rop = float(active_df[rop_col].median()) if rop_col in active_df.columns and not active_df.empty else 10.0
        base_wob = float(active_df[wob_col].median()) if wob_col in active_df.columns and not active_df.empty else 20.0
        base_rpm = float(active_df[rpm_col].median()) if rpm_col in active_df.columns and not active_df.empty else 120.0
        base_flow = float(active_df[flow_col].median()) if flow_col in active_df.columns and not active_df.empty else 550.0
        base_torque = float(active_df[torque_col].median()) if torque_col in active_df.columns and not active_df.empty else 7500.0

        if ssi_col and ssi_col in active_df.columns:
            base_ssi = float(active_df[ssi_col].mean())
        elif rpm_col in active_df.columns:
            base_ssi = self.calculate_ssi_from_rpm(active_df[rpm_col])
        else:
            base_ssi = 0.2

        # Measured depth interval
        if depth_col in df.columns and not df.empty:
            d_start = float(df[depth_col].min())
            d_end = float(df[depth_col].max())
            interval_depth_m = max(5.0, d_end - d_start)
        else:
            interval_depth_m = 100.0

        # Detect founder point if not provided
        if founder_result is None:
            founder_result = self.founder_analyzer.analyze(
                df,
                depth_col=depth_col,
                wob_col=wob_col,
                rop_col=rop_col,
                torque_col=torque_col,
                rpm_col=rpm_col,
            )

        # -------------------------------------------------------------
        # 1. WOB Optimization & Founder Point Constraint
        # -------------------------------------------------------------
        if not np.isnan(founder_result.wob_recommended) and founder_result.regime == "founder_detected":
            target_wob = founder_result.wob_recommended
            founder_limit = founder_result.wob_founder
        elif founder_result.regime == "pre_founder":
            # Formation has not foundered; safe to push WOB up by 15-20% towards founder limit
            target_wob = min(self.max_rig_wob_klbf * 0.85, base_wob * 1.18)
            founder_limit = max(target_wob * 1.12, base_wob * 1.25)
        elif founder_result.regime == "post_founder":
            # Back off WOB to relieve bit balling and cutter damage
            target_wob = max(self.min_rig_wob_klbf, base_wob * 0.82)
            founder_limit = base_wob * 0.90
        else:
            # Fallback
            target_wob = base_wob
            founder_limit = base_wob * 1.2

        # -------------------------------------------------------------
        # 2. Stick-Slip Mitigation & RPM Optimization (SSI < 0.5)
        # -------------------------------------------------------------
        if base_ssi >= target_ssi_limit:
            # Drillstring is experiencing torsional oscillations
            # Increasing RPM moves the system out of the stick-slip limit cycle
            ssi_excess = base_ssi - target_ssi_limit
            # Critical rotary speed scaling
            rpm_boost_factor = 1.0 + min(0.40, ssi_excess * 0.65)
            target_rpm = min(self.max_rig_rpm, base_rpm * rpm_boost_factor)

            # If RPM alone cannot fully attenuate severe stick-slip, trim WOB slightly
            if target_rpm >= self.max_rig_rpm * 0.95 and base_ssi > 0.8:
                target_wob = max(self.min_rig_wob_klbf, target_wob * 0.92)

            projected_ssi = float(np.clip(
                base_ssi * ((base_rpm / max(1.0, target_rpm)) ** 1.35) * ((target_wob / max(1.0, base_wob)) ** 0.5),
                0.10,
                target_ssi_limit,
            ))
        else:
            target_rpm = base_rpm
            projected_ssi = base_ssi

        # -------------------------------------------------------------
        # 3. Top Drive Torque Constraint
        # -------------------------------------------------------------
        # Estimate expected torque: T ~ mu * WOB * D_bit
        # Scale proportionally from baseline torque
        wob_torque_ratio = target_wob / max(1.0, base_wob)
        est_torque = base_torque * (wob_torque_ratio ** 1.05)

        # Check against topdrive limit
        if est_torque > self.topdrive_max_torque_ft_lbf:
            # Scale back WOB to respect 95% of topdrive continuous rating
            reduction_ratio = (self.topdrive_max_torque_ft_lbf * 0.95) / est_torque
            target_wob = max(self.min_rig_wob_klbf, target_wob * reduction_ratio)
            est_torque = self.topdrive_max_torque_ft_lbf * 0.95

        # Clamp WOB and RPM to rig envelope
        rec_wob = float(np.clip(target_wob, self.min_rig_wob_klbf, min(self.max_rig_wob_klbf, founder_limit)))
        rec_rpm = float(np.clip(target_rpm, self.min_rig_rpm, self.max_rig_rpm))

        # -------------------------------------------------------------
        # 4. Flow Rate & Hole Cleaning Optimization
        # -------------------------------------------------------------
        # Higher projected ROP requires proportional cuttings removal capacity
        rec_flow = float(np.clip(
            max(base_flow, base_flow * 1.05),
            self.min_rig_flow_gpm,
            self.max_rig_flow_gpm,
        ))

        # -------------------------------------------------------------
        # 5. Projected ROP Modeling
        # -------------------------------------------------------------
        # In the linear/sub-founder cutting regime:
        # ROP ~ (WOB / WOB_base)^0.95 * (RPM / RPM_base)^0.82
        # If over-foundering was present (WOB_base > founder_limit), backing off improves ROP
        if base_wob > founder_limit:
            # Relieving bit balling yields instant cutting recovery
            founder_recovery = 1.15
            wob_factor = (rec_wob / max(1.0, founder_limit)) ** 0.85
        else:
            founder_recovery = 1.0
            wob_factor = (rec_wob / max(1.0, base_wob)) ** 0.95

        rpm_factor = (rec_rpm / max(1.0, base_rpm)) ** 0.80

        # Stick-slip mitigation bonus: eliminating stick-slip restores cutting efficiency
        ssi_efficiency_boost = 1.0 + max(0.0, (base_ssi - projected_ssi) * 0.18)

        projected_rop = max(
            base_rop,
            base_rop * wob_factor * rpm_factor * founder_recovery * ssi_efficiency_boost,
        )

        rop_gain_pct = ((projected_rop - base_rop) / max(0.01, base_rop)) * 100.0

        # -------------------------------------------------------------
        # 6. Business ROI & Financial Quantification
        # -------------------------------------------------------------
        baseline_hours = round(interval_depth_m / max(0.1, base_rop), 2)
        projected_hours = round(interval_depth_m / max(0.1, projected_rop), 2)
        hours_saved = round(max(0.0, baseline_hours - projected_hours), 2)

        hourly_rate = rig_day_rate / 24.0
        cost_savings_usd = round(hours_saved * hourly_rate, 2)

        # -------------------------------------------------------------
        # 7. Construct Operating Envelope
        # -------------------------------------------------------------
        envelope = DrillingEnvelope(
            wob_min_klbf=round(max(self.min_rig_wob_klbf, rec_wob * 0.80), 1),
            wob_opt_klbf=round(rec_wob, 1),
            wob_max_klbf=round(min(self.max_rig_wob_klbf, founder_limit), 1),
            rpm_min=round(max(self.min_rig_rpm, rec_rpm * 0.85), 0),
            rpm_opt=round(rec_rpm, 0),
            rpm_max=round(min(self.max_rig_rpm, rec_rpm * 1.15), 0),
            flow_min_gpm=round(max(self.min_rig_flow_gpm, rec_flow * 0.90), 0),
            flow_opt_gpm=round(rec_flow, 0),
            flow_max_gpm=round(min(self.max_rig_flow_gpm, rec_flow * 1.10), 0),
        )

        summary = (
            f"DrillOpt Recommendation for {interval_depth_m:.1f} m Interval:\n"
            f"- Setpoints: WOB {rec_wob:.1f} klbf | RPM {rec_rpm:.0f} | Flow {rec_flow:.0f} gpm\n"
            f"- Performance: ROP {base_rop:.1f} -> {projected_rop:.1f} m/hr (+{rop_gain_pct:.1f}%)\n"
            f"- Stick-Slip Mitigation: SSI {base_ssi:.2f} -> {projected_ssi:.2f} (Target < {target_ssi_limit:.2f})\n"
            f"- Efficiency & ROI: {hours_saved:.2f} drilling hours saved | Projected Savings: ${cost_savings_usd:,.2f} USD "
            f"(Rig Day Rate: ${rig_day_rate:,.0f}/day)"
        )

        return OptimizationResult(
            baseline_rop_m_hr=round(base_rop, 2),
            projected_rop_m_hr=round(projected_rop, 2),
            rop_gain_pct=round(rop_gain_pct, 1),
            drilling_depth_m=round(interval_depth_m, 2),
            baseline_hours=round(baseline_hours, 2),
            projected_hours=round(projected_hours, 2),
            hours_saved=round(hours_saved, 2),
            day_rate_usd=round(rig_day_rate, 2),
            cost_savings_usd=round(cost_savings_usd, 2),
            envelope=envelope,
            recommended_wob_klbf=round(rec_wob, 1),
            recommended_rpm=round(rec_rpm, 0),
            recommended_flow_gpm=round(rec_flow, 0),
            wob_founder_limit_klbf=round(founder_limit, 1),
            initial_ssi=round(base_ssi, 2),
            projected_ssi=round(projected_ssi, 2),
            estimated_torque_ft_lbf=round(est_torque, 0),
            topdrive_torque_limit_ft_lbf=round(self.topdrive_max_torque_ft_lbf, 0),
            summary=summary,
            details={
                "founder_regime": founder_result.regime,
                "founder_wob": founder_result.wob_founder,
                "hourly_rig_rate_usd": round(hourly_rate, 2),
            },
        )

    def optimize_section(
        self,
        df: pd.DataFrame,
        section_length_m: Optional[float] = None,
        day_rate_usd: Optional[float] = None,
        **kwargs,
    ) -> OptimizationResult:
        """
        Optimizes a defined section of telemetry and scales savings to section length if provided.
        """
        res = self.optimize_interval(df, day_rate_usd=day_rate_usd, **kwargs)
        if section_length_m is not None and section_length_m > 0:
            b_hrs = round(section_length_m / max(0.1, res.baseline_rop_m_hr), 2)
            p_hrs = round(section_length_m / max(0.1, res.projected_rop_m_hr), 2)
            h_saved = round(max(0.0, b_hrs - p_hrs), 2)
            savings = round(h_saved * (res.day_rate_usd / 24.0), 2)
            new_res = OptimizationResult(
                baseline_rop_m_hr=res.baseline_rop_m_hr,
                projected_rop_m_hr=res.projected_rop_m_hr,
                rop_gain_pct=res.rop_gain_pct,
                drilling_depth_m=round(section_length_m, 2),
                baseline_hours=b_hrs,
                projected_hours=p_hrs,
                hours_saved=h_saved,
                day_rate_usd=res.day_rate_usd,
                cost_savings_usd=savings,
                envelope=res.envelope,
                recommended_wob_klbf=res.recommended_wob_klbf,
                recommended_rpm=res.recommended_rpm,
                recommended_flow_gpm=res.recommended_flow_gpm,
                wob_founder_limit_klbf=res.wob_founder_limit_klbf,
                initial_ssi=res.initial_ssi,
                projected_ssi=res.projected_ssi,
                estimated_torque_ft_lbf=res.estimated_torque_ft_lbf,
                topdrive_torque_limit_ft_lbf=res.topdrive_torque_limit_ft_lbf,
                summary=self.generate_executive_summary(res),
                details=res.details,
            )
            return new_res
        return res

    def generate_executive_summary(self, result: OptimizationResult) -> str:
        """Generates an executive briefing markdown summary of the optimization result."""
        return (
            f"### Executive Drilling Optimization Summary\n\n"
            f"- **Depth Interval**: {result.drilling_depth_m:.1f} m\n"
            f"- **Rate of Penetration**: {result.baseline_rop_m_hr:.1f} m/hr ➔ **{result.projected_rop_m_hr:.1f} m/hr** "
            f"(+{result.rop_gain_pct:.1f}%)\n"
            f"- **Recommended Setpoints**: WOB **{result.recommended_wob_klbf:.1f} klbf** | "
            f"RPM **{result.recommended_rpm:.0f}** | Flow **{result.recommended_flow_gpm:.0f} gpm**\n"
            f"- **Founder Point Limit**: {result.wob_founder_limit_klbf:.1f} klbf\n"
            f"- **Stick-Slip Mitigation**: SSI {result.initial_ssi:.2f} ➔ **{result.projected_ssi:.2f}**\n"
            f"- **Top Drive Continuous Torque**: {result.estimated_torque_ft_lbf:,.0f} ft-lbf / {result.topdrive_torque_limit_ft_lbf:,.0f} ft-lbf\n\n"
            f"**Economic Value Created**:\n"
            f"- Rig Time Saved: **{result.hours_saved:.1f} hours**\n"
            f"- Direct Cost Savings: **${result.cost_savings_usd:,.2f} USD** (at ${result.day_rate_usd:,.0f}/day day rate)"
        )

    def build_what_if_matrix(
        self,
        df: pd.DataFrame,
        wob_candidates_klbf: Sequence[float],
        rpm_candidates: Sequence[float],
        day_rate_usd: Optional[float] = None,
        wob_col: str = "wob",
        rop_col: str = "rop",
        rpm_col: str = "rpm",
        torque_col: str = "torque",
        depth_col: str = "depth",
    ) -> pd.DataFrame:
        """
        Evaluates a 2D parameter grid (WOB x RPM) computing projected ROP, SSI,
        drilling hours saved, and cost savings for each combination.
        """
        rig_day_rate = day_rate_usd if day_rate_usd is not None else self.default_day_rate_usd
        active_df = self.founder_analyzer.filter_active_drilling(df, wob_col=wob_col, rop_col=rop_col, rpm_col=rpm_col)
        if active_df.empty:
            active_df = df.copy()

        base_rop = float(active_df[rop_col].median()) if rop_col in active_df.columns and not active_df.empty else 10.0
        base_wob = float(active_df[wob_col].median()) if wob_col in active_df.columns and not active_df.empty else 20.0
        base_rpm = float(active_df[rpm_col].median()) if rpm_col in active_df.columns and not active_df.empty else 120.0
        base_torque = float(active_df[torque_col].median()) if torque_col in active_df.columns and not active_df.empty else 7500.0

        if depth_col in df.columns and not df.empty:
            interval_depth = max(5.0, float(df[depth_col].max()) - float(df[depth_col].min()))
        else:
            interval_depth = 100.0

        founder_res = self.founder_analyzer.analyze(df, depth_col=depth_col, wob_col=wob_col, rop_col=rop_col)
        wob_founder = founder_res.wob_founder if not np.isnan(founder_res.wob_founder) else 45.0

        records = []
        hourly_rate = rig_day_rate / 24.0
        base_hours = interval_depth / max(0.1, base_rop)

        for w in wob_candidates_klbf:
            for r in rpm_candidates:
                # WOB factor: if w <= founder, linear cutting response, else founder penalty
                if w <= wob_founder:
                    wob_factor = (w / max(1.0, base_wob)) ** 0.95
                else:
                    over_ratio = w / max(1.0, wob_founder)
                    wob_factor = (wob_founder / max(1.0, base_wob)) ** 0.95 * np.exp(-1.5 * (over_ratio - 1.0))

                rpm_factor = (r / max(1.0, base_rpm)) ** 0.80
                proj_rop = max(0.2, base_rop * wob_factor * rpm_factor)
                rop_gain = ((proj_rop - base_rop) / max(0.01, base_rop)) * 100.0

                est_torque = base_torque * ((w / max(1.0, base_wob)) ** 1.05)
                est_ssi = float(np.clip(
                    0.25 * ((base_rpm / max(1.0, r)) ** 1.35) * ((w / max(1.0, base_wob)) ** 0.6),
                    0.05,
                    2.0,
                ))

                proj_hours = interval_depth / proj_rop
                hours_saved = max(0.0, base_hours - proj_hours)
                cost_savings = hours_saved * hourly_rate

                is_safe = (
                    w <= wob_founder
                    and est_torque <= self.topdrive_max_torque_ft_lbf
                    and est_ssi < 0.50
                )

                records.append({
                    "wob_klbf": round(w, 1),
                    "rpm": round(r, 0),
                    "projected_rop_m_hr": round(proj_rop, 2),
                    "rop_gain_pct": round(rop_gain, 1),
                    "projected_ssi": round(est_ssi, 2),
                    "estimated_torque_ft_lbf": round(est_torque, 0),
                    "hours_saved": round(hours_saved, 2),
                    "cost_savings_usd": round(cost_savings, 2),
                    "is_safe_window": is_safe,
                })

        return pd.DataFrame(records)

