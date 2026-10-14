"""
Interactive Drilling Parameter Simulator & Financial ROI Panel
==============================================================
Provides real-time "What-If" parameter simulation:
- Interactive parameter sliders for candidate WOB, RPM, Flow Rate, and Rig Day Rate.
- Instant recalculation of projected ROP, drilling hours saved, and direct USD financial ROI.
- Operational envelope validation against Founder Point, Top Drive continuous torque, and Stick-Slip SSI limits.
- 2D sensitivity response heatmap (WOB vs RPM) mapping projected ROP across operational space.

Author: Tiago Meilan
Project: DrillOpt-Engine
"""

from typing import Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.optimizer.road_map import RoadMapOptimizer, OptimizationResult
from src.optimizer.founder_point import FounderPointAnalyzer

BG_COLOR = "#0b111e"
PLOT_BG = "#10192d"
GRID_COLOR = "#1f2d48"
TEXT_COLOR = "#e2e8f0"


def render_simulator_panel(
    df: pd.DataFrame,
    default_day_rate_usd: float = 50000.0,
    bit_diameter_in: float = 8.5,
) -> None:
    """
    Renders the complete interactive What-If Parameter Simulator in Streamlit.

    Parameters
    ----------
    df : pd.DataFrame
        Active drilling telemetry dataset.
    default_day_rate_usd : float
        Current rig day rate in USD.
    bit_diameter_in : float
        Bit diameter in inches.
    """
    st.markdown("### Real-Time Drilling Parameter Optimization & ROI Simulator")
    st.markdown(
        "Explore how varying **Weight on Bit (WOB)** and **Rotary Speed (RPM)** impacts "
        "penetration rate, stick-slip mitigation, and tangible rig cost savings in USD."
    )

    # Initialize optimizer with specified bit diameter
    founder_analyzer = FounderPointAnalyzer(bit_diameter_in=bit_diameter_in)
    optimizer = RoadMapOptimizer(
        founder_analyzer=founder_analyzer,
        default_day_rate_usd=default_day_rate_usd,
    )

    # Baseline telemetry values from active drilling data
    active_df = founder_analyzer.filter_active_drilling(df)
    if active_df.empty:
        active_df = df

    base_rop = float(active_df["rop"].median()) if "rop" in active_df.columns else 12.0
    base_wob = float(active_df["wob"].median()) if "wob" in active_df.columns else 18.0
    base_rpm = float(active_df["rpm"].median()) if "rpm" in active_df.columns else 110.0
    base_flow = float(active_df["flow"].median()) if "flow" in active_df.columns else 550.0
    base_ssi = float(active_df["ssi"].mean()) if "ssi" in active_df.columns else 0.42

    # Calculate founder point result
    founder_res = founder_analyzer.analyze(active_df)
    wob_founder = founder_res.wob_founder if not np.isnan(founder_res.wob_founder) else 36.0
    wob_rec_auto = founder_res.wob_recommended if not np.isnan(founder_res.wob_recommended) else round(wob_founder * 0.90, 1)

    # Calculate optimal interval result
    opt_result = optimizer.optimize_interval(active_df, founder_result=founder_res, day_rate_usd=default_day_rate_usd)

    # =========================================================================
    # Controls & Parameter Inputs
    # =========================================================================
    st.markdown("---")
    col_ctrl_1, col_ctrl_2, col_ctrl_3, col_ctrl_4 = st.columns(4)

    with col_ctrl_1:
        st.markdown("**Weight on Bit (WOB)**")
        cand_wob = st.slider(
            "Candidate WOB (klbf)",
            min_value=6.0,
            max_value=50.0,
            value=float(wob_rec_auto),
            step=0.5,
            help=f"Detected Founder Point: {wob_founder:.1f} klbf. Values above this risk severe bit balling.",
        )

    with col_ctrl_2:
        st.markdown("**Rotary Speed (RPM)**")
        cand_rpm = st.slider(
            "Candidate RPM",
            min_value=50.0,
            max_value=200.0,
            value=float(opt_result.recommended_rpm),
            step=5.0,
            help="Higher RPM helps suppress stick-slip torsional limit cycles.",
        )

    with col_ctrl_3:
        st.markdown("**Rig Economics**")
        cand_day_rate = st.number_input(
            "Rig Day Rate ($/day)",
            min_value=15000.0,
            max_value=250000.0,
            value=float(default_day_rate_usd),
            step=5000.0,
            format="%.0f",
            help="Total operational spread cost per 24-hour day.",
        )

    with col_ctrl_4:
        st.markdown("**Target Interval**")
        section_length = st.number_input(
            "Interval Length (m)",
            min_value=20.0,
            max_value=2000.0,
            value=float(max(50.0, opt_result.drilling_depth_m)),
            step=50.0,
            help="Total footage/depth over which the new parameters will be maintained.",
        )

    # =========================================================================
    # Physics & Economics Recalculation
    # =========================================================================
    # 1. WOB Factor (linear vs founder penalty)
    if cand_wob <= wob_founder:
        wob_factor = (cand_wob / max(1.0, base_wob)) ** 0.95
    else:
        over_ratio = cand_wob / max(1.0, wob_founder)
        wob_factor = (wob_founder / max(1.0, base_wob)) ** 0.95 * np.exp(-1.5 * (over_ratio - 1.0))

    # 2. RPM Factor
    rpm_factor = (cand_rpm / max(1.0, base_rpm)) ** 0.80

    # 3. Projected SSI
    cand_ssi = float(np.clip(
        base_ssi * ((base_rpm / max(1.0, cand_rpm)) ** 1.35) * ((cand_wob / max(1.0, base_wob)) ** 0.5),
        0.05,
        2.0,
    ))
    ssi_boost = 1.0 + max(0.0, (base_ssi - cand_ssi) * 0.18)

    # 4. Projected ROP
    proj_rop = max(0.5, base_rop * wob_factor * rpm_factor * ssi_boost)
    rop_gain_pct = ((proj_rop - base_rop) / max(0.01, base_rop)) * 100.0

    # 5. Torque Estimation
    base_torque = float(active_df["torque"].median()) if "torque" in active_df.columns else 7000.0
    est_torque = base_torque * ((cand_wob / max(1.0, base_wob)) ** 1.05)

    # 6. Time and Dollar Savings
    base_hours = section_length / max(0.1, base_rop)
    proj_hours = section_length / max(0.1, proj_rop)
    hours_saved = max(0.0, base_hours - proj_hours)
    hourly_rate = cand_day_rate / 24.0
    cost_savings = hours_saved * hourly_rate

    # Operational status checks
    is_over_founder = cand_wob > wob_founder
    is_high_torque = est_torque > 25000.0
    is_high_ssi = cand_ssi >= 0.50

    # =========================================================================
    # Executive KPI Cards
    # =========================================================================
    st.markdown("---")
    col_kpi_1, col_kpi_2, col_kpi_3, col_kpi_4 = st.columns(4)

    with col_kpi_1:
        st.metric(
            label="Projected ROP",
            value=f"{proj_rop:.1f} m/hr",
            delta=f"{rop_gain_pct:+.1f}% vs Baseline ({base_rop:.1f})",
        )

    with col_kpi_2:
        st.metric(
            label="Drilling Time Saved",
            value=f"{hours_saved:.1f} hrs",
            delta=f"{base_hours:.1f} ➔ {proj_hours:.1f} hrs",
            delta_color="normal",
        )

    with col_kpi_3:
        st.metric(
            label="Direct Cost Savings (USD)",
            value=f"${cost_savings:,.0f}",
            delta=f"${cand_day_rate:,.0f}/day rate",
            delta_color="normal",
        )

    with col_kpi_4:
        ssi_status = "SAFE" if cand_ssi < 0.5 else ("WARN" if cand_ssi < 1.0 else "SEVERE")
        st.metric(
            label="Projected Stick-Slip (SSI)",
            value=f"{cand_ssi:.2f} ({ssi_status})",
            delta=f"Baseline: {base_ssi:.2f}",
            delta_color="inverse" if cand_ssi > base_ssi else "normal",
        )

    # Operational Warnings
    if is_over_founder:
        st.error(
            f"⚠️ **FOUNDER EXCEEDED**: Candidate WOB ({cand_wob:.1f} klbf) exceeds the detected "
            f"founder point ({wob_founder:.1f} klbf). Floundering and severe bit balling will occur!"
        )
    elif is_high_ssi:
        st.warning(
            f"⚠️ **STICK-SLIP RISK**: Projected SSI ({cand_ssi:.2f}) exceeds the 0.50 SPE threshold. "
            f"Increase RPM or moderate WOB to eliminate torsional oscillation."
        )
    elif is_high_torque:
        st.warning(
            f"⚠️ **HIGH TORQUE**: Estimated torque ({est_torque:,.0f} ft-lbf) approaches top drive continuous limit."
        )
    else:
        st.success(
            f"✅ **OPTIMAL OPERATING WINDOW**: Parameters are strictly within safe mechanical limits. "
            f"WOB is below founder point ({wob_founder:.1f} klbf) and SSI < 0.50."
        )

    # =========================================================================
    # 2D Parameter Response Heatmap (WOB vs RPM)
    # =========================================================================
    st.markdown("#### 2D Sensitivity Response Surface: Projected ROP & Safe Window")

    w_grid = np.linspace(8.0, 48.0, 25)
    r_grid = np.linspace(60.0, 180.0, 25)

    z_rop = np.zeros((len(r_grid), len(w_grid)))
    for i, r in enumerate(r_grid):
        for j, w in enumerate(w_grid):
            if w <= wob_founder:
                wf = (w / max(1.0, base_wob)) ** 0.95
            else:
                wf = (wob_founder / max(1.0, base_wob)) ** 0.95 * np.exp(-1.5 * (w / wob_founder - 1.0))
            rf = (r / max(1.0, base_rpm)) ** 0.80
            s_est = float(np.clip(base_ssi * ((base_rpm / r) ** 1.35) * ((w / base_wob) ** 0.5), 0.05, 2.0))
            s_boost = 1.0 + max(0.0, (base_ssi - s_est) * 0.18)
            z_rop[i, j] = max(0.5, base_rop * wf * rf * s_boost)

    fig_heat = go.Figure(
        data=go.Contour(
            z=z_rop,
            x=w_grid,
            y=r_grid,
            colorscale="Viridis",
            contours=dict(showlabels=True, labelfont=dict(size=10, color="white")),
            colorbar=dict(title="Projected ROP (m/hr)"),
            hovertemplate="WOB: %{x:.1f} klbf<br>RPM: %{y:.0f}<br>ROP: %{z:.1f} m/hr<extra></extra>",
        )
    )

    # Vertical line marking founder limit
    fig_heat.add_vline(
        x=wob_founder,
        line=dict(color="#ef4444", width=2.5, dash="dash"),
        annotation_text="Founder Limit",
        annotation_position="top right",
        annotation_font=dict(color="#ef4444", size=11, family="Segoe UI Semibold"),
    )

    # Marker of current candidate selection
    fig_heat.add_trace(
        go.Scatter(
            x=[cand_wob],
            y=[cand_rpm],
            mode="markers+text",
            marker=dict(symbol="cross", size=14, color="#ef4444", line=dict(color="#ffffff", width=2)),
            text=["CURRENT SELECTION"],
            textposition="top center",
            textfont=dict(color="#ef4444", size=11, family="Segoe UI Semibold"),
            name="Selected Setpoint",
        )
    )

    fig_heat.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=PLOT_BG,
        height=450,
        margin=dict(l=55, r=25, t=40, b=45),
        xaxis=dict(title="Weight on Bit (klbf)", gridcolor=GRID_COLOR),
        yaxis=dict(title="Rotary Speed (RPM)", gridcolor=GRID_COLOR),
    )

    st.plotly_chart(fig_heat, use_container_width=True)

    # =========================================================================
    # Operational Road Map Envelope Card
    # =========================================================================
    env = opt_result.envelope
    st.markdown("#### Recommended Engineering Envelope Setpoints")
    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        st.info(
            f"**WOB Envelope (klbf)**\n\n"
            f"- Min: `{env.wob_min_klbf:.1f}`\n"
            f"- **Optimal: `{env.wob_opt_klbf:.1f}`**\n"
            f"- Max (Founder Limit): `{env.wob_max_klbf:.1f}`"
        )
    with col_e2:
        st.info(
            f"**RPM Envelope**\n\n"
            f"- Min: `{env.rpm_min:.0f}`\n"
            f"- **Optimal: `{env.rpm_opt:.0f}`**\n"
            f"- Max: `{env.rpm_max:.0f}`"
        )
    with col_e3:
        st.info(
            f"**Flow Rate Envelope (gpm)**\n\n"
            f"- Min: `{env.flow_min_gpm:.0f}`\n"
            f"- **Optimal: `{env.flow_opt_gpm:.0f}`**\n"
            f"- Max: `{env.flow_max_gpm:.0f}`"
        )
