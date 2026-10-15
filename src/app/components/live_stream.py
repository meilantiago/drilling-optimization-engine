"""
Real-Time Telemetry Streamer & Live Drilling Oscilloscope Component
====================================================================
Simulates live MWD/LWD downhole telemetry streaming into a Real-Time Operations Center (RTOC).
Renders instantaneous rig gauges, high-frequency rolling oscilloscopes,
and automated real-time SPE 92194 advisory alarms.

Author: Tiago Meilan
Project: DrillOpt-Engine
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


BG_COLOR = "#0b111e"
PLOT_BG = "#10192d"
GRID_COLOR = "#1f2d48"
TEXT_COLOR = "#e2e8f0"


def evaluate_realtime_alarm(row: pd.Series) -> Dict[str, str]:
    """
    Evaluates real-time drilling physics at the current bit position and generates
    instantaneous SPE 92194 alarms and operational recommendations.
    """
    wob = float(row.get("wob", 20.0))
    founder_wob = float(row.get("founder_wob", 30.0))
    ssi = float(row.get("ssi", 0.2))
    mse = float(row.get("mse_psi", 10000.0))
    ucs = float(row.get("ucs_psi", 10000.0))
    formation = str(row.get("formation", "Unknown"))
    rop = float(row.get("rop", 20.0))

    over_energy = mse / max(100.0, ucs)

    # 1. Severe Stick-Slip
    if ssi >= 1.0:
        return {
            "severity": "CRITICAL",
            "badge_class": "badge-crit",
            "title": f"🚨 CRITICAL ALARM: Torsional Stick-Slip Detected in {formation}",
            "diagnostic": (
                f"Stick-Slip Index is {ssi:.2f} (Full stall-whip cycle). The drill bit is periodically "
                f"stopping at 0 RPM and whipping up to >2.5x surface speed. Severe risk of twist-off or PDC cutter impact fracture."
            ),
            "recommendation": "IMMEDIATE ACTION: Reduce WOB by 4-6 klbf and increase surface RPM (+25 RPM) to break out of torsional resonance.",
            "color": "#ef4444",
        }
    # 2. Moderate Stick-Slip
    elif ssi >= 0.5:
        return {
            "severity": "WARNING",
            "badge_class": "badge-warn",
            "title": f"⚠️ CAUTION: Torsional Oscillation (SSI = {ssi:.2f})",
            "diagnostic": f"Elevated torsional vibration developing in {formation}. Energy dissipation is increasing.",
            "recommendation": "ADVISORY: Boost surface RPM by 10-15 RPM and monitor torque variance closely.",
            "color": "#f59e0b",
        }
    # 3. Founder Point Saturation / Over-foundering
    elif wob >= founder_wob and over_energy > 2.2:
        return {
            "severity": "WARNING",
            "badge_class": "badge-warn",
            "title": f"⚠️ FOUNDER POINT DETECTED @ {float(row.get('depth', 0)):.1f}m",
            "diagnostic": (
                f"Current WOB ({wob:.1f} klbf) exceeds formation founder limit ({founder_wob:.1f} klbf). "
                f"Mechanical Specific Energy has spiked to {mse:,.0f} psi ({over_energy:.1f}x UCS) with no ROP gain ({rop:.1f} m/hr)."
            ),
            "recommendation": f"CORRECTIVE ACTION: Back off WOB to {(founder_wob * 0.9):.1f} klbf (90% founder threshold) to restore shearing efficiency.",
            "color": "#f59e0b",
        }
    # 4. Clean shearing / optimal efficiency
    else:
        return {
            "severity": "NORMAL",
            "badge_class": "badge-safe",
            "title": f"✅ OPTIMAL DRILLING: Efficient Shearing in {formation}",
            "diagnostic": f"Drilling efficiently at {rop:.1f} m/hr. MSE ({mse:,.0f} psi) is aligned with rock strength ({ucs:,.0f} psi).",
            "recommendation": "STATUS: Maintain current operating parameters (WOB & RPM).",
            "color": "#10b981",
        }


def create_live_oscilloscope_plot(df_window: pd.DataFrame) -> go.Figure:
    """
    Creates a 4-channel real-time strip-chart / oscilloscope showing high-frequency
    telemetry waveforms over the recent drilling window.
    """
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "Rate of Penetration (m/hr) & Weight on Bit (klbf)",
            "Rotary Speed (Surface vs Downhole Bit RPM)",
            "Surface Drilling Torque (ft-lbf)",
            "Mechanical Specific Energy (MSE) vs Rock UCS (psi)",
        ),
    )

    x_depth = df_window["depth"].values

    # Track 1: ROP & WOB
    fig.add_trace(
        go.Scatter(
            x=x_depth,
            y=df_window["rop"],
            name="ROP (m/hr)",
            line=dict(color="#06b6d4", width=2),
            mode="lines",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_depth,
            y=df_window["wob"],
            name="WOB (klbf)",
            line=dict(color="#38bdf8", width=1.5, dash="dot"),
            mode="lines",
            yaxis="y5",
        ),
        row=1, col=1,
    )

    # Track 2: Surface RPM vs Bit RPM (showing stick-slip fluctuations)
    surface_rpm = df_window["rpm"].values
    ssi = df_window.get("ssi", pd.Series(0.2, index=df_window.index)).values
    # Downhole bit RPM simulation based on stick-slip index
    bit_rpm = surface_rpm * (1.0 + ssi * np.sin(np.linspace(0, 8 * np.pi, len(df_window))))
    bit_rpm = np.maximum(0.0, bit_rpm)

    fig.add_trace(
        go.Scatter(
            x=x_depth,
            y=surface_rpm,
            name="Surface RPM",
            line=dict(color="#c084fc", width=1.5),
            mode="lines",
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_depth,
            y=bit_rpm,
            name="Bit RPM (Downhole)",
            line=dict(color="#f43f5e", width=2),
            mode="lines",
        ),
        row=2, col=1,
    )

    # Track 3: Torque
    fig.add_trace(
        go.Scatter(
            x=x_depth,
            y=df_window["torque"],
            name="Torque (ft-lbf)",
            line=dict(color="#f59e0b", width=2),
            mode="lines",
        ),
        row=3, col=1,
    )

    # Track 4: MSE vs UCS
    fig.add_trace(
        go.Scatter(
            x=x_depth,
            y=df_window["mse_psi"],
            name="Teale MSE (psi)",
            line=dict(color="#ef4444", width=2),
            mode="lines",
        ),
        row=4, col=1,
    )
    if "ucs_psi" in df_window.columns:
        fig.add_trace(
            go.Scatter(
                x=x_depth,
                y=df_window["ucs_psi"],
                name="Rock UCS (psi)",
                line=dict(color="#94a3b8", width=1.5, dash="dash"),
                mode="lines",
            ),
            row=4, col=1,
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT_COLOR, size=10),
        height=680,
        margin=dict(l=50, r=30, t=40, b=30),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )

    fig.update_xaxes(gridcolor=GRID_COLOR, showgrid=True, title_text="Measured Depth (m)", row=4, col=1)
    for r in range(1, 5):
        fig.update_yaxes(gridcolor=GRID_COLOR, showgrid=True, row=r, col=1)

    return fig
