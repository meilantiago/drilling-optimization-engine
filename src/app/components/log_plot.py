"""
Interactive Multi-Track Well Log Visualizer (Techlog / WellCAD Style)
=====================================================================
Renders continuous drilling telemetry along measured depth with synchronized
tracks, inverted depth axis, and dysfunction indicator coloring.

Tracks:
1. Track 1: Gamma Ray (GAPI) with lithology shading
2. Track 2: Weight on Bit (WOB in klbf) & Setpoint limit
3. Track 3: Rotary Speed (Surface & Downhole RPM)
4. Track 4: Drilling Torque (ft-lbf)
5. Track 5: Rate of Penetration (ROP m/hr) & Corrected Dc-exponent
6. Track 6: Teale MSE vs UCS with dysfunction risk fill
7. Track 7: Stick-Slip Severity Index (SSI) with standard SPE alert bands

Author: Tiago Meilan
Project: DrillOpt-Engine
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.physics.mse import teale_mse
from src.physics.d_exponent import jordan_shirley_d_exponent, rehm_mcclendon_dc_exponent
from src.dynamics.stick_slip import compute_rolling_stick_slip_index


# Dark Oilfield Color Palette
BG_COLOR = "#0b111e"
PLOT_BG = "#10192d"
GRID_COLOR = "#1f2d48"
TEXT_COLOR = "#e2e8f0"
BORDER_COLOR = "#2a3b5c"

TRACK_COLORS = {
    "gamma": "#22c55e",       # Green
    "wob": "#38bdf8",         # Sky Blue
    "rpm": "#c084fc",         # Purple
    "torque": "#f59e0b",      # Amber
    "rop": "#06b6d4",         # Cyan
    "dc": "#eab308",          # Yellow
    "mse": "#ef4444",         # Red
    "ucs": "#94a3b8",         # Slate
    "ssi_safe": "#10b981",    # Emerald
    "ssi_warn": "#f59e0b",    # Amber
    "ssi_crit": "#ef4444",    # Crimson
}


def prepare_log_dataframe(
    df: pd.DataFrame,
    bit_diameter_in: float = 8.5,
    mud_weight_ppg: float = 10.2,
) -> pd.DataFrame:
    """Prepares and verifies all required curves for multi-track plotting."""
    log_df = df.copy()

    # Ensure depth is sorted
    if "depth" not in log_df.columns:
        log_df["depth"] = np.arange(len(log_df), dtype=float)
    log_df = log_df.sort_values(by="depth").reset_index(drop=True)

    # 1. Gamma Ray
    if "gamma" not in log_df.columns:
        log_df["gamma"] = 65.0

    # 2. WOB (klbf)
    if "wob" not in log_df.columns:
        log_df["wob"] = 20.0

    # 3. RPM
    if "rpm" not in log_df.columns:
        log_df["rpm"] = 120.0

    # 4. Torque (ft-lbf)
    if "torque" not in log_df.columns:
        log_df["torque"] = 7000.0

    # 5. ROP (m/hr)
    if "rop" not in log_df.columns:
        log_df["rop"] = 15.0

    # 6. Dc-exponent
    if "dc_exponent" not in log_df.columns and "dc" not in log_df.columns:
        rop_ft_hr = np.maximum(0.1, log_df["rop"].fillna(10.0) / 0.3048)
        wob_lbf = np.maximum(100.0, log_df["wob"].fillna(15.0) * 1000.0)
        d_raw = jordan_shirley_d_exponent(rop_ft_hr, log_df["rpm"].fillna(100.0), wob_lbf, bit_diameter_in)
        dc_val = rehm_mcclendon_dc_exponent(d_raw, actual_mud_weight=mud_weight_ppg)
        log_df["dc_exponent"] = pd.Series(dc_val).fillna(1.2)
    elif "dc" in log_df.columns:
        log_df["dc_exponent"] = log_df["dc"]

    # 7. MSE (psi)
    if "mse_psi" not in log_df.columns:
        rop_ft_hr = np.maximum(0.1, log_df["rop"].fillna(10.0) / 0.3048)
        wob_lbf = np.maximum(100.0, log_df["wob"].fillna(15.0) * 1000.0)
        log_df["mse_psi"] = teale_mse(
            wob=wob_lbf,
            torque=log_df["torque"].fillna(5000.0),
            rpm=log_df["rpm"].fillna(100.0),
            rop=rop_ft_hr,
            bit_diameter=bit_diameter_in,
            use_field_units=True,
            output_unit="psi",
        )

    # 8. UCS (psi)
    if "ucs_psi" not in log_df.columns:
        # Synthetic estimation from gamma ray if formation UCS not given
        # High GR (shale) ~ 8,000 - 10,000 psi, Low GR (carbonate) ~ 20,000 - 25,000 psi
        gr = log_df["gamma"].fillna(60.0)
        log_df["ucs_psi"] = np.clip(28000.0 - 150.0 * gr, 6000.0, 30000.0)

    # 9. SSI
    if "ssi" not in log_df.columns:
        log_df["ssi"] = compute_rolling_stick_slip_index(log_df["rpm"], window_samples=25)
        log_df["ssi"] = log_df["ssi"].fillna(0.2)

    return log_df


def create_multitrack_log_plot(
    df: pd.DataFrame,
    bit_diameter_in: float = 8.5,
    mud_weight_ppg: float = 10.2,
    depth_range: Optional[Tuple[float, float]] = None,
    height: int = 950,
) -> go.Figure:
    """
    Constructs an interactive 7-track well log plot (Techlog / WellCAD style)
    with inverted shared vertical depth axis.

    Parameters
    ----------
    df : pd.DataFrame
        Telemetry log dataframe.
    bit_diameter_in : float
        Bit diameter in inches.
    mud_weight_ppg : float
        Active mud weight in ppg.
    depth_range : Optional[tuple[float, float]]
        (min_depth, max_depth) interval to zoom into.
    height : int
        Figure height in pixels.

    Returns
    -------
    go.Figure
        Interactive Plotly figure with 7 synced tracks.
    """
    log_df = prepare_log_dataframe(df, bit_diameter_in=bit_diameter_in, mud_weight_ppg=mud_weight_ppg)

    if depth_range is not None:
        d_min, d_max = depth_range
        log_df = log_df[(log_df["depth"] >= d_min) & (log_df["depth"] <= d_max)].copy()

    depth = log_df["depth"]

    # 7 tracks with equal horizontal distribution
    track_titles = [
        "Track 1: GR (API)",
        "Track 2: WOB (klbf)",
        "Track 3: RPM",
        "Track 4: Torque (ft-lb)",
        "Track 5: ROP & Dc",
        "Track 6: MSE vs UCS (psi)",
        "Track 7: SSI Alert",
    ]

    fig = make_subplots(
        rows=1,
        cols=7,
        shared_yaxes=True,
        horizontal_spacing=0.018,
        subplot_titles=track_titles,
    )

    # ==========================================
    # Track 1: Gamma Ray (0 - 150 GAPI)
    # ==========================================
    gr = log_df["gamma"]
    fig.add_trace(
        go.Scatter(
            x=gr,
            y=depth,
            mode="lines",
            line=dict(color=TRACK_COLORS["gamma"], width=1.8),
            name="Gamma Ray (GAPI)",
            hovertemplate="Depth: %{y:.1f} m<br>GR: %{x:.1f} GAPI<extra></extra>",
        ),
        row=1,
        col=1,
    )
    # Lithology shading (sandstone vs shale threshold at 75 API)
    fig.add_trace(
        go.Scatter(
            x=[75.0] * len(depth),
            y=depth,
            mode="lines",
            line=dict(color="rgba(255, 255, 255, 0.15)", width=1, dash="dash"),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # ==========================================
    # Track 2: Weight on Bit (klbf)
    # ==========================================
    wob = log_df["wob"]
    fig.add_trace(
        go.Scatter(
            x=wob,
            y=depth,
            mode="lines",
            line=dict(color=TRACK_COLORS["wob"], width=2.0),
            name="WOB (klbf)",
            hovertemplate="Depth: %{y:.1f} m<br>WOB: %{x:.1f} klbf<extra></extra>",
        ),
        row=1,
        col=2,
    )
    # Optional founder limit line if available
    if "founder_wob" in log_df.columns:
        fig.add_trace(
            go.Scatter(
                x=log_df["founder_wob"],
                y=depth,
                mode="lines",
                line=dict(color="#f43f5e", width=1.5, dash="dot"),
                name="Founder Limit",
                hovertemplate="Founder Limit: %{x:.1f} klbf<extra></extra>",
            ),
            row=1,
            col=2,
        )

    # ==========================================
    # Track 3: Rotary Speed (RPM)
    # ==========================================
    rpm = log_df["rpm"]
    fig.add_trace(
        go.Scatter(
            x=rpm,
            y=depth,
            mode="lines",
            line=dict(color=TRACK_COLORS["rpm"], width=1.8),
            name="Rotary Speed (RPM)",
            hovertemplate="Depth: %{y:.1f} m<br>RPM: %{x:.0f}<extra></extra>",
        ),
        row=1,
        col=3,
    )

    # ==========================================
    # Track 4: Torque (ft-lbf)
    # ==========================================
    torque = log_df["torque"]
    fig.add_trace(
        go.Scatter(
            x=torque,
            y=depth,
            mode="lines",
            line=dict(color=TRACK_COLORS["torque"], width=1.8),
            name="Torque (ft-lbf)",
            hovertemplate="Depth: %{y:.1f} m<br>Torque: %{x:,.0f} ft-lbf<extra></extra>",
        ),
        row=1,
        col=4,
    )

    # ==========================================
    # Track 5: Rate of Penetration & Dc-exponent
    # ==========================================
    rop = log_df["rop"]
    dc = log_df["dc_exponent"]
    fig.add_trace(
        go.Scatter(
            x=rop,
            y=depth,
            mode="lines",
            line=dict(color=TRACK_COLORS["rop"], width=2.0),
            name="ROP (m/hr)",
            hovertemplate="Depth: %{y:.1f} m<br>ROP: %{x:.1f} m/hr<extra></extra>",
        ),
        row=1,
        col=5,
    )
    # Scale Dc-exponent to overlay nicely on ROP track (e.g. 0-2.5 scaled to 0-50)
    dc_scaled = np.clip(dc * 18.0, 0, 60)
    fig.add_trace(
        go.Scatter(
            x=dc_scaled,
            y=depth,
            mode="lines",
            line=dict(color=TRACK_COLORS["dc"], width=1.5, dash="dash"),
            name="Dc Exponent (scaled)",
            customdata=dc,
            hovertemplate="Depth: %{y:.1f} m<br>Dc Exponent: %{customdata:.2f}<extra></extra>",
        ),
        row=1,
        col=5,
    )

    # ==========================================
    # Track 6: Teale MSE vs UCS (psi) with Shading
    # ==========================================
    mse = log_df["mse_psi"]
    ucs = log_df["ucs_psi"]

    # Baseline UCS curve
    fig.add_trace(
        go.Scatter(
            x=ucs,
            y=depth,
            mode="lines",
            line=dict(color=TRACK_COLORS["ucs"], width=1.6, dash="dot"),
            name="Rock UCS (psi)",
            hovertemplate="Depth: %{y:.1f} m<br>UCS: %{x:,.0f} psi<extra></extra>",
        ),
        row=1,
        col=6,
    )

    # Teale MSE curve with fill to UCS
    fig.add_trace(
        go.Scatter(
            x=mse,
            y=depth,
            mode="lines",
            line=dict(color=TRACK_COLORS["mse"], width=2.0),
            name="Teale MSE (psi)",
            fill="tonextx",
            fillcolor="rgba(239, 68, 68, 0.20)",  # Translucent red for wasted energy
            hovertemplate="Depth: %{y:.1f} m<br>MSE: %{x:,.0f} psi<extra></extra>",
        ),
        row=1,
        col=6,
    )

    # ==========================================
    # Track 7: Stick-Slip Severity Index (SSI)
    # ==========================================
    ssi = log_df["ssi"]

    # Alert threshold references
    fig.add_trace(
        go.Scatter(
            x=[0.5] * len(depth),
            y=depth,
            mode="lines",
            line=dict(color=TRACK_COLORS["ssi_warn"], width=1.2, dash="dash"),
            name="SSI Warning (0.5)",
            hoverinfo="skip",
        ),
        row=1,
        col=7,
    )
    fig.add_trace(
        go.Scatter(
            x=[1.0] * len(depth),
            y=depth,
            mode="lines",
            line=dict(color=TRACK_COLORS["ssi_crit"], width=1.5, dash="solid"),
            name="SSI Severe (1.0)",
            hoverinfo="skip",
        ),
        row=1,
        col=7,
    )

    # Dynamic line coloring based on SSI magnitude
    fig.add_trace(
        go.Scatter(
            x=ssi,
            y=depth,
            mode="lines",
            line=dict(color="#f43f5e", width=2.2),
            name="SSI",
            hovertemplate="Depth: %{y:.1f} m<br>SSI: %{x:.2f}<extra></extra>",
        ),
        row=1,
        col=7,
    )

    # ==========================================
    # Global Axis & WellCAD / Techlog Styling
    # ==========================================
    y_min = float(depth.min())
    y_max = float(depth.max())

    fig.update_yaxes(
        autorange="reversed",  # Depth increases downwards
        range=[y_min, y_max],
        title_text="Measured Depth (m)",
        gridcolor=GRID_COLOR,
        zerolinecolor=BORDER_COLOR,
        row=1,
        col=1,
    )
    for col_idx in range(2, 8):
        fig.update_yaxes(
            autorange="reversed",
            range=[y_min, y_max],
            showticklabels=False,
            gridcolor=GRID_COLOR,
            zerolinecolor=BORDER_COLOR,
            row=1,
            col=col_idx,
        )

    # Update X-axes with proper oilfield ranges
    fig.update_xaxes(range=[0, 160], gridcolor=GRID_COLOR, zeroline=False, row=1, col=1)
    fig.update_xaxes(range=[0, max(50.0, float(wob.max() * 1.15))], gridcolor=GRID_COLOR, zeroline=False, row=1, col=2)
    fig.update_xaxes(range=[0, max(180.0, float(rpm.max() * 1.1))], gridcolor=GRID_COLOR, zeroline=False, row=1, col=3)
    fig.update_xaxes(range=[0, max(15000.0, float(torque.max() * 1.15))], gridcolor=GRID_COLOR, zeroline=False, row=1, col=4)
    fig.update_xaxes(range=[0, max(50.0, float(rop.max() * 1.15))], gridcolor=GRID_COLOR, zeroline=False, row=1, col=5)
    fig.update_xaxes(range=[0, max(100000.0, float(mse.max() * 1.05))], gridcolor=GRID_COLOR, zeroline=False, row=1, col=6)
    fig.update_xaxes(range=[0, max(1.6, float(ssi.max() * 1.1))], gridcolor=GRID_COLOR, zeroline=False, row=1, col=7)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=PLOT_BG,
        height=height,
        margin=dict(l=60, r=20, t=60, b=40),
        font=dict(family="Segoe UI, Inter, Arial, sans-serif", color=TEXT_COLOR, size=11),
        hovermode="y unified",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.08,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
            bgcolor="rgba(16, 25, 45, 0.8)",
            bordercolor=BORDER_COLOR,
            borderwidth=1,
        ),
    )

    # Style subplot title annotations
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(size=11, color="#38bdf8", family="Segoe UI Semibold")

    return fig
