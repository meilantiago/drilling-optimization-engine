"""
Interactive Diagnostic Cross-Plots & Vibration Spectral Visualizations
=======================================================================
Provides specialized engineering diagnostics:
1. WOB vs ROP Cross-Plot with Founder Point detection curve, safe window, and founder limit.
2. Drillstring Torsional FFT / Welch Power Spectral Density (PSD) with natural frequency modes (f0, 3*f0, 5*f0).
3. MSE vs UCS Rock Destructive Efficiency Cross-Plot.
4. Axial Vibration & Bit Bounce Shock Event Timeline.

Author: Tiago Meilan
Project: DrillOpt-Engine
"""

from typing import Any, Dict, Optional, Sequence, Union
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.optimizer.founder_point import FounderPointResult

BG_COLOR = "#0b111e"
PLOT_BG = "#10192d"
GRID_COLOR = "#1f2d48"
TEXT_COLOR = "#e2e8f0"
BORDER_COLOR = "#2a3b5c"


def create_founder_point_plot(
    wob: Union[Sequence[float], np.ndarray, pd.Series],
    rop: Union[Sequence[float], np.ndarray, pd.Series],
    result: Optional[Union[FounderPointResult, Dict[str, Any]]] = None,
    mse: Optional[Union[Sequence[float], np.ndarray, pd.Series]] = None,
    title: str = "Drill-Off Curve & Founder Point Detection (SPE 92194)",
) -> go.Figure:
    """
    Constructs an interactive WOB vs ROP cross-plot illustrating the
    linear cutting regime, founder inflection point, and recommended safe operating window.

    Parameters
    ----------
    wob : Sequence[float]
        Weight on Bit array in klbf.
    rop : Sequence[float]
        Rate of Penetration array in m/hr.
    result : Optional[FounderPointResult]
        Outcome of FounderPointAnalyzer.
    mse : Optional[Sequence[float]]
        MSE values to color telemetry points.
    title : str
        Chart header title.

    Returns
    -------
    go.Figure
        Plotly Figure with interactive data points, fitted model curve, and safe window.
    """
    w_arr = np.asarray(wob, dtype=float)
    r_arr = np.asarray(rop, dtype=float)

    mask = np.isfinite(w_arr) & np.isfinite(r_arr) & (w_arr > 0) & (r_arr > 0)
    w_clean = w_arr[mask]
    r_clean = r_arr[mask]

    fig = go.Figure()

    # 1. Telemetry Scatter Points
    marker_kwargs: Dict[str, Any] = dict(
        size=7,
        opacity=0.65,
        line=dict(width=0.5, color="#ffffff"),
    )

    if mse is not None:
        mse_clean = np.asarray(mse, dtype=float)[mask]
        marker_kwargs.update(
            color=mse_clean,
            colorscale="Viridis",
            colorbar=dict(
                title=dict(text="MSE (psi)", font=dict(color=TEXT_COLOR, size=10)),
                thickness=12,
                tickfont=dict(color=TEXT_COLOR, size=9),
            ),
            showscale=True,
        )
    else:
        marker_kwargs.update(color="#38bdf8")

    fig.add_trace(
        go.Scatter(
            x=w_clean,
            y=r_clean,
            mode="markers",
            marker=marker_kwargs,
            name="Drilling Telemetry",
            hovertemplate="WOB: %{x:.1f} klbf<br>ROP: %{y:.1f} m/hr<extra></extra>",
        )
    )

    # 2. Extract Founder Diagnostics
    w_founder = None
    w_rec = None
    r_founder = None
    poly_coeffs = None

    if result is not None:
        if isinstance(result, FounderPointResult):
            w_founder = result.wob_founder
            w_rec = result.wob_recommended
            r_founder = result.rop_at_founder
            if "poly_coeffs" in result.details:
                poly_coeffs = result.details["poly_coeffs"]
        elif isinstance(result, dict):
            w_founder = result.get("wob_founder")
            w_rec = result.get("wob_recommended")
            r_founder = result.get("rop_at_founder")
            poly_coeffs = result.get("details", {}).get("poly_coeffs")

    # 3. Fitted Curvature or Trend Curve
    if len(w_clean) > 5:
        w_span = np.linspace(float(w_clean.min()), float(w_clean.max()), 100)
        if poly_coeffs is not None and len(poly_coeffs) == 3:
            r_curve = np.polyval(poly_coeffs, w_span)
        else:
            p_fit = np.polyfit(w_clean, r_clean, deg=2)
            r_curve = np.polyval(p_fit, w_span)

        fig.add_trace(
            go.Scatter(
                x=w_span,
                y=np.maximum(0.1, r_curve),
                mode="lines",
                line=dict(color="#f59e0b", width=2.8),
                name="Fitted Mechanical Model",
                hoverinfo="skip",
            )
        )

    # 4. Safe Operating Window & Floundering Shading
    if w_founder is not None and not np.isnan(w_founder):
        rec_val = w_rec if (w_rec is not None and not np.isnan(w_rec)) else w_founder * 0.90
        max_x = max(float(w_clean.max() * 1.05), w_founder * 1.25) if len(w_clean) > 0 else w_founder * 1.3
        y_max = max(float(r_clean.max() * 1.15), 40.0) if len(r_clean) > 0 else 40.0

        # Safe Window Shading (Green)
        fig.add_vrect(
            x0=0,
            x1=rec_val,
            fillcolor="rgba(16, 185, 129, 0.12)",
            layer="below",
            line_width=0,
            annotation_text="SAFE DRILLING WINDOW",
            annotation_position="top left",
            annotation_font=dict(color="#10b981", size=10, family="Segoe UI Semibold"),
        )

        # Flounder / Bit Balling Risk Zone (Red)
        fig.add_vrect(
            x0=w_founder,
            x1=max_x,
            fillcolor="rgba(239, 68, 68, 0.12)",
            layer="below",
            line_width=0,
            annotation_text="FLOUNDERING & BIT BALLING ZONE",
            annotation_position="top right",
            annotation_font=dict(color="#ef4444", size=10, family="Segoe UI Semibold"),
        )

        # Vertical line at Founder Point
        fig.add_vline(
            x=w_founder,
            line=dict(color="#ef4444", width=2.0, dash="dash"),
            annotation_text=f"Founder WOB = {w_founder:.1f} klbf",
            annotation_position="bottom right",
            annotation_font=dict(color="#ef4444", size=11, family="Segoe UI Semibold"),
        )

        # Vertical line at Recommended Setpoint
        fig.add_vline(
            x=rec_val,
            line=dict(color="#10b981", width=2.0, dash="dot"),
            annotation_text=f"Recommended WOB = {rec_val:.1f} klbf (90%)",
            annotation_position="bottom left",
            annotation_font=dict(color="#10b981", size=11, family="Segoe UI Semibold"),
        )

        # Star marker at detected inflection point
        if r_founder is not None and not np.isnan(r_founder):
            fig.add_trace(
                go.Scatter(
                    x=[w_founder],
                    y=[r_founder],
                    mode="markers+text",
                    marker=dict(symbol="star", size=15, color="#fbbf24", line=dict(color="#000000", width=1.5)),
                    text=["FOUNDER POINT"],
                    textposition="top center",
                    textfont=dict(color="#fbbf24", size=11, family="Segoe UI Semibold"),
                    name="Founder Point",
                    hovertemplate=f"Founder Point<br>WOB: {w_founder:.1f} klbf<br>ROP: {r_founder:.1f} m/hr<extra></extra>",
                )
            )

    fig.update_layout(
        title=dict(text=title, font=dict(color="#38bdf8", size=14, family="Segoe UI Semibold")),
        template="plotly_dark",
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=PLOT_BG,
        height=480,
        margin=dict(l=55, r=25, t=55, b=45),
        xaxis=dict(
            title="Weight on Bit (WOB, klbf)",
            gridcolor=GRID_COLOR,
            zerolinecolor=BORDER_COLOR,
            tickfont=dict(color=TEXT_COLOR),
        ),
        yaxis=dict(
            title="Rate of Penetration (ROP, m/hr)",
            gridcolor=GRID_COLOR,
            zerolinecolor=BORDER_COLOR,
            tickfont=dict(color=TEXT_COLOR),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=10, color=TEXT_COLOR),
            bgcolor="rgba(16, 25, 45, 0.7)",
        ),
    )

    return fig


def create_vibration_spectrum_plot(
    freqs: Sequence[float],
    psd: Sequence[float],
    f0: float = 0.25,
    dominant_frequency: Optional[float] = None,
    harmonics_count: int = 3,
    title: str = "Torsional Vibration Power Spectral Density & Drillstring Resonance Modes",
) -> go.Figure:
    """
    Constructs an interactive Plotly chart of Power Spectral Density (PSD) vs Frequency (Hz),
    marking theoretical drillstring torsional resonance modes (f0, 3*f0, 5*f0).

    Parameters
    ----------
    freqs : Sequence[float]
        Frequencies array in Hz.
    psd : Sequence[float]
        Power Spectral Density values (ft-lbf^2/Hz or amplitude).
    f0 : float
        Drillstring fundamental natural resonance frequency (f0 = c_s / 4L).
    dominant_frequency : Optional[float]
        Detected dominant peak frequency.
    harmonics_count : int
        Number of odd harmonics to display.
    title : str
        Chart header title.

    Returns
    -------
    go.Figure
        Interactive Plotly spectral density figure.
    """
    freq_arr = np.asarray(freqs, dtype=float)
    psd_arr = np.asarray(psd, dtype=float)

    # Filter out zero or negative frequencies
    valid = freq_arr > 0.01
    f_plot = freq_arr[valid]
    p_plot = psd_arr[valid]

    fig = go.Figure()

    # 1. Continuous PSD Curve
    fig.add_trace(
        go.Scatter(
            x=f_plot,
            y=p_plot,
            mode="lines",
            line=dict(color="#38bdf8", width=2.0),
            fill="tozeroy",
            fillcolor="rgba(56, 189, 248, 0.15)",
            name="Torque Spectral Energy",
            hovertemplate="Frequency: %{x:.3f} Hz<br>Power Density: %{y:.2f}<extra></extra>",
        )
    )

    # 2. Resonant Harmonic Markers: f_n = (2n - 1) * f0 for n=1, 2, 3
    harmonic_labels = [
        ("Fundamental Mode (f₀)", f0, "#ef4444"),
        ("3rd Harmonic Mode (3f₀)", 3.0 * f0, "#f59e0b"),
        ("5th Harmonic Mode (5f₀)", 5.0 * f0, "#a855f7"),
    ]

    for label, h_freq, h_color in harmonic_labels[:harmonics_count]:
        if (len(f_plot) == 0) or (h_freq <= float(np.max(f_plot)) * 1.1):
            # Resonant mode vertical line
            fig.add_vline(
                x=h_freq,
                line=dict(color=h_color, width=1.8, dash="dash"),
                annotation_text=f"{label}: {h_freq:.2f} Hz",
                annotation_position="top right",
                annotation_font=dict(color=h_color, size=10, family="Segoe UI Semibold"),
            )
            # Resonance sensitivity band (+/- 10%)
            fig.add_vrect(
                x0=h_freq * 0.90,
                x1=h_freq * 1.10,
                fillcolor=h_color,
                opacity=0.08,
                layer="below",
                line_width=0,
            )

    # 3. Peak Detection Highlight
    peak_f = dominant_frequency
    if peak_f is None and len(p_plot) > 0:
        max_idx = np.argmax(p_plot)
        peak_f = float(f_plot[max_idx])
        peak_p = float(p_plot[max_idx])
    elif peak_f is not None and len(f_plot) > 0:
        peak_p = float(np.interp(peak_f, f_plot, p_plot))
    else:
        peak_p = 0.0

    if peak_f is not None and peak_f > 0:
        fig.add_trace(
            go.Scatter(
                x=[peak_f],
                y=[peak_p],
                mode="markers+text",
                marker=dict(size=10, color="#fbbf24", symbol="diamond"),
                text=[f"Observed Peak: {peak_f:.2f} Hz"],
                textposition="top center",
                textfont=dict(color="#fbbf24", size=10, family="Segoe UI Semibold"),
                name="Dominant Peak",
                hovertemplate=f"Peak Frequency: {peak_f:.3f} Hz<br>Power: {peak_p:.2f}<extra></extra>",
            )
        )

    max_freq_display = min(float(np.max(f_plot)), max(2.5, 6.0 * f0)) if len(f_plot) > 0 else 2.5

    fig.update_layout(
        title=dict(text=title, font=dict(color="#38bdf8", size=14, family="Segoe UI Semibold")),
        template="plotly_dark",
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=PLOT_BG,
        height=480,
        margin=dict(l=55, r=25, t=55, b=45),
        xaxis=dict(
            title="Torsional Oscillation Frequency (Hz)",
            range=[0.0, max_freq_display],
            gridcolor=GRID_COLOR,
            zerolinecolor=BORDER_COLOR,
            tickfont=dict(color=TEXT_COLOR),
        ),
        yaxis=dict(
            title="Power Spectral Density (PSD)",
            gridcolor=GRID_COLOR,
            zerolinecolor=BORDER_COLOR,
            tickfont=dict(color=TEXT_COLOR),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=10, color=TEXT_COLOR),
            bgcolor="rgba(16, 25, 45, 0.7)",
        ),
    )

    return fig


def create_mse_ucs_crossplot(
    ucs: Union[Sequence[float], np.ndarray, pd.Series],
    mse: Union[Sequence[float], np.ndarray, pd.Series],
    formation: Optional[Sequence[str]] = None,
    title: str = "Teale MSE vs Rock UCS Efficiency Envelope",
) -> go.Figure:
    """
    Cross-plots Teale MSE vs Rock UCS with theoretical efficiency reference lines
    (Dupriest 35% efficiency line, 10% line, and extreme dysfunction >10x UCS).
    """
    u_arr = np.asarray(ucs, dtype=float)
    m_arr = np.asarray(mse, dtype=float)

    mask = np.isfinite(u_arr) & np.isfinite(m_arr) & (u_arr > 0) & (m_arr > 0)
    u_clean = u_arr[mask]
    m_clean = m_arr[mask]

    fig = go.Figure()

    # Data scatter
    fig.add_trace(
        go.Scatter(
            x=u_clean,
            y=m_clean,
            mode="markers",
            marker=dict(size=6, color="#06b6d4", opacity=0.6, line=dict(width=0.5, color="#ffffff")),
            name="Telemetry Points",
            hovertemplate="UCS: %{x:,.0f} psi<br>MSE: %{y:,.0f} psi<extra></extra>",
        )
    )

    # Reference lines
    u_max = max(35000.0, float(u_clean.max() * 1.1)) if len(u_clean) > 0 else 35000.0
    u_ref = np.linspace(5000, u_max, 50)

    # 100% efficiency: MSE = UCS
    fig.add_trace(
        go.Scatter(
            x=u_ref,
            y=u_ref,
            mode="lines",
            line=dict(color="#10b981", width=1.5, dash="dot"),
            name="100% Theoretical Limit (MSE = UCS)",
        )
    )
    # 35% efficiency (Dupriest PDC benchmark: MSE = UCS / 0.35 ~ 2.85 * UCS)
    fig.add_trace(
        go.Scatter(
            x=u_ref,
            y=u_ref / 0.35,
            mode="lines",
            line=dict(color="#38bdf8", width=2.0),
            name="Optimal PDC Baseline (EFF = 35%)",
        )
    )
    # 10% efficiency (MSE = 10 * UCS, Severe dysfunction)
    fig.add_trace(
        go.Scatter(
            x=u_ref,
            y=u_ref / 0.10,
            mode="lines",
            line=dict(color="#ef4444", width=1.8, dash="dash"),
            name="Dysfunction Threshold (EFF = 10%)",
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(color="#38bdf8", size=14, family="Segoe UI Semibold")),
        template="plotly_dark",
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=PLOT_BG,
        height=450,
        margin=dict(l=55, r=25, t=55, b=45),
        xaxis=dict(title="Rock Unconfined Compressive Strength (UCS, psi)", gridcolor=GRID_COLOR),
        yaxis=dict(title="Teale Mechanical Specific Energy (MSE, psi)", gridcolor=GRID_COLOR),
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center", font=dict(size=10)),
    )

    return fig
