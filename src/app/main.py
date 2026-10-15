"""
DrillOpt-Engine: Real-Time Drilling Mechanics, MSE & Dysfunction Detection Console
=================================================================================
Interactive Streamlit Application modeled after modern Real-Time Operations Centers
(RTOC) and drilling cabin displays.

Features:
- Subsurface Multi-Track Well Log (Techlog / WellCAD style) with synchronized tracks
- Automated Founder Point detection and safe operating envelope estimation
- Torsional FFT spectral analysis with drillstring harmonic resonance modes (f0, 3*f0, 5*f0)
- Axial shock, bit bounce, and PDC cutter wear progression diagnostics
- Interactive What-If parameter optimization with direct USD rig cost savings
- ExxonMobil SPE 92194 Dysfunction Symptoms Matrix & physics reference

Author: Tiago Meilan
Project: DrillOpt-Engine
"""

import os
import time
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import streamlit as st

# Application & physics imports
from src.data.loader import DrillingDataLoader
from src.data.generator import generate_benchmark_files
from src.data.rig_state import RigStateClassifier, RigState
from src.app.components.live_stream import (
    create_live_oscilloscope_plot,
    evaluate_realtime_alarm,
)
from src.physics.mse import teale_mse, dupriest_mse, mechanical_efficiency
from src.dynamics.stick_slip import (
    analyze_torsional_welch,
    drillstring_torsional_natural_frequency,
    compute_rolling_stick_slip_index,
)
from src.dynamics.bit_bounce import (
    detect_axial_shocks,
    compute_axial_vibration_index,
    classify_bit_bounce_severity,
)
from src.dynamics.wear_model import (
    compute_cumulative_dissipated_energy,
    evaluate_bit_wear_state,
)
from src.optimizer.founder_point import FounderPointAnalyzer
from src.optimizer.road_map import RoadMapOptimizer
from src.app.components.log_plot import create_multitrack_log_plot
from src.app.components.cross_plots import (
    create_founder_point_plot,
    create_vibration_spectrum_plot,
    create_mse_ucs_crossplot,
)
from src.app.components.simulator import render_simulator_panel


# =============================================================================
# Streamlit Page Configuration & Dark Oilfield Styling
# =============================================================================
st.set_page_config(
    page_title="DrillOpt-Engine | RTOC Drilling Console",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* Dark oilfield aesthetic */
    .stApp {
        background-color: #0b111e;
        color: #e2e8f0;
    }
    .metric-card {
        background-color: #10192d;
        border: 1px solid #1f2d48;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #64748b;
    }
    .badge-safe {
        background-color: #064e3b;
        color: #34d399;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-warn {
        background-color: #78350f;
        color: #fbbf24;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-crit {
        background-color: #7f1d1d;
        color: #f87171;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    /* Tab headers */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #10192d;
        border: 1px solid #1f2d48;
        border-radius: 6px 6px 0px 0px;
        color: #94a3b8;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
        border-bottom: 2px solid #38bdf8 !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# =============================================================================
# Data Loading & Initialization Helper Functions
# =============================================================================
@st.cache_data(show_spinner="Loading drilling dataset...")
def load_default_benchmark(seed: int = 42) -> pd.DataFrame:
    """Loads or generates benchmark drilling telemetry with explicit seed key."""
    base_dir = PROJECT_ROOT / "data" / "processed"
    parquet_path = base_dir / "benchmark_drilling.parquet"
    las_path = base_dir / "benchmark_drilling.las"

    if parquet_path.exists():
        return DrillingDataLoader.load_file(parquet_path)
    elif las_path.exists():
        return DrillingDataLoader.load_file(las_path)
    else:
        # Generate new benchmark files if not present
        pq, _ = generate_benchmark_files(output_dir=str(base_dir), random_seed=seed)
        return DrillingDataLoader.load_file(pq)


def get_dataset(choice: str, uploaded_file) -> pd.DataFrame:
    """Retrieves dataframe based on user sidebar selection."""
    curr_seed = st.session_state.get("benchmark_seed", 42)
    if choice == "Benchmark Dataset":
        return load_default_benchmark(seed=curr_seed)
    elif choice == "Upload Custom Log" and uploaded_file is not None:
        file_ext = Path(uploaded_file.name).suffix.lower()
        # Save temp file for lasio / loaders
        temp_dir = PROJECT_ROOT / "data" / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / uploaded_file.name
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return DrillingDataLoader.load_file(temp_path)
    else:
        return load_default_benchmark(seed=curr_seed)


# =============================================================================
# Sidebar: Controls & Well Configuration
# =============================================================================
with st.sidebar:
    st.markdown("## ⚙️ DrillOpt-Engine")
    st.markdown("*Autonomous Drilling Mechanics & MSE Optimization*")
    st.markdown("---")

    # 1. Surveillance Mode Selector
    st.markdown("### 📡 Surveillance Mode")
    surveillance_mode = st.radio(
        "Operating Mode:",
        ["🔴 Live RTOC Streaming Simulation", "📊 Complete Section Review"],
        index=0,
        help="Switch between second-by-second live drilling playback and static full-well review.",
    )

    st.markdown("---")

    # 2. Dataset Selection
    st.markdown("### 📁 Telemetry Source")
    dataset_option = st.radio(
        "Select Telemetry Ingestion:",
        ["Benchmark Dataset", "Upload Custom Log"],
        help="Benchmark includes stratified formations, founder tests, and vibration dynamics.",
    )

    uploaded_file = None
    if dataset_option == "Upload Custom Log":
        uploaded_file = st.file_uploader(
            "Upload CWLS LAS, Parquet, or CSV",
            type=["las", "parquet", "pq", "csv"],
        )

    if st.button("🔄 Regenerate Benchmark Run", use_container_width=True):
        with st.spinner("Generating physical drilling simulation with fresh stratigraphy..."):
            new_seed = int(np.random.randint(1000, 999999))
            generate_benchmark_files(
                output_dir=str(PROJECT_ROOT / "data" / "processed"),
                random_seed=new_seed,
                randomize_formations=True,
            )
            st.cache_data.clear()
            st.session_state["benchmark_seed"] = new_seed
            st.session_state["stream_idx"] = 120
            st.session_state["is_streaming"] = False
            st.success(f"New wellbore profile generated! (Seed #{new_seed})")
            st.rerun()

    st.markdown("---")

    # 3. Wellbore Configuration Inputs
    st.markdown("### 🛠️ Rig & Wellbore Setup")
    bit_diam_in = st.number_input(
        "Bit Outer Diameter (in)",
        min_value=3.75,
        max_value=26.0,
        value=8.50,
        step=0.25,
        help="Standard hole section outer diameter (typically 8.5 in or 12.25 in).",
    )
    mud_wt_ppg = st.number_input(
        "Drilling Mud Density (ppg)",
        min_value=8.33,
        max_value=19.0,
        value=10.20,
        step=0.10,
        help="Used for hydraulic pressure drops and Rehm-McClendon Dc-exponent.",
    )
    rig_day_rate = st.number_input(
        "Rig Spread Day Rate ($/day)",
        min_value=10000.0,
        max_value=500000.0,
        value=50000.0,
        step=5000.0,
        format="%.0f",
        help="Total daily rig and personnel operating cost.",
    )

    st.markdown("---")
    st.markdown("### ⏱️ Trajectory Filter")


# =============================================================================
# Load Telemetry & Compute Primary Indices
# =============================================================================
df = get_dataset(dataset_option, uploaded_file)

# Classify rig state
classifier = RigStateClassifier()
df = classifier.classify_dataframe(df)

# Calculate MSE if missing
if "mse_psi" not in df.columns:
    df["mse_psi"] = teale_mse(
        wob=df["wob"] * 1000.0,
        torque=df["torque"],
        rpm=df["rpm"],
        rop=df["rop"] / 0.3048,
        bit_diameter=bit_diam_in,
        use_field_units=True,
    )

# Calculate SSI if missing
if "ssi" not in df.columns:
    df["ssi"] = compute_rolling_stick_slip_index(df["rpm"], window_samples=25).fillna(0.2)

d_min_total = float(df["depth"].min())
d_max_total = float(df["depth"].max())

# Initialize streaming session state
if "stream_idx" not in st.session_state:
    st.session_state["stream_idx"] = 120
if "is_streaming" not in st.session_state:
    st.session_state["is_streaming"] = False

# =============================================================================
# Surveillance Mode: Real-Time Stream vs Complete Log Review
# =============================================================================
if surveillance_mode == "🔴 Live RTOC Streaming Simulation":
    # Bound stream_idx within dataframe
    cur_idx = int(np.clip(st.session_state.get("stream_idx", 120), 10, len(df) - 1))
    st.session_state["stream_idx"] = cur_idx
    cur_row = df.iloc[cur_idx]
    cur_depth_val = float(cur_row["depth"])

    # Sidebar bit position scrubber
    with st.sidebar:
        st.markdown("### 🎛️ Live Bit Positioning")
        scrub_depth = st.slider(
            "Current Bit Depth (m)",
            min_value=d_min_total,
            max_value=d_max_total,
            value=cur_depth_val,
            step=1.0,
            help="Drag to manually scrub the bit position anywhere along the wellbore timeline.",
        )
        if abs(scrub_depth - cur_depth_val) > 0.4:
            st.session_state["stream_idx"] = int(np.argmin(np.abs(df["depth"].values - scrub_depth)))
            cur_idx = st.session_state["stream_idx"]
            cur_row = df.iloc[cur_idx]
            cur_depth_val = float(cur_row["depth"])

        stream_speed = st.select_slider(
            "Streaming Speed:",
            options=["Realistic (0.30s)", "Normal (0.12s)", "High Speed (0.04s)"],
            value="Normal (0.12s)",
        )

    # Active dataset is the section drilled so far down to the bit
    df_filtered = df.iloc[: cur_idx + 1].copy()
    depth_window = (d_min_total, cur_depth_val)

    # Instantaneous Metrics
    cur_rop = float(cur_row["rop"])
    cur_wob = float(cur_row["wob"])
    cur_rpm = float(cur_row["rpm"])
    cur_mse = float(cur_row["mse_psi"])
    cur_ssi = float(cur_row["ssi"])
    cur_rig_state = str(cur_row["rig_state"])
    cur_formation = str(cur_row.get("formation", "Vaca Muerta Shale"))

    # Active drilling slice
    active_drilling_slice = df_filtered[df_filtered["rig_state"] == RigState.DRILLING_ROTARY.value]
    if active_drilling_slice.empty:
        active_drilling_slice = df_filtered

else:
    # Complete Section Review
    with st.sidebar:
        st.markdown("### ⏱️ Trajectory Interval Filter")
        depth_window = st.slider(
            "Measured Depth Interval (m)",
            min_value=d_min_total,
            max_value=d_max_total,
            value=(d_min_total, d_max_total),
            step=5.0,
        )

    df_filtered = df[(df["depth"] >= depth_window[0]) & (df["depth"] <= depth_window[1])].copy()
    if df_filtered.empty:
        df_filtered = df.copy()

    active_drilling_slice = df_filtered[df_filtered["rig_state"] == RigState.DRILLING_ROTARY.value]
    if active_drilling_slice.empty:
        active_drilling_slice = df_filtered

    cur_rop = float(active_drilling_slice["rop"].mean())
    cur_wob = float(active_drilling_slice["wob"].mean())
    cur_rpm = float(active_drilling_slice["rpm"].mean())
    cur_mse = float(active_drilling_slice["mse_psi"].mean())
    cur_ssi = float(active_drilling_slice["ssi"].iloc[-1]) if len(active_drilling_slice) > 0 else 0.25
    cur_rig_state = str(df_filtered["rig_state"].iloc[-1]) if len(df_filtered) > 0 else "UNKNOWN"
    cur_formation = str(df_filtered.get("formation", pd.Series(["Vaca Muerta"])).iloc[-1])

# =============================================================================
# Executive Header & KPI Cards
# =============================================================================
st.markdown("## 📡 Real-Time Drilling Operations Console")
if surveillance_mode == "🔴 Live RTOC Streaming Simulation":
    st.markdown(
        f"*🔴 **LIVE MWD TELEMETRY STREAMING** — Bit Depth: **{cur_depth_val:.1f} m** | Active Formation: **{cur_formation}** | Total Section Target: **{d_max_total:.1f} m***"
    )
else:
    st.markdown(
        f"*Historical Review — Measured Depth Interval **{depth_window[0]:.1f} m** to **{depth_window[1]:.1f} m** "
        f"({depth_window[1] - depth_window[0]:.1f} m span)*"
    )

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    depth_lbl = "Bit Measured Depth" if surveillance_mode == "🔴 Live RTOC Streaming Simulation" else "Measured Depth"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{depth_lbl}</div>
            <div class="metric-value">{float(df_filtered['depth'].max()):.1f} <span style="font-size:1rem;color:#94a3b8">m</span></div>
            <div class="metric-sub">Progress: {((float(df_filtered['depth'].max()) - d_min_total) / max(1.0, d_max_total - d_min_total) * 100):.1f}% of section</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    rop_lbl = "Instant ROP" if surveillance_mode == "🔴 Live RTOC Streaming Simulation" else "Avg ROP"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{rop_lbl}</div>
            <div class="metric-value">{cur_rop:.1f} <span style="font-size:1rem;color:#94a3b8">m/hr</span></div>
            <div class="metric-sub">{(cur_rop / 0.3048):.1f} ft/hr</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    mse_lbl = "Instant Teale MSE" if surveillance_mode == "🔴 Live RTOC Streaming Simulation" else "Avg Teale MSE"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{mse_lbl}</div>
            <div class="metric-value">{cur_mse:,.0f} <span style="font-size:1rem;color:#94a3b8">psi</span></div>
            <div class="metric-sub">{(cur_mse * 0.00689476):.1f} MPa</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    if cur_ssi < 0.5:
        badge_html = f'<span class="badge-safe">SAFE ({cur_ssi:.2f})</span>'
    elif cur_ssi < 1.0:
        badge_html = f'<span class="badge-warn">MODERATE ({cur_ssi:.2f})</span>'
    else:
        badge_html = f'<span class="badge-crit">SEVERE ({cur_ssi:.2f})</span>'

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Stick-Slip Index</div>
            <div class="metric-value" style="font-size:1.5rem">{badge_html}</div>
            <div class="metric-sub">Threshold: &lt; 0.50 SPE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col5:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Rig State & Formation</div>
            <div class="metric-value" style="font-size:1.15rem;color:#38bdf8">{cur_formation}</div>
            <div class="metric-sub">{cur_rig_state.replace('_', ' ')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =============================================================================
# Live Streaming Controls & Real-Time Oscilloscope Panel
# =============================================================================
if surveillance_mode == "🔴 Live RTOC Streaming Simulation":
    alarm_info = evaluate_realtime_alarm(cur_row)

    # Real-Time Downhole Advisory Alert Banner
    st.markdown(
        f"""
        <div style="background-color:#10192d;border-left:5px solid {alarm_info['color']};padding:14px 18px;border-radius:6px;margin-top:10px;margin-bottom:14px;box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);">
            <div style="font-size:1.05rem;font-weight:700;color:{alarm_info['color']}">{alarm_info['title']}</div>
            <div style="font-size:0.92rem;color:#cbd5e1;margin-top:4px;line-height:1.4">{alarm_info['diagnostic']}</div>
            <div style="font-size:0.88rem;color:#38bdf8;font-weight:600;margin-top:6px">💡 {alarm_info['recommendation']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Transport Controls Bar
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1.5, 1.2, 1.2, 1.2])
    with ctrl_col1:
        is_playing = st.session_state.get("is_streaming", False)
        if is_playing:
            if st.button("⏸ Pause Stream", use_container_width=True):
                st.session_state["is_streaming"] = False
                st.rerun()
        else:
            if st.button("▶ Start Live Stream", type="primary", use_container_width=True):
                st.session_state["is_streaming"] = True
                st.rerun()

    with ctrl_col2:
        if st.button("⏩ Step (+10m)", use_container_width=True):
            st.session_state["stream_idx"] = min(len(df) - 1, cur_idx + 50)
            st.rerun()

    with ctrl_col3:
        if st.button("↺ Reset Bit (2000m)", use_container_width=True):
            st.session_state["stream_idx"] = 60
            st.session_state["is_streaming"] = False
            st.rerun()

    with ctrl_col4:
        st.markdown(
            f"<div style='padding-top:8px;font-size:0.85rem;color:#94a3b8'>Bit Status: <b style='color:#38bdf8'>{'STREAMING' if is_playing else 'STANDBY'}</b></div>",
            unsafe_allow_html=True,
        )

    # Rolling Oscilloscope Window
    window_samples = 60
    df_window = df.iloc[max(0, cur_idx - window_samples) : cur_idx + 1]

    st.markdown("#### 📈 Real-Time MWD Telemetry Oscilloscope (Active Drilling Window)")
    osci_fig = create_live_oscilloscope_plot(df_window)
    st.plotly_chart(osci_fig, use_container_width=True)

    # Auto-advance streaming loop if active
    if is_playing:
        delay = 0.04 if "High" in stream_speed else (0.12 if "Normal" in stream_speed else 0.28)
        # Advance 15 samples per frame
        advance_steps = 12
        if cur_idx + advance_steps >= len(df) - 1:
            st.session_state["stream_idx"] = len(df) - 1
            st.session_state["is_streaming"] = False
            st.toast("🏁 Total section depth reached (2500m)!", icon="🏁")
            st.rerun()
        else:
            st.session_state["stream_idx"] = cur_idx + advance_steps
            time.sleep(delay)
            st.rerun()

st.markdown("---")

# =============================================================================
# Navigation Tabs
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Subsurface Multi-Track Log View",
    "🔬 Dysfunction Diagnostics",
    "💡 Drilling Optimization & ROI",
    "📚 Theory & SPE Framework",
])


# =============================================================================
# Tab 1: Subsurface Multi-Track Log View (Techlog / WellCAD style)
# =============================================================================
with tab1:
    st.markdown("### Interactive WellCAD / Techlog Subsurface Well Log")
    st.markdown(
        "Synchronized 7-track continuous drilling log showing depth increasing downwards. "
        "Drag or zoom vertically to inspect local intervals across all tracks simultaneously."
    )

    log_fig = create_multitrack_log_plot(
        df_filtered,
        bit_diameter_in=bit_diam_in,
        mud_weight_ppg=mud_wt_ppg,
        height=950,
    )
    st.plotly_chart(log_fig, use_container_width=True)

    with st.expander("ℹ️ Multi-Track Channel Descriptions & Thresholds"):
        st.markdown(
            """
            - **Track 1 (GR)**: Gamma Ray in GAPI. Standard shale baseline at 75 API (sandstone < 75, shale >= 75).
            - **Track 2 (WOB)**: Weight on Bit (klbf). Red dotted line marks the dynamic founder point threshold.
            - **Track 3 (RPM)**: Drillstring rotary speed (RPM).
            - **Track 4 (Torque)**: Surface drillstring torque (ft-lbf).
            - **Track 5 (ROP & Dc)**: Rate of penetration (m/hr, cyan) and Rehm-McClendon pore pressure Dc-exponent (dashed yellow).
            - **Track 6 (MSE vs UCS)**: Teale Mechanical Specific Energy vs rock Unconfined Compressive Strength. Red shading indicates energy wasted in dysfunctions.
            - **Track 7 (SSI)**: Stick-Slip Severity Index. Green (< 0.50), Amber (0.50 - 1.00), Red (>= 1.00 stall/reverse rotation).
            """
        )


# =============================================================================
# Tab 2: Dysfunction Diagnostics
# =============================================================================
with tab2:
    st.markdown("### Drilling Dysfunction & Dynamics Diagnostics")
    st.markdown(
        "Multi-physics diagnostic suite detecting **Founder Point Saturation**, "
        "**Stick-Slip Torsional Resonance (FFT/Welch)**, **Axial Bit Bounce Shocks**, and **PDC Cutter Wear**."
    )

    col_diag_left, col_diag_right = st.columns(2)

    # 1. Founder Point Analysis
    with col_diag_left:
        st.markdown("#### 1. Automated Founder Point Cross-Plot")
        founder_analyzer = FounderPointAnalyzer(bit_diameter_in=bit_diam_in)
        founder_result = founder_analyzer.analyze(active_drilling_slice)

        founder_fig = create_founder_point_plot(
            wob=active_drilling_slice["wob"],
            rop=active_drilling_slice["rop"],
            result=founder_result,
            mse=active_drilling_slice["mse_psi"],
        )
        st.plotly_chart(founder_fig, use_container_width=True)

        if founder_result.regime == "founder_detected":
            st.success(
                f"**Founder Point Confirmed**: Inflection detected at **{founder_result.wob_founder:.1f} klbf**. "
                f"Recommended continuous operating setpoint is **{founder_result.wob_recommended:.1f} klbf** "
                f"(Confidence R² = {founder_result.confidence_score:.2f})."
            )
        else:
            st.info(
                f"**Regime Status**: `{founder_result.regime}`. "
                f"Operating WOB is currently below the formation founder limit."
            )

    # 2. Stick-Slip Spectral Decomposition
    with col_diag_right:
        st.markdown("#### 2. Drillstring Torsional FFT / Welch Spectrum")
        # Assume 1 Hz telemetry sampling rate for continuous drilling
        torque_signal = active_drilling_slice["torque"].dropna().values
        depth_for_res = float(active_drilling_slice["depth"].mean()) if not active_drilling_slice.empty else 2250.0

        f0, t0, harmonics = drillstring_torsional_natural_frequency(
            drillstring_length=depth_for_res,
            length_unit="m",
            harmonics_count=3,
        )

        if len(torque_signal) >= 16:
            spec_res = analyze_torsional_welch(torque_signal, sampling_rate=2.0)
            psd_fig = create_vibration_spectrum_plot(
                freqs=spec_res.frequencies,
                psd=spec_res.spectrum,
                f0=f0,
                dominant_frequency=spec_res.dominant_frequency,
            )
            st.plotly_chart(psd_fig, use_container_width=True)

            # Check resonance alignment
            freq_diff_pct = abs(spec_res.dominant_frequency - f0) / max(0.01, f0) * 100.0
            if freq_diff_pct < 15.0:
                st.error(
                    f"⚠️ **RESONANCE ALARM**: Observed peak ({spec_res.dominant_frequency:.2f} Hz) aligns within "
                    f"{freq_diff_pct:.1f}% of theoretical drillstring fundamental resonance f₀ ({f0:.2f} Hz). "
                    f"Immediate RPM adjustment required!"
                )
            else:
                st.info(
                    f"**Spectral Health**: Dominant frequency is {spec_res.dominant_frequency:.2f} Hz "
                    f"(Natural f₀ = {f0:.2f} Hz, Period T₀ = {t0:.2f} s)."
                )
        else:
            st.warning("Insufficient telemetry samples in selected interval for FFT analysis.")

    st.markdown("---")

    # 3. Axial Bit Bounce & 4. PDC Cutter Dull Progression
    col_sub_1, col_sub_2 = st.columns(2)

    with col_sub_1:
        st.markdown("#### 3. Axial Vibration & Bit Bounce Severity")
        wob_series = active_drilling_slice["wob"] * 1000.0
        shock_df, shock_summary = detect_axial_shocks(wob_series, window_samples=25)

        st.markdown(
            f"- **Axial Vibration Index (AVI)**: `{shock_summary.mean_vibration_index:.2f}` (Max: `{shock_summary.max_vibration_index:.2f}`)\n"
            f"- **Shock Spikes Detected**: `{shock_summary.shock_events_count}` events\n"
            f"- **Off-Bottom Detachment**: `{shock_summary.bounce_events_count}` events\n"
            f"- **Severity Category**: **{shock_summary.severity_category.upper()}**"
        )

        if shock_summary.severity_category in ["High", "Severe"]:
            st.error("⚠️ Severe axial bit bounce detected! Risk of PDC impact cutter chipping.")
        else:
            st.success("✅ Axial load stability is within safe cutter tolerance limits.")

    with col_sub_2:
        st.markdown("#### 4. PDC Cutter Dull Grading & Wear Progression")
        cum_dissipated = compute_cumulative_dissipated_energy(
            mse=active_drilling_slice["mse_psi"],
            ucs=active_drilling_slice.get("ucs_psi", 10000.0),
            depth=active_drilling_slice["depth"] / 0.3048,  # feet
            bit_diameter=bit_diam_in,
        )
        total_energy_ft_lbf = float(cum_dissipated.iloc[-1]) if len(cum_dissipated) > 0 else 0.0
        wear_state = evaluate_bit_wear_state(
            cumulative_energy_ft_lbf=total_energy_ft_lbf,
            drilled_depth_ft=(depth_window[1] - depth_window[0]) / 0.3048,
        )

        st.markdown(
            f"- **Cumulative Dissipated Energy**: `{wear_state.cumulative_energy_dissipated_mj:.1f} MJ`\n"
            f"- **Estimated IADC Dull Grade**: **{wear_state.iadc_wear_grade_integer} / 8** (Continuous: `{wear_state.wear_grade_continuous:.2f}`)\n"
            f"- **Bit Condition**: **{wear_state.wear_condition}**\n"
            f"- **Remaining Energy Capacity**: `{wear_state.remaining_energy_capacity_mj:.1f} MJ`"
        )


# =============================================================================
# Tab 3: Drilling Optimization & ROI
# =============================================================================
with tab3:
    render_simulator_panel(
        df=df_filtered,
        default_day_rate_usd=rig_day_rate,
        bit_diameter_in=bit_diam_in,
    )


# =============================================================================
# Tab 4: Theory & SPE Framework
# =============================================================================
with tab4:
    st.markdown("### Drilling Mechanics Foundations & SPE 92194 Framework")
    st.markdown(
        "DrillOpt-Engine implements the operational surveillance methodology established by "
        "**Teale (1965)** and **Dupriest & Koederitz (ExxonMobil, SPE/IADC 92194)**."
    )

    # 1. ExxonMobil Symptoms Matrix
    st.markdown("#### 📋 ExxonMobil Real-Time MSE Symptoms Matrix (SPE 92194)")
    st.markdown(
        """
| Dysfunction | MSE Response | ROP Response | Torque Response | Primary Physical Root Cause | Corrective Action |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Bit Founder** | Rapid Increase | Flattens / Drops with WOB | Stabilizes / Disproportionate | Cutting depth exceeds junk slot / cleaning capacity | Back off WOB to 90% of founder point |
| **Bit Balling** | Sharp Explosion | Abrupt Plunge | Drops or Erratic | Cuttings pack around cutters and blade junk slots | Pump sweeps, boost flow rate / TFA, lift bit |
| **Bottom Hole Balling** | High Baseline | Low Steady ROP | Low Steady | Recutting crushed rock bed; inadequate bottom hole hydraulics | Optimize nozzle TFA, increase GPM |
| **Stick-Slip** | Cyclic Spikes | Fluctuating | Severe Oscillations (0 to 2x mean) | Dynamic torsional resonance; friction deceleration | Boost RPM (+20-30%), moderate WOB |
| **Bit Bounce** | Erratic Scatter | Unstable | Shock Chattering | Axial harmonic resonance across hard interbedding | Adjust RPM and WOB away from axial resonance |
| **BHA Whirl** | Elevated | Poor Penetration | High Variable Drag | Centrifugal force throws drill collars into borehole wall | Reduce RPM, stabilize string, increase WOB |
        """
    )

    st.markdown("---")

    # 2. Governing Mathematical Equations
    st.markdown("#### 📐 Mathematical Formulations")

    col_eq1, col_eq2 = st.columns(2)

    with col_eq1:
        st.markdown(
            r"""
            ##### Teale (1965) Mechanical Specific Energy
            $$\text{MSE} = \frac{\text{WOB}}{A_b} + \frac{120 \pi \cdot \text{RPM} \cdot \text{Torque}}{A_b \cdot \text{ROP}}$$

            *Where in Imperial Field Units:*
            - $\text{WOB}$ = Weight on Bit ($\text{lbf}$)
            - $\text{Torque}$ = Drilling torque ($\text{ft-lbf}$)
            - $\text{RPM}$ = Rotary speed ($\text{rev/min}$)
            - $\text{ROP}$ = Rate of penetration ($\text{ft/hr}$)
            - $A_b = \frac{\pi}{4} D_b^2$ = Bit area ($\text{in}^2$)
            - $\text{MSE}$ = Specific energy ($\text{psi}$)
            """
        )

    with col_eq2:
        st.markdown(
            r"""
            ##### Stick-Slip Severity Index (SSI) & Resonance
            $$\text{SSI} = \frac{\text{RPM}_{\max} - \text{RPM}_{\min}}{2 \cdot \text{RPM}_{\text{avg}}}$$

            - **$\text{SSI} < 0.5$**: Mild vibration (Safe)
            - **$0.5 \le \text{SSI} < 1.0$**: Moderate stick-slip
            - **$\text{SSI} \ge 1.0$**: Critical stall & backward slip

            ##### Drillstring Torsional Natural Frequency
            $$f_0 = \frac{c_s}{4 L}, \quad f_n = (2n - 1) f_0$$
            *With steel shear velocity $c_s \approx 3,170 \text{ m/s}$.*
            """
        )

    st.markdown("---")

    # 3. Executive Elevator Pitch
    st.markdown("#### 💼 Value Proposition & Engineering Pitch")
    st.info(
        """
        **DrillOpt-Engine** bridges the gap between raw drilling telemetry and automated mechanical optimization:
        - **Eliminates Floundering**: Detects when additional WOB no longer yields ROP and instead destroys cutters.
        - **Extends Bit Life**: Identifies stick-slip and bit bounce before catastrophic cutter impact delamination occurs.
        - **Generates Quantified Financial ROI**: Directly calculates rig hours saved and hundreds of thousands of dollars in reduced spread costs per well section.
        """
    )
