# DrillOpt-Engine

A Python library and real-time operations telemetry framework for rotary drilling mechanics, Mechanical Specific Energy (MSE) monitoring, torsional vibration diagnostics, and drilling parameter optimization.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests: 51/51 Passing](https://img.shields.io/badge/tests-51%2F51%20passing-brightgreen.svg)](https://pytest.org)
[![Data Formats: LAS 2.0 | Parquet](https://img.shields.io/badge/formats-LAS%202.0%20%7C%20Parquet-orange.svg)](#telemetry-ingestion-and-standards)

---

## Overview

DrillOpt-Engine processes surface and downhole measurement-while-drilling (MWD) telemetry to compute real-time rock mechanics indicators, diagnose drilling dysfunctions, and establish the operational parameter envelope (WOB, RPM, Flow Rate).

In deep horizontal and directional wells, non-productive time (NPT) caused by drillstring vibrations, bit balling, and premature cutter degradation represents a significant operational cost. DrillOpt-Engine implements analytical models established in SPE literature to:
- Monitor **Mechanical Specific Energy (MSE)** continuously and compare it against formation rock strength (UCS).
- Detect **torsional stick-slip** oscillations and identify drillstring acoustic resonance risks using spectral decomposition (FFT / Welch PSD).
- Locate the **Founder Point** (the physical threshold where increasing WOB ceases to improve ROP due to bit balling or bottom-hole cutter pack-off).
- Deliver real-time diagnostic advisories conforming to the **SPE/IADC 92194** diagnostic workflow.
- Simulate and visualize drilling telemetry through a multi-track subsurface log viewer and an interactive Real-Time Operations Center (RTOC) dashboard.

---

## Core Formulation and Governing Equations

### 1. Mechanical Specific Energy (MSE)

Mechanical Specific Energy quantifies the work required to remove a unit volume of rock (Teale, 1965). In oilfield units:

$$\text{MSE}_{\text{Teale}} = \frac{\text{WOB}}{A_b} + \frac{120\pi \cdot \text{RPM} \cdot T}{A_b \cdot \text{ROP}} \quad [\text{psi}]$$

Where:
- $\text{WOB}$ is the weight on bit ($\text{lbf}$).
- $A_b = \frac{\pi}{4} D_b^2$ is the cross-sectional bit area ($\text{in}^2$).
- $\text{RPM}$ is the rotary speed ($\text{rev/min}$).
- $T$ is the drillstring torque ($\text{ft-lbf}$).
- $\text{ROP}$ is the rate of penetration ($\text{ft/hr}$).

The axial component $\frac{\text{WOB}}{A_b}$ generally contributes less than 5% of total energy in rotary drilling; the torsional term accounts for the remaining 95%+.

#### Dupriest Formulation and Mechanical Efficiency (SPE/IADC 92194)
Dupriest and Koederitz (2005) introduced the baseline mechanical efficiency factor ($EFF_m$, typically $\sim 0.35$ for pristine PDC bits):

$$\text{MSE}_{\text{Dupriest}} = EFF_m \cdot \text{MSE}_{\text{Teale}}$$

$$\text{Efficiency } (EFF_m) = \frac{\text{UCS}}{\text{MSE}_{\text{Teale}}}$$

When $\text{MSE} \approx \frac{\text{UCS}}{EFF_m}$, rock destruction occurs through efficient shearing. A sharp increase in MSE without a corresponding increase in UCS indicates parasitic energy dissipation (e.g., bit balling, interfacial cutter wear, or vibrations).

---

### 2. Torsional Drillstring Dynamics and Stick-Slip

The drillstring is modeled as an elastic torsional waveguide. The propagation speed of torsional shear waves in API 5DP steel pipe is:

$$c_s = \sqrt{\frac{G}{\rho}} \approx 3,170 \text{ m/s}$$

For a drillstring of total length $L$, the fundamental quarter-wave natural torsional frequency ($f_0$) is:

$$f_0 = \frac{c_s}{4L} \quad [\text{Hz}]$$

#### Stick-Slip Severity Index (SSI)
Following standard IADC definitions, torsional vibration amplitude is normalized as:

$$\text{SSI} = \frac{\text{RPM}_{\max} - \text{RPM}_{\min}}{2 \cdot \text{RPM}_{\text{avg}}}$$

- **$\text{SSI} < 0.5$ (Level 1)**: Negligible / stable rotation.
- **$0.5 \le \text{SSI} < 1.0$ (Level 2)**: Moderate torsional vibration; elevated cyclic cutter fatigue.
- **$\text{SSI} \ge 1.0$ (Level 3 - Critical)**: Full stick-slip. The bit periodically halts ($0\text{ RPM}$) while torque accumulates until static friction breaks, causing angular acceleration up to $2.5 \times \text{RPM}_{\text{surf}}$.

Frequencies are extracted via Fast Fourier Transform (FFT) and Welch Power Spectral Density (PSD) to detect whether surface excitation frequencies or BHA harmonics align with $f_0, 3f_0, 5f_0$.

---

### 3. Normalized Drillability and Pore Pressure ($D$-exponent)

To decouple rock drillability from surface operational parameters, Jordan and Shirley (1966) formulated the dimensionless $D$-exponent:

$$D = \frac{\log_{10}\left(\frac{\text{ROP}}{60 \cdot \text{RPM}}\right)}{\log_{10}\left(\frac{12 \cdot \text{WOB}}{10^6 \cdot D_b}\right)}$$

Rehm and McClendon (1971) corrected for mud weight overbalance ($D_c$):

$$D_c = D \cdot \left(\frac{\rho_{\text{normal}}}{\rho_{\text{actual}}}\right)$$

Reversals in the normally increasing compaction trend of $D_c$ with depth are monitored to identify the onset of abnormal formation pore pressure (Eaton, 1975).

---

### 4. Bit Hydraulics

- **Total Flow Area (TFA)** across $N$ nozzles sized in 32nds of an inch:
  $$\text{TFA} = \sum_{i=1}^{N} \frac{\pi}{4} \left(\frac{d_i}{32}\right)^2 \quad [\text{in}^2]$$
- **Nozzle Exit Velocity ($v_n$):**
  $$v_n = \frac{0.32086 \cdot Q}{\text{TFA}} \quad [\text{ft/s}]$$
- **Bit Pressure Drop ($\Delta P_{\text{bit}}$):**
  $$\Delta P_{\text{bit}} = \frac{\rho \cdot Q^2}{10858 \cdot C_d^2 \cdot \text{TFA}^2} \quad [\text{psi}]$$
- **Hydraulic Horsepower per Square Inch (HSI):**
  $$\text{HSI} = \frac{\Delta P_{\text{bit}} \cdot Q}{1714 \cdot A_b} \quad \left[\frac{\text{hp}}{\text{in}^2}\right]$$
- **Jet Impact Force ($F_j$):**
  $$F_j = 0.01823 \cdot C_d \cdot Q \cdot \sqrt{\rho \cdot \Delta P_{\text{bit}}} \quad [\text{lbf}]$$

---

### 5. Automated Founder Point Detection

The Founder Point represents the mechanical limit where cutter penetration saturates. The algorithm evaluates localized depth windows:
1. Performs piecewise linear and polynomial curvature regression on $\text{ROP} = f(\text{WOB})$.
2. Identifies the point of diminishing returns where $\frac{\partial \text{ROP}}{\partial \text{WOB}} \to 0$ and $\frac{\partial \text{MSE}}{\partial \text{WOB}} > 0$.
3. Establishes the safe operating envelope at $90\%$ of $\text{WOB}_{\text{founder}}$ to maximize ROP without inducing cutter balling or stick-slip.

---

## Repository Architecture

```text
drilling-optimization-engine/
├── data/
│   ├── raw/                        # Raw CWLS LAS and CSV telemetry logs
│   └── processed/                  # Standardized Parquet and benchmark LAS logs
├── src/
│   ├── data/
│   │   ├── unit_converter.py       # Strict bidirectional unit conversion engine
│   │   ├── loader.py               # LAS 2.0 (lasio), Parquet and CSV ingestion
│   │   ├── rig_state.py            # Rule-based rig state classification
│   │   └── generator.py            # Synthetic physics-coupled telemetry generator
│   ├── physics/
│   │   ├── mse.py                  # Teale and Dupriest MSE, mechanical efficiency
│   │   ├── d_exponent.py           # D, Dc-exponent and Eaton pore pressure
│   │   └── hydraulics.py           # TFA, nozzle velocity, delta P, HSI, jet impact
│   ├── dynamics/
│   │   ├── stick_slip.py           # SSI, FFT/Welch spectral analysis, resonance risk
│   │   ├── bit_bounce.py           # Axial vibration index (AVI), impact shocks
│   │   └── wear_model.py           # Cumulative energy dissipation and IADC bit dull grading
│   ├── optimizer/
│   │   ├── founder_point.py        # Automated founder point detection algorithms
│   │   └── road_map.py             # Operational parameter recommendation and ROI model
│   └── app/
│       ├── main.py                 # Streamlit application entrypoint
│       └── components/
│           ├── live_stream.py      # RTOC streaming simulation, alarms and oscilloscope
│           ├── log_plot.py         # Subsurface multi-track log viewer (7 synchronized tracks)
│           ├── cross_plots.py      # Founder point cross-plots and vibration spectra
│           └── simulator.py        # Interactive "What-If" parameter and ROI simulator
└── tests/                          # Automated test suite (51 unit and integration tests)
    ├── test_units.py
    ├── test_loader.py
    ├── test_rig_state.py
    ├── test_mse.py
    ├── test_dynamics.py
    ├── test_optimizer.py
    └── test_end_to_end.py
```

---

## Telemetry Ingestion and Standards

The data loader ([`src/data/loader.py`](file:///c:/Users/Usuario%201/Desktop/CONSEGUIR%20TRABAJO/Proyectos/drilling-optimization-engine/src/data/loader.py)) harmonizes heterogeneous oilfield telemetry mnemonics:

| Canonical Channel | Standard Unit | Recognized MWD/LWD Mnemonics |
|---|---|---|
| `depth` | $\text{m}$ or $\text{ft}$ | `DEPT`, `DEPTH`, `MD`, `MEASURED_DEPTH` |
| `wob` | $\text{klbf}$ | `WOB`, `SWOB`, `DWOB`, `WOB_K`, `WEIGHT_ON_BIT` |
| `rop` | $\text{m/hr}$ or $\text{ft/hr}$ | `ROP`, `ROPA`, `ROP_AVG`, `RATE_OF_PENETRATION` |
| `rpm` | $\text{rev/min}$ | `RPM`, `SRPM`, `RPM_SURF`, `ROTARY_SPEED` |
| `torque` | $\text{ft-lbf}$ or $\text{kN}\cdot\text{m}$ | `TORQUE`, `STOR`, `TORQ`, `TQ`, `SURFACE_TORQUE` |
| `flow` | $\text{gpm}$ or $\text{L/min}$ | `FLOW`, `FLOWIN`, `PUMP`, `FLOW_RATE`, `Q` |
| `spp` | $\text{psi}$ | `SPP`, `PUMP_PRESS`, `PRESSURE`, `STANDPIPE_PRESSURE` |
| `gamma` | $\text{gAPI}$ | `GR`, `GAM`, `GAMMA`, `GR_CORR`, `GAMMA_RAY` |

---

## Installation and Setup

### Prerequisites
- Python 3.11 or higher
- Git

### 1. Clone Repository
```bash
git clone https://github.com/meilantiago/drilling-optimization-engine.git
cd drilling-optimization-engine
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Linux / macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Verification and Testing

The repository includes 51 automated unit and integration tests verifying numerical consistency, unit conversions, division-by-zero handling, and end-to-end pipeline execution.

Run the test suite via pytest:
```bash
python -m pytest tests/ -v
```

Expected output:
```text
============================= 51 passed in 1.60s =============================
```

---

## Running the RTOC Dashboard

Launch the Streamlit web application:
```bash
python -m streamlit run src/app/main.py
```

### Features available in the application:
1. **Live RTOC Telemetry Streaming Simulation**:
   - Interactive transport controls (`Start Live Stream`, `Pause`, `Step +10m`, `Reset Bit`).
   - Dynamic real-time MWD gauges updating as the bit advances through geological formations.
   - Rule-based **SPE 92194 Real-Time Advisory Banners** detecting clean shearing, founder point, or critical stick-slip with actionable operational mitigations.
   - High-frequency 4-channel strip-chart oscilloscope.
2. **Subsurface Multi-Track Log View**:
   - 7 synchronized depth-indexed tracks: Gamma Ray, WOB, RPM (surface vs. bit), Torque, ROP with $D_c$-exponent, Teale MSE vs. Rock UCS with dysfunction shading, and Stick-Slip Severity Index.
3. **Dysfunction Diagnostics**:
   - Localized Founder Point regression cross-plots.
   - Welch PSD frequency spectrum cross-referenced with drillstring acoustic resonance modes ($f_0, 3f_0, 5f_0$).
   - Cumulative PDC cutter energy dissipation and IADC dull grading.
4. **Drilling Optimization and ROI**:
   - Formation-by-formation parameter road map with recommended WOB, RPM, and flow rates.
   - Rig Day Rate savings calculator quantifying drilling hours saved and direct USD cost reductions.
5. **Geological Benchmark Generator**:
   - Interactive generator producing randomized synthetic wells with variable stratigraphy, rock hardness, and sensor noise for algorithm validation.

---

## References

1. **Teale, R.** (1965). *The Concept of Specific Energy in Rock Drilling*. International Journal of Rock Mechanics and Mining Sciences & Geomechanics Abstracts, 2(1), 57-73.
2. **Dupriest, F. E., & Koederitz, W. L.** (2005). *Maximizing Drill Rates with Real-Time Surveillance of Mechanical Specific Energy*. SPE/IADC Drilling Conference, SPE/IADC 92194.
3. **Jordan, J. R., & Shirley, O. J.** (1966). *Application of Recommended Formation Evaluation Programs in the Gulf Coast Area*. Journal of Petroleum Technology, 18(10), 1335-1342. SPE 1395.
4. **Rehm, W. A., & McClendon, R.** (1971). *Measurement of Formation Pressure From Drilling Data*. SPE Annual Fall Meeting, SPE 3601.
5. **Eaton, B. A.** (1975). *The Equation for Geopressure Prediction from Well Logs*. SPE Annual Fall Meeting, SPE 5544.
6. **Brett, J. F.** (1992). *The Genesis of Torsional Drillstring Vibrations*. SPE Drilling Engineering, 7(3), 168-174. SPE 18049.

---

## Author

**Tiago Meilan**  
- **Degree**: Licenciatura en Ciencias Físicas, Universidad de Buenos Aires (UBA).  
- **Fields of interest**: Applied Probability & Statistics, Mathematical Physics, Drilling Mechanics & Geomechanics.  
- **GitHub**: [github.com/meilantiago](https://github.com/meilantiago)  
- **LinkedIn**: [linkedin.com/in/tiago-meilan](https://www.linkedin.com/in/tiago-meilan-6a723a260/)  
- **Email**: meilantiago@gmail.com
