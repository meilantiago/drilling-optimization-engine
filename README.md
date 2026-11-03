# DrillOpt-Engine

A Python library for rotary drilling mechanics, Mechanical Specific Energy (MSE) monitoring, torsional vibration diagnostics, and drilling parameter optimization.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/meilantiago/drilling-optimization-engine/actions/workflows/tests.yml/badge.svg)](https://github.com/meilantiago/drilling-optimization-engine/actions/workflows/tests.yml)
[![Data Formats: LAS 2.0 | Parquet](https://img.shields.io/badge/formats-LAS%202.0%20%7C%20Parquet-orange.svg)](#telemetry-ingestion-and-standards)

---

## Overview

DrillOpt-Engine processes surface and downhole measurement-while-drilling (MWD) telemetry to compute real-time rock mechanics indicators, diagnose drilling dysfunctions, and establish the operational parameter envelope (WOB, RPM, Flow Rate).

In deep horizontal and directional wells, non-productive time (NPT) caused by drillstring vibrations, bit balling, and premature cutter degradation represents a significant operational cost. DrillOpt-Engine implements analytical models established in SPE literature to:
- Monitor **Mechanical Specific Energy (MSE)** continuously and compare it against formation rock strength (UCS).
- Detect **torsional stick-slip** oscillations and identify drillstring acoustic resonance risks using spectral decomposition (FFT / Welch PSD).
- Locate the **Founder Point** (the physical threshold where increasing WOB ceases to improve ROP due to bit balling or bottom-hole cutter pack-off).
- Deliver real-time diagnostic advisories conforming to the **SPE/IADC 92194** diagnostic workflow.

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

To decouple rock drillability from surface operational parameters, Jorden and Shirley (1966) formulated the dimensionless $D$-exponent:

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
  $$\Delta P_{\text{bit}} = \frac{\rho \cdot Q^2}{12032 \cdot C_d^2 \cdot \text{TFA}^2} \quad [\text{psi}]$$
- **Hydraulic Horsepower per Square Inch (HSI):**
  $$\text{HSI} = \frac{\Delta P_{\text{bit}} \cdot Q}{1714 \cdot A_b} \quad \left[\frac{\text{hp}}{\text{in}^2}\right]$$
- **Jet Impact Force ($F_j$):**
  $$F_j = \frac{\rho \cdot Q \cdot v_n}{1930.4} \quad [\text{lbf}]$$

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
│   └── optimizer/
│       ├── founder_point.py        # Automated founder point detection algorithms
│       └── road_map.py             # Operational parameter recommendation and ROI model
└── tests/                          # Automated pytest suite
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

The data loader ([`src/data/loader.py`](src/data/loader.py)) maps common oilfield telemetry mnemonics to canonical channel names. Units are **not** converted automatically: input curves must already be in the units below.

| Canonical Channel | Expected Unit | Recognized Mnemonics |
|---|---|---|
| `depth` | m | `DEPTH`, `DEPT`, `MD`, `MEASURED_DEPTH`, `BIT_DEPTH`, `HOLE_DEPTH` |
| `rop` | m/hr | `ROP`, `ROPA`, `ROP_AVG`, `RATE_OF_PENETRATION`, `ROP5`, `ROPM` |
| `wob` | klbf | `WOB`, `SWOB`, `DWOB`, `WEIGHT_ON_BIT`, `WOB_AVG`, `WOB_SURF` |
| `rpm` | rev/min | `RPM`, `SRPM`, `DRPM`, `SURF_RPM`, `BIT_RPM`, `ROTARY_SPEED` |
| `torque` | ft-lbf | `TORQUE`, `TORQ`, `STOR`, `SURF_TORQUE`, `BIT_TORQUE`, `TORQUE_AVG` |
| `flow` | gpm | `FLOW`, `FLOW_IN`, `FLOWIN`, `PUMP_FLOW`, `FLOW_RATE`, `Q_PUMP` |
| `spp` | psi | `SPP`, `STANDPIPE_PRESSURE`, `PUMP_PRESS`, `PUMP_PRESSURE`, `PRESS_SPP` |
| `gamma` | gAPI | `GR`, `GAM`, `GAMMA`, `GR_CAL`, `GAMMA_RAY`, `GRD` |

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

The repository includes 52 automated unit and integration tests verifying numerical consistency, unit conversions, division-by-zero handling, and end-to-end pipeline execution.

Run the test suite via pytest:
```bash
python -m pytest tests/ -v
```

Expected output:
```text
============================= 52 passed =============================
```

---

## Limitations

All results in this repository come from a **synthetic** benchmark well (`data/processed/benchmark_drilling.*`) produced by `src/data/generator.py`. No field data has been run through the pipeline yet. Known gaps:

- **Founder point validation is circular.** The generator places the ROP peak at a known `founder_wob` per formation and the analyzer recovers it. Recovery is close in shale and sandstone, but it is off by about 11% in limestone and 24% in dolomite, where the interval is misclassified as pre-founder.
- **Projected ROP gain and savings are rule-based.** The road map applies fixed scaling factors and never projects a ROP below baseline, so it reports a gain even on flat data with no WOB exploration.
- **Simulator columns leak into the optimizer.** When the input has `ssi` or `mse_psi` columns (the benchmark does), they are used directly instead of being recomputed.
- **Vibration analysis on a depth log.** Stick-slip and bit bounce need high-frequency time series (downhole RPM, accelerations). At 0.2 m sampling (~40 s per sample) the indices are only depth-averaged proxies, and SSI is computed from surface RPM.
- **No unit conversion on LAS load.** Curves must already be in the expected units listed above.
- **Founder definition.** The founder is taken as the ROP maximum of a fitted parabola, while Dupriest defines it as the onset of non-linear ROP response. A single founder is applied across all formations in an interval.
- **Hydraulics** cover bit-level terms only (TFA, jet velocity, bit ΔP, HSI, impact force). There are no parasitic pressure losses, ECD or rheology models, so SPP is not used as a constraint.
- **Bit wear** is a relative excess-energy index, not a calibrated IADC dull grade.

---

## References

1. **Teale, R.** (1965). *The Concept of Specific Energy in Rock Drilling*. International Journal of Rock Mechanics and Mining Sciences & Geomechanics Abstracts, 2(1), 57-73.
2. **Dupriest, F. E., & Koederitz, W. L.** (2005). *Maximizing Drill Rates with Real-Time Surveillance of Mechanical Specific Energy*. SPE/IADC Drilling Conference, SPE/IADC 92194.
3. **Jorden, J. R., & Shirley, O. J.** (1966). *Application of Drilling Performance Data to Overburden and Formation Pressure Detection*. Journal of Petroleum Technology, 18(11). SPE 1407.
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
