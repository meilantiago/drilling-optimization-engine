# 🛠️ DrillOpt-Engine: Real-Time Drilling Mechanics, MSE & Operational Optimization

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: Pytest 100%](https://img.shields.io/badge/tests-51%2F51%20passing-brightgreen.svg)](https://pytest.org)
[![Domain: Oil & Gas](https://img.shields.io/badge/Domain-Oil%20%26%20Gas%20%2F%20Drilling-orange.svg)](#)

> **Industrial-grade drilling physics engine and real-time operational dashboard for rotary drilling optimization, mechanical specific energy (MSE) tracking, torsional stick-slip diagnosis, and automated Founder Point detection.**

---

## 📌 Executive Overview & Business Value

In unconventional reservoirs such as **Vaca Muerta (Neuquén Basin, Argentina)** and deep hard-rock basins, lateral horizontal sections extend over **2,500 to 3,500 meters**. Modern drilling operations incur:
- **Rig Day Rates** between **\$40,000 and \$70,000 USD/day** (often exceeding \$100,000 USD/day with directional MWD/LWD services).
- **Non-Productive Time (NPT)**: Premature PDC bit cutter failure or drillstring twist-offs require round trips (*tripping pipe*) spanning **24 to 48 hours**, wasting **\$80,000 to \$150,000+ USD** in rig time alone.

`DrillOpt-Engine` acts as an analytical co-pilot for drilling engineers and Real-Time Operations Centers (RTOC). By combining first-principles rock mechanics, torsional elastodynamics, and high-frequency digital signal processing, it identifies drilling dysfunctions before catastrophic tool damage occurs and optimizes the operational parameter envelope (WOB, RPM, Flow Rate).

---

## 🔬 Core Physics & Governing Equations

### 1. Mechanical Specific Energy (Teale, 1965 & Dupriest, 2005)
MSE represents the mechanical work consumed to excavate a unit volume of rock:
$$\text{MSE}_{\text{Teale}} = \frac{\text{WOB}}{A_b} + \frac{120\pi \cdot \text{RPM} \cdot \text{Torque}}{A_b \cdot \text{ROP}} \quad [\text{psi}]$$

Where:
- Axial work $\left(\frac{\text{WOB}}{A_b}\right)$ accounts for $\sim 1\% - 5\%$ of total energy.
- Torsional cutting work $\left(\frac{120\pi \cdot \text{RPM} \cdot T}{A_b \cdot \text{ROP}}\right)$ accounts for $\sim 95\% - 99\%$ of total energy.

**Mechanical Efficiency & Rock Baseline (SPE/IADC 92194):**
$$\text{EFF}_m = \frac{\text{UCS}}{\text{MSE}_{\text{Teale}}} \quad (\approx 30\% - 40\% \text{ for pristine PDC bits})$$
When $\text{MSE}_{\text{Teale}} \gg \frac{\text{UCS}}{\text{EFF}_m}$, parasitic energy is being dissipated into vibrations, friction, or bit balling rather than rock removal.

### 2. Torsional Stick-Slip & Quarter-Wave Resonance
The drillstring behaves as an elastic torsional waveguide with shear wave velocity $c_s = \sqrt{G/\rho} \approx 3,170 \text{ m/s}$. Fundamental quarter-wave natural resonance is:
$$f_0 = \frac{c_s}{4L} \quad [\text{Hz}]$$

**Stick-Slip Severity Index (SSI):**
$$\text{SSI} = \frac{\text{RPM}_{\max} - \text{RPM}_{\min}}{2 \times \text{RPM}_{\text{avg}}}$$
- $\text{SSI} < 0.5$: Safe / laminar rotation.
- $0.5 \le \text{SSI} < 1.0$: Moderate torsional oscillation.
- $\text{SSI} \ge 1.0$: Full Stick-Slip (drill bit periodically stops at $0\text{ RPM}$ and whips up to $2.5\times \text{RPM}_{\text{surf}}$).

### 3. Bit Hydraulics & Hole Cleaning
$$\text{TFA} = \sum \frac{\pi}{4} \left(\frac{d_{i}}{32}\right)^2 \quad [\text{in}^2]$$
$$\text{HSI} = \frac{\Delta P_{\text{bit}} \cdot Q}{1714 \cdot A_b} \quad \left[\frac{\text{hp}}{\text{in}^2}\right]$$

### 4. Normalized Drillability & Pore Pressure ($D_c$-exponent)
$$D = \frac{\log_{10}\left(\frac{\text{ROP}}{60 \cdot \text{RPM}}\right)}{\log_{10}\left(\frac{12 \cdot \text{WOB}}{10^6 \cdot D_b}\right)}, \quad D_c = D \cdot \left(\frac{\rho_{\text{normal}}}{\rho_{\text{actual}}}\right)$$

---

## 🏗️ Repository Architecture

```text
drilling-optimization-engine/
│
├── data/
│   ├── raw/                        # Raw CWLS LAS / CSV telemetry files
│   └── processed/                  # Standardized Parquet & benchmark LAS datasets
│
├── src/
│   ├── data/
│   │   ├── unit_converter.py       # Strict bidirectional oilfield unit conversions
│   │   ├── loader.py               # LAS (lasio), Parquet & CSV ingestion with mnemonic harmonization
│   │   ├── rig_state.py            # Rule-based rig state classifier (On-bottom, Reaming, Sliding, etc.)
│   │   └── generator.py            # High-fidelity physics-coupled telemetry generator
│   │
│   ├── physics/
│   │   ├── mse.py                  # Teale & Dupriest MSE, mechanical efficiency, over-energy ratio
│   │   ├── d_exponent.py           # Jordan & Shirley D-exponent, Rehm & McClendon Dc, Eaton pore pressure
│   │   └── hydraulics.py           # TFA, nozzle velocity, bit pressure loss, HSI, annular velocity
│   │
│   ├── dynamics/
│   │   ├── stick_slip.py           # SSI, FFT & Welch PSD spectral decomposition, resonance risk
│   │   ├── bit_bounce.py           # Axial vibration index (AVI), impact shocks, off-bottom bounce
│   │   └── wear_model.py           # Cumulative dissipated energy & continuous/IADC PDC dull grade (0-8)
│   │
│   ├── optimizer/
│   │   ├── founder_point.py        # Automated founder point detection via piecewise & curvature regression
│   │   └── road_map.py             # Multi-physical operational envelope recommendation & USD ROI engine
│   │
│   └── app/
│       ├── main.py                 # Streamlit Real-Time Operations Center (RTOC) application
│       └── components/
│           ├── log_plot.py         # Plotly 7-track subsurface well log (WellCAD / Techlog style)
│           ├── cross_plots.py      # Founder Point cross-plot, FFT spectrum & MSE-UCS envelope
│           └── simulator.py        # Interactive "What-If" parameter simulator with sensitivity heatmap
│
└── tests/                          # 51 unit & end-to-end integration tests (100% pass)
    ├── test_units.py
    ├── test_loader.py
    ├── test_rig_state.py
    ├── test_mse.py
    ├── test_dynamics.py
    ├── test_optimizer.py
    └── test_end_to_end.py
```

---

## 🚀 Quickstart & Installation

### 1. Clone & Setup Environment
```bash
git clone https://github.com/meilantiago/drilling-optimization-engine.git
cd drilling-optimization-engine
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
pytest tests/ -v
# Output: 51 passed in 1.4s (100% pass rate)
```

### 3. Launch Interactive RTOC Dashboard
```bash
streamlit run src/app/main.py
```

---

## 🖥️ RTOC Dashboard Features

1. **Subsurface Multi-Track Well Log (Techlog / WellCAD Style)**:
   - Synchronized depth-indexed continuous tracks:
     - **Track 1**: Gamma Ray (GAPI) with sandstone/shale lithology cutoff.
     - **Track 2**: Weight on Bit (WOB) with dynamic Founder Point threshold.
     - **Track 3**: Surface & Bit Rotary Speed (RPM).
     - **Track 4**: Surface Drilling Torque (ft-lbf).
     - **Track 5**: ROP (m/hr) & Overpressured $D_c$-exponent.
     - **Track 6**: Teale MSE (psi) vs Rock UCS (psi) with highlighted dysfunction shading.
     - **Track 7**: Stick-Slip Severity Index (SSI) with SPE safety color bands.
2. **Real-Time Dysfunction Diagnostics**:
   - Automated Founder Point regression curve identifying maximum usable WOB before flounder.
   - Welch PSD / FFT spectrum cross-referenced against drillstring quarter-wave natural resonance modes ($f_0, 3f_0, 5f_0$).
   - Axial impact shock detector and cumulative PDC cutter dull grading.
3. **Drilling Parameter Road Map & Financial ROI Simulator**:
   - Live "What-If" parameter sliders for candidate WOB, RPM, and Rig Day Rate.
   - Real-time sensitivity contour heatmap ($WOB \times RPM$) with instant projected time saved (hours) and direct USD cost savings.
4. **Engineering Knowledge Base**:
   - ExxonMobil / Dupriest SPE 92194 symptoms matrix and operational decision workflows.

---

## 👨‍💻 Author

**Tiago Meilan**  
- **Degree**: B.S. in Physics (*Licenciatura en Ciencias Físicas*), Universidad de Buenos Aires (UBA).  
- **Specialization**: Applied Probability & Statistics, Mathematical Modeling, Real-Time Sensor Processing.  
- **LinkedIn**: [linkedin.com/in/tiago-meilan](https://www.linkedin.com/in/tiago-meilan-6a723a260/)  
- **GitHub**: [github.com/meilantiago](https://github.com/meilantiago)  
- **Email**: meilantiago@gmail.com
