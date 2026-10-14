"""
Synthetic & Benchmark Drilling Telemetry Generator
===================================================
High-fidelity physical simulation of drilling dynamics and surface/downhole telemetry:
- Stratified geological layers: Sandstone, Shale, Limestone, Dolomite with realistic UCS & Gamma
- Physics-coupled cutting mechanics: linear cutting regime transitioning into Founder Point saturation
- Mechanical Specific Energy (MSE) explosion upon bit balling / over-foundering
- Intermittent stick-slip torsional limit cycles in high-friction / hard rocks
- Bit bounce axial vibrations across interbedded formation transitions
- Sensor Gaussian noise and measurement artifacts
- Dual export: CWLS LAS 2.0 (via `lasio`) and Apache Parquet formats

Author: Tiago Meilan
Project: DrillOpt-Engine
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import lasio
import numpy as np
import pandas as pd

from .unit_converter import UnitConverter


@dataclass
class FormationLayer:
    """Geological formation layer properties."""
    name: str
    top_depth_m: float
    bottom_depth_m: float
    ucs_psi: float
    gamma_mean_api: float
    gamma_std_api: float
    friction_coeff: float
    founder_wob_klbf: float
    stick_slip_susceptibility: float  # 0.0 (immune) to 1.0 (extreme stick-slip risk)
    drillability_k: float = 1.0       # Empirical drillability index


DEFAULT_FORMATIONS: List[FormationLayer] = [
    FormationLayer(
        name="Shale",
        top_depth_m=2000.0,
        bottom_depth_m=2120.0,
        ucs_psi=8000.0,
        gamma_mean_api=110.0,
        gamma_std_api=8.0,
        friction_coeff=0.28,
        founder_wob_klbf=24.0,
        stick_slip_susceptibility=0.15,
        drillability_k=1.1,
    ),
    FormationLayer(
        name="Sandstone",
        top_depth_m=2120.0,
        bottom_depth_m=2240.0,
        ucs_psi=12000.0,
        gamma_mean_api=48.0,
        gamma_std_api=6.0,
        friction_coeff=0.35,
        founder_wob_klbf=32.0,
        stick_slip_susceptibility=0.25,
        drillability_k=0.95,
    ),
    FormationLayer(
        name="Limestone",
        top_depth_m=2240.0,
        bottom_depth_m=2360.0,
        ucs_psi=22000.0,
        gamma_mean_api=28.0,
        gamma_std_api=5.0,
        friction_coeff=0.46,
        founder_wob_klbf=38.0,
        stick_slip_susceptibility=0.75,
        drillability_k=0.70,
    ),
    FormationLayer(
        name="Dolomite",
        top_depth_m=2360.0,
        bottom_depth_m=2450.0,
        ucs_psi=28000.0,
        gamma_mean_api=18.0,
        gamma_std_api=4.0,
        friction_coeff=0.52,
        founder_wob_klbf=42.0,
        stick_slip_susceptibility=0.90,
        drillability_k=0.55,
    ),
    FormationLayer(
        name="Sandstone_Lower",
        top_depth_m=2450.0,
        bottom_depth_m=2500.0,
        ucs_psi=11500.0,
        gamma_mean_api=52.0,
        gamma_std_api=7.0,
        friction_coeff=0.34,
        founder_wob_klbf=30.0,
        stick_slip_susceptibility=0.20,
        drillability_k=1.0,
    ),
]


class DrillingDataGenerator:
    """
    Simulates high-resolution, physics-coupled drilling telemetry.
    Models ROP response, founder saturation, stick-slip limit cycles, and bit bounce.
    """

    def __init__(
        self,
        bit_diameter_in: float = 8.5,
        mud_weight_ppg: float = 10.2,
        formations: Optional[List[FormationLayer]] = None,
        random_seed: int = 42,
    ):
        self.bit_diameter_in = bit_diameter_in
        self.mud_weight_ppg = mud_weight_ppg
        self.bit_area_in2 = (np.pi / 4.0) * (bit_diameter_in ** 2)
        self.formations = formations if formations is not None else DEFAULT_FORMATIONS
        self.rng = np.random.default_rng(random_seed)

    def _get_formation_at_depth(self, depth_m: float) -> FormationLayer:
        """Finds the active formation layer corresponding to a measured depth."""
        for layer in self.formations:
            if layer.top_depth_m <= depth_m <= layer.bottom_depth_m:
                return layer
        return self.formations[-1]

    def generate_profile(
        self,
        start_depth_m: float = 2000.0,
        end_depth_m: float = 2500.0,
        depth_step_m: float = 0.2,
        base_rpm: float = 120.0,
        base_flow_gpm: float = 550.0,
        add_founder_sweeps: bool = True,
    ) -> pd.DataFrame:
        """
        Generates continuous telemetry depth profile across all geological formations.

        Parameters
        ----------
        start_depth_m : float
            Starting measured depth (meters).
        end_depth_m : float
            Ending measured depth (meters).
        depth_step_m : float
            Depth interval increment (meters).
        base_rpm : float
            Nominal surface rotary speed (RPM).
        base_flow_gpm : float
            Nominal mud flow rate (GPM).
        add_founder_sweeps : bool
            Whether to inject realistic WOB drill-off / ramp sweeps that trigger
            the founder point saturation regime.

        Returns
        -------
        pd.DataFrame
            Telemetry dataframe containing canonical channels:
            depth, rop, wob, rpm, torque, flow, spp, gamma,
            along with physics ground-truth: formation, ucs_psi, mse_psi, ssi.
        """
        depths = np.arange(start_depth_m, end_depth_m + depth_step_m, depth_step_m)
        n_samples = len(depths)

        # Allocate channel arrays
        wob_arr = np.zeros(n_samples)
        rpm_arr = np.zeros(n_samples)
        rop_arr = np.zeros(n_samples)
        torque_arr = np.zeros(n_samples)
        flow_arr = np.zeros(n_samples)
        spp_arr = np.zeros(n_samples)
        gamma_arr = np.zeros(n_samples)
        formation_names: List[str] = []
        ucs_arr = np.zeros(n_samples)
        ssi_arr = np.zeros(n_samples)
        founder_wob_arr = np.zeros(n_samples)

        # Formation boundaries for bit bounce detection
        boundary_depths = [f.top_depth_m for f in self.formations if f.top_depth_m > start_depth_m]

        for i, d in enumerate(depths):
            fm = self._get_formation_at_depth(d)
            formation_names.append(fm.name)
            ucs_arr[i] = fm.ucs_psi
            founder_wob = fm.founder_wob_klbf
            founder_wob_arr[i] = founder_wob

            # Base operational commands
            flow = base_flow_gpm + self.rng.normal(0, 4.0)
            rpm_cmd = base_rpm + self.rng.normal(0, 1.5)

            # WOB profile: realistic operational variation + step tests / founder sweeps
            rel_depth_in_layer = (d - fm.top_depth_m) / max(1.0, (fm.bottom_depth_m - fm.top_depth_m))
            
            if add_founder_sweeps:
                # Drill-off / WOB sweep pattern: ramps from 12 klbf up to 1.25x founder WOB
                # creating periodic founder saturation zones to benchmark optimizer
                cycle_phase = (d % 35.0) / 35.0  # 35m drill-off sweep cycle
                wob_target = 14.0 + cycle_phase * (founder_wob * 1.22 - 14.0)
            else:
                wob_target = founder_wob * 0.88 + 3.0 * np.sin(d / 12.0)

            # Bit bounce effect near layer boundaries (axial vibration)
            bounce_wob = 0.0
            for b_depth in boundary_depths:
                dist = abs(d - b_depth)
                if dist < 2.5:  # within 2.5 meters of boundary
                    # Damped harmonic oscillation
                    bounce_wob += 7.5 * np.exp(-dist / 1.0) * np.cos(2.0 * np.pi * dist / 0.5)

            wob = max(2.0, wob_target + bounce_wob + self.rng.normal(0, 0.45))
            wob_arr[i] = wob

            # Stick-slip torsional dynamics
            # Risk increases with high WOB, hard/frictional formation, and lower RPM
            wob_ratio = wob / founder_wob
            ss_excitation = fm.stick_slip_susceptibility * (wob_ratio ** 1.8) * (100.0 / max(50.0, rpm_cmd))
            ssi = float(np.clip(ss_excitation + self.rng.normal(0, 0.04), 0.05, 1.45))
            ssi_arr[i] = ssi

            # Instantaneous / oscillatory RPM response when stick-slip occurs
            if ssi > 0.5:
                # Oscillations between 0 and 2.5x base RPM
                rpm_fluctuation = ssi * base_rpm * np.sin(2.0 * np.pi * d / 0.8)
                rpm_actual = float(np.clip(rpm_cmd + rpm_fluctuation, 0.0, 2.5 * base_rpm))
            else:
                rpm_actual = float(max(10.0, rpm_cmd + self.rng.normal(0, 1.2)))
            rpm_arr[i] = rpm_actual

            # Cutting Physics: Rate of Penetration (ROP)
            # Reference rock UCS = 10,000 psi
            rock_strength_factor = 10000.0 / fm.ucs_psi
            cleaning_factor = (flow / 500.0) ** 0.35
            speed_factor = (rpm_actual / 100.0) ** 0.85

            # Founder Point Transition:
            # For WOB < WOB_founder: Linear cutting regime
            # For WOB >= WOB_founder: ROP saturates, rolls over, and bit balls up
            w_norm = wob / founder_wob
            if w_norm <= 1.0:
                # Linear regime with slight hydraulic softening as w_norm approaches 1
                rop_response = w_norm * (1.0 - 0.08 * (w_norm ** 2))
            else:
                # Floundering regime: severe ROP drop-off due to chip recutting & balling
                over_wob = w_norm - 1.0
                rop_response = 0.92 * np.exp(-1.8 * (over_wob ** 1.4))

            base_rop_m_hr = (
                32.0
                * fm.drillability_k
                * rock_strength_factor
                * speed_factor
                * cleaning_factor
                * rop_response
            )

            # Add bit bounce disturbance and sensor Gaussian noise
            rop_noise = self.rng.normal(0, 0.4)
            rop = float(max(0.2, base_rop_m_hr + rop_noise))
            rop_arr[i] = rop

            # Torque Physics: Bit torque + drillstring drag
            # T_bit = 1/3 * mu * WOB * D_bit
            # Convert WOB (klbf) to lbf: wob * 1000
            # Bit torque in ft-lbf:
            # Over-foundering increases friction and bit chatter dramatically
            mu_effective = fm.friction_coeff * (1.0 + max(0.0, w_norm - 1.0) * 0.85)
            bit_torque = (1.0 / 3.0) * mu_effective * (wob * 1000.0) * (self.bit_diameter_in / 12.0)
            drillstring_drag = 0.95 * d  # parasitic drag increasing with depth
            torque_noise = self.rng.normal(0, 120.0)
            torque = float(max(100.0, bit_torque + drillstring_drag + torque_noise))
            torque_arr[i] = torque

            # Standpipe Pressure (SPP):
            # Parasitic line losses (~ Q^1.8 * depth) + bit nozzle pressure drop (~ Q^2)
            parasitic_loss = 0.0035 * (flow ** 1.78) * (d / 1000.0)
            bit_nozzle_drop = 0.0042 * (flow ** 2.0)
            spp_noise = self.rng.normal(0, 18.0)
            spp = float(max(500.0, parasitic_loss + bit_nozzle_drop + spp_noise))
            spp_arr[i] = spp

            # Gamma Ray (API)
            gamma = float(np.clip(
                self.rng.normal(fm.gamma_mean_api, fm.gamma_std_api),
                5.0,
                200.0
            ))
            gamma_arr[i] = gamma
            flow_arr[i] = flow

        # Mechanical Specific Energy (Teale, 1965)
        # MSE = (WOB / A_bit) + (120 * pi * RPM * Torque) / (A_bit * ROP)
        # Imperial: WOB in lbf, Torque in ft-lbf, ROP in ft/hr, Bit Area in in2 -> MSE in psi
        wob_lbf = wob_arr * 1000.0
        rop_ft_hr = rop_arr / 0.3048
        axial_term = wob_lbf / self.bit_area_in2
        torsional_term = (120.0 * np.pi * rpm_arr * torque_arr) / (self.bit_area_in2 * np.maximum(rop_ft_hr, 0.1))
        mse_arr = axial_term + torsional_term

        df = pd.DataFrame({
            "depth": np.round(depths, 2),
            "rop": np.round(rop_arr, 2),
            "wob": np.round(wob_arr, 2),
            "rpm": np.round(rpm_arr, 1),
            "torque": np.round(torque_arr, 1),
            "flow": np.round(flow_arr, 1),
            "spp": np.round(spp_arr, 1),
            "gamma": np.round(gamma_arr, 1),
            "formation": formation_names,
            "ucs_psi": np.round(ucs_arr, 0),
            "founder_wob": np.round(founder_wob_arr, 2),
            "ssi": np.round(ssi_arr, 3),
            "mse_psi": np.round(mse_arr, 1),
        })

        return df

    def export_dataset(
        self,
        df: pd.DataFrame,
        parquet_path: Union[str, Path],
        las_path: Optional[Union[str, Path]] = None,
        well_name: str = "BENCHMARK_WELL_01",
    ) -> Tuple[Path, Optional[Path]]:
        """
        Exports the generated drilling telemetry to Parquet and CWLS LAS 2.0.

        Parameters
        ----------
        df : pd.DataFrame
            Drilling telemetry dataframe.
        parquet_path : Union[str, Path]
            Destination path for Parquet file.
        las_path : Optional[Union[str, Path]]
            Destination path for CWLS LAS file.
        well_name : str
            Well name identifier for the LAS header.

        Returns
        -------
        Tuple[Path, Optional[Path]]
            Paths to the written Parquet and LAS files.
        """
        p_parquet = Path(parquet_path)
        p_parquet.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p_parquet, index=False)

        p_las = None
        if las_path is not None:
            p_las = Path(las_path)
            p_las.parent.mkdir(parents=True, exist_ok=True)

            las = lasio.LASFile()
            las.well.WELL.value = well_name
            las.well.FLD.value = "BENCHMARK_DRILLOPT_BASIN"
            las.well.CNTY.value = "USA"
            las.well.NULL.value = -999.25

            # Required canonical drilling curves
            las.append_curve("DEPTH", df["depth"].values, unit="M", descr="Measured Depth")
            las.append_curve("ROP", df["rop"].values, unit="M/HR", descr="Rate of Penetration")
            las.append_curve("WOB", df["wob"].values, unit="KLBF", descr="Weight on Bit")
            las.append_curve("RPM", df["rpm"].values, unit="RPM", descr="Rotary Speed")
            las.append_curve("TORQUE", df["torque"].values, unit="FT-LBF", descr="Surface Torque")
            las.append_curve("FLOW", df["flow"].values, unit="GPM", descr="Pump Flow Rate")
            las.append_curve("SPP", df["spp"].values, unit="PSI", descr="Standpipe Pressure")
            las.append_curve("GAMMA", df["gamma"].values, unit="GAPI", descr="Gamma Ray")

            # Additional physics curves if present
            if "mse_psi" in df.columns:
                las.append_curve("MSE", df["mse_psi"].values, unit="PSI", descr="Mechanical Specific Energy")
            if "ssi" in df.columns:
                las.append_curve("SSI", df["ssi"].values, unit="UNITLESS", descr="Stick Slip Severity Index")

            las.write(str(p_las), version=2.0)

        return p_parquet, p_las


def generate_benchmark_files(
    output_dir: Union[str, Path] = "data/processed",
    start_depth: float = 2000.0,
    end_depth: float = 2500.0,
) -> Tuple[Path, Path]:
    """
    Convenience function to generate and save standard benchmark files:
    `data/processed/benchmark_drilling.parquet` and
    `data/processed/benchmark_drilling.las`.
    """
    generator = DrillingDataGenerator()
    df = generator.generate_profile(start_depth_m=start_depth, end_depth_m=end_depth)

    out_dir = Path(output_dir)
    parquet_path = out_dir / "benchmark_drilling.parquet"
    las_path = out_dir / "benchmark_drilling.las"

    generator.export_dataset(df, parquet_path=parquet_path, las_path=las_path)
    return parquet_path, las_path


if __name__ == "__main__":
    pq, las = generate_benchmark_files()
    print(f"Generated benchmark files:\n  - Parquet: {pq}\n  - LAS: {las}")

