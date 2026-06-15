"""Scenario data model + geometry/footprint helpers for the FWU simulator."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import numpy as np

from fwu.constants import KNOTS_TO_MPS, ORIFICE_K, CELL_SIZE_MM, HULL_SHAPES


# -----------------------------------------------------------------------------
# Scenario data class
# -----------------------------------------------------------------------------
@dataclass
class Scenario:
    """All parameters that define one simulation run."""
    # Array geometry
    array_width_mm: int = 1700
    n_row1: int = 4
    n_row2: int = 3
    disc_pitch_mm: int = 520
    row_pitch_mm: int = 260
    yaw_deg: float = 0.0

    # Disc & nozzles
    disc_diameter_mm: int = 300
    n_nozzles: int = 3
    nozzle_radius_mm: int = 140
    nozzle_cant_deg: int = 10
    standoff_mm: int = 15
    counter_rotate: bool = True

    # Operating point
    rpm: int = 850
    rov_speed_kn: float = 0.3
    nozzle_exit_mm: float = 1.3
    # Causality (fixed-displacement pumps = FLOW source): you set pump
    # frequency → flow; velocity = Q/A; nozzle pressure = ½ρ(v/Cd)² and the
    # subsea/topside pressures are all CONSEQUENCES. Flow is the single input,
    # capped at the pump capacity.
    total_flow_lpm: float = 250.0           # commanded flow (measured op point)
    pump_flow_cap_lpm: float = 270.0        # 2 Denjet CE100-300 = 4.5 L/s max
    # Umbilical hose/relief pressure ceiling — the one true PRESSURE limit: if
    # the topside needed to push the flow exceeds this, the relief bypasses
    # and full flow can't be reached.
    hose_pressure_ceiling_bar: float = 250.0
    # Topside → subsea transmission: MEASURED ~0.57 across two systems (SSO3 +
    # one other, Mar 2023–Mar 2025) — a ~43% umbilical line loss that scales
    # with pressure. Used to show the topside pressure the flow implies.
    pressure_transmission_ratio: float = 0.57

    # Jet / impact physics (submerged turbulent free jet). All "assumed"
    # values are calibratable — replace with measured numbers from a film /
    # dye firing test (see jet_velocity_at_hull / impact measures below).
    water_density: float = 1000.0      # kg/m³ (fresh; 1026 seawater)
    nozzle_cd: float = 0.92            # discharge coeff (derived ref. 140 bar — verify)
    decay_K: float = 5.5              # far-field centreline decay constant (assumed)
    core_factor: float = 5.0          # potential-core length = factor · exit dia (straight bore)
    jet_half_angle_deg: float = 14.0  # free-jet spread half-angle (assumed — measure)
    skin_friction_cf: float = 0.003   # wall-jet skin-friction coeff for shear (assumed)
    cleaning_measure: str = "Wall shear stress"  # which measure gates cleaning

    # Jet footprint model
    footprint_mode: str = "Physical jet (exit dia + standoff)"
    footprint_dia_mm_override: int = 70
    jet_spread_deg: float = 8.0     # free-jet half-cone spread angle
    pressure_profile: str = "Gaussian (peak at centre)"

    # Hull strip & threshold
    sim_length_mm: int = 2000
    clean_threshold: float = 0.5    # bar·s — dose heatmap threshold only
    auto_grid: bool = True          # derive cell size from the footprint
    cell_size_mm: float = 5.0       # manual hull-grid resolution (when auto off)

    # Cleaning criterion: an intensity GATE (the chosen impact measure at the
    # hull must exceed a removal threshold for the fouling type) plus a DOSE
    # gate (a minimum number of nozzle passes over the cell). The threshold
    # units follow cleaning_measure: bar for stagnation/mean, kPa for shear.
    removal_pressure_bar: float = 6.0    # default ~ soft biofilm in kPa shear
    min_passes: int = 2                  # passes required once above the gate

    # Post-processing
    steady_state_only: bool = True

    @property
    def n_nozzles_total(self) -> int:
        """Total nozzles in the array = discs × nozzles per disc."""
        return (self.n_row1 + self.n_row2) * self.n_nozzles

    # ---- Jet exit state (flow-controlled, clamped by both ceilings) --------
    @property
    def nozzle_exit_area_m2(self) -> float:
        d = self.nozzle_exit_mm / 1000.0
        return math.pi * (d / 2.0) ** 2

    @property
    def hose_allowed_flow_lpm(self) -> float:
        """
        Max total flow the hose/relief ceiling permits: the topside cannot
        exceed the ceiling, so the nozzle pressure ≤ ceiling × ratio, which
        bounds the velocity v = Cd·√(2·P/ρ) and hence the flow v·A·n.
        """
        p_subsea_max = self.hose_pressure_ceiling_bar \
            * self.pressure_transmission_ratio
        v_max = self.nozzle_cd * math.sqrt(
            2.0 * max(p_subsea_max, 0.0) * 1e5 / max(self.water_density, 1e-6))
        q_m3s = v_max * self.nozzle_exit_area_m2 * self.n_nozzles_total
        return q_m3s * 1000.0 * 60.0

    @property
    def delivered_flow_lpm(self) -> float:
        """
        Flow actually delivered = commanded, clamped by BOTH ceilings: the
        pump capacity AND the hose-allowed flow (the relief bypasses excess).
        This is what the jet and the sim actually run on.
        """
        return min(self.total_flow_lpm, self.pump_flow_cap_lpm,
                   self.hose_allowed_flow_lpm)

    @property
    def flow_per_nozzle_lpm(self) -> float:
        """Delivered flow per nozzle (L/min)."""
        return self.delivered_flow_lpm / max(self.n_nozzles_total, 1)

    @property
    def jet_exit_velocity(self) -> float:
        """Exit velocity v = Q/A per nozzle (m/s), from the DELIVERED flow."""
        q_m3s = self.flow_per_nozzle_lpm / 1000.0 / 60.0
        return q_m3s / max(self.nozzle_exit_area_m2, 1e-12)

    @property
    def subsea_pressure_bar(self) -> float:
        """Nozzle (subsea) pressure from the delivered flow: ½ρ(v/Cd)²."""
        v = self.jet_exit_velocity
        return 0.5 * self.water_density * (v / max(self.nozzle_cd, 1e-6)) ** 2 / 1e5

    @property
    def topside_pressure_bar(self) -> float:
        """Topside pressure the delivered flow needs = subsea ÷ transmission."""
        return self.subsea_pressure_bar / max(self.pressure_transmission_ratio,
                                               1e-6)

    @property
    def limiting_ceiling(self) -> str:
        """Which ceiling binds the delivered flow: 'pump flow', 'hose
        pressure', or 'none' (commanded flow is below both)."""
        cmd = self.total_flow_lpm
        if self.delivered_flow_lpm >= cmd - 1e-6:
            return "none"
        # Whichever ceiling is the lower one is the binding constraint.
        return ("hose pressure"
                if self.hose_allowed_flow_lpm < self.pump_flow_cap_lpm
                else "pump flow")

    @property
    def flow_throttled(self) -> bool:
        """True when a ceiling clamped delivered flow below commanded."""
        return self.delivered_flow_lpm < self.total_flow_lpm - 1e-6

    @property
    def at_flow_cap(self) -> bool:
        """True when the delivered flow is at the pump capacity."""
        return self.delivered_flow_lpm >= self.pump_flow_cap_lpm - 1e-6

    @property
    def pressure_ceiling_exceeded(self) -> bool:
        """Kept for back-compat: the hose ceiling is the binding limit."""
        return self.limiting_ceiling == "hose pressure"

    @property
    def pressure_bar(self) -> float:
        """
        Jet dynamic (stagnation) pressure at the EXIT, ½ρv². This is the
        'nozzle-side' impact pressure the jet starts with; the value reaching
        the hull is lower after the submerged-jet decay (see below). Replaces
        the earlier orifice-law nozzle pressure with the physically correct
        ½ρv² from the exit velocity.
        """
        return 0.5 * self.water_density * self.jet_exit_velocity ** 2 / 1e5

    @property
    def core_length_mm(self) -> float:
        return self.core_factor * max(self.nozzle_exit_mm, 1e-6)

    def jet_velocity_at_hull(self, standoff_mm: float | None = None) -> float:
        """
        Centreline velocity reaching the hull (m/s) for a submerged turbulent
        free jet. Constant v₀ within the potential core (≈ core_factor·d),
        then decays as v_c/v₀ = K·d/x beyond it. x is the jet-axis path
        length (standoff along the canted axis).
        """
        x = (self.standoff_mm if standoff_mm is None else standoff_mm)
        x = x / max(math.cos(math.radians(self.nozzle_cant_deg)), 1e-6)
        core = self.core_length_mm
        v0 = self.jet_exit_velocity
        if x <= core:
            return v0
        return v0 * self.decay_K * self.nozzle_exit_mm / x

    # ---- Three impact measures at the hull (all from v_c) -----------------
    def stagnation_pressure_bar(self, standoff_mm: float | None = None) -> float:
        """(1) Centreline stagnation pressure ½ρv_c² (bar) — normal head-on
        impact; drives hard-fouling fracture."""
        v = self.jet_velocity_at_hull(standoff_mm)
        return 0.5 * self.water_density * v ** 2 / 1e5

    def mean_impact_pressure_bar(self, standoff_mm: float | None = None) -> float:
        """(2) Total jet reaction force ÷ footprint area (bar) — bulk normal
        load. Falls fast as the footprint grows with standoff."""
        v0 = self.jet_exit_velocity
        q_m3s = self.flow_per_nozzle_lpm / 1000.0 / 60.0
        force = self.water_density * q_m3s * v0          # ρ·Q·v, N
        fp_m = self.footprint_dia(standoff_mm) / 1000.0
        area = math.pi * (fp_m / 2.0) ** 2
        return (force / max(area, 1e-12)) / 1e5

    def wall_shear_kpa(self, standoff_mm: float | None = None) -> float:
        """(3) Wall shear stress τ = Cf·½ρv_c² (kPa) — tangential 'scrub' the
        spreading jet exerts; the shear-removal cleaning mechanism."""
        v = self.jet_velocity_at_hull(standoff_mm)
        return self.skin_friction_cf * 0.5 * self.water_density * v ** 2 / 1000.0

    def cleaning_intensity(self, standoff_mm: float | None = None) -> float:
        """The impact measure that gates cleaning, per `cleaning_measure`.
        Units depend on the choice: bar for stagnation/mean, kPa for shear."""
        if self.cleaning_measure.startswith("Stagnation"):
            return self.stagnation_pressure_bar(standoff_mm)
        if self.cleaning_measure.startswith("Mean"):
            return self.mean_impact_pressure_bar(standoff_mm)
        return self.wall_shear_kpa(standoff_mm)

    # ---- Backward analysis: what does cleaning the fouling REQUIRE? ---------
    # The real operating target is the impact at the hull. Work back from a
    # required intensity (the fouling's removal threshold) through the jet
    # decay to the subsea/topside pressure the operator must regulate to.
    def _decay_factor(self, standoff_mm: float | None = None) -> float:
        x = (self.standoff_mm if standoff_mm is None else standoff_mm)
        x = x / max(math.cos(math.radians(self.nozzle_cant_deg)), 1e-6)
        core = self.core_length_mm
        return 1.0 if x <= core else self.decay_K * self.nozzle_exit_mm / x

    def required_subsea_pressure_bar(self, intensity: float | None = None,
                                     standoff_mm: float | None = None) -> float:
        """
        Subsea (nozzle) pressure needed so the chosen impact measure reaches
        `intensity` at the hull (default: the removal threshold). Inverts
        measure → v_c → v_exit → ½ρ(v_exit/Cd)². Mean-pressure can't be
        cleanly inverted (footprint-dependent); approximated via stagnation.
        """
        target = self.removal_pressure_bar if intensity is None else intensity
        rho = self.water_density
        if self.cleaning_measure.startswith("Wall"):
            v_c = math.sqrt(target * 1000.0 / (self.skin_friction_cf * 0.5 * rho))
        else:  # stagnation (and mean, approximated): ½ρv_c² = target bar
            v_c = math.sqrt(target * 1e5 / (0.5 * rho))
        decay = max(self._decay_factor(standoff_mm), 1e-6)
        v_exit = v_c / decay
        return 0.5 * rho * (v_exit / max(self.nozzle_cd, 1e-6)) ** 2 / 1e5

    def required_topside_pressure_bar(self, intensity: float | None = None,
                                      standoff_mm: float | None = None) -> float:
        """Topside the operator must regulate to = required subsea ÷ ratio."""
        return self.required_subsea_pressure_bar(intensity, standoff_mm) \
            / max(self.pressure_transmission_ratio, 1e-6)

    @property
    def max_subsea_pressure_bar(self) -> float:
        """
        Highest subsea pressure reachable at the CURRENT nozzle count, limited
        by whichever binds first: the pumps (all flow through n nozzles) or the
        hose ceiling. Fewer nozzles raises the pump-limited value.
        """
        # Pump-limited: full pump flow through n nozzles.
        q = self.pump_flow_cap_lpm / 1000.0 / 60.0
        v_pump = q / max(self.nozzle_exit_area_m2 * self.n_nozzles_total, 1e-12)
        p_pump = 0.5 * self.water_density * (v_pump / max(self.nozzle_cd, 1e-6)) ** 2 / 1e5
        # Hose-limited: topside ≤ ceiling → subsea ≤ ceiling × ratio.
        p_hose = self.hose_pressure_ceiling_bar * self.pressure_transmission_ratio
        return min(p_pump, p_hose)

    def footprint_dia(self, standoff_mm: float | None = None) -> float:
        if self.footprint_mode == "Manual override":
            return float(self.footprint_dia_mm_override)
        if self.footprint_mode.startswith("Linear with pressure"):
            return 60.0 + (self.pressure_bar - 50.0) / (600.0 - 50.0) * 20.0
        # Physical jet: a free jet of exit diameter d0 spreads as a cone of
        # half-angle θ over the standoff L, so its impingement footprint is
        #   d_fp = d0 + 2·L·tan(θ).
        # Both the nozzle exit diameter and the distance to the hull drive it.
        x = self.standoff_mm if standoff_mm is None else standoff_mm
        return (self.nozzle_exit_mm
                + 2.0 * x
                * math.tan(math.radians(self.jet_spread_deg)))

    def impact_radius_mm(self) -> float:
        return max(
            0.0,
            self.nozzle_radius_mm
            - self.standoff_mm * math.tan(math.radians(self.nozzle_cant_deg)),
        )

    @property
    def resolved_cell_mm(self) -> float:
        """
        Hull-grid cell size actually used by the sim. In auto mode it is
        derived from the footprint (≈ footprint / 4) so the disk is always
        well-resolved (≥ ~4 cells across) and the result no longer swings
        with an arbitrary grid choice. Clamped to a sane range.
        """
        if self.auto_grid:
            # Lower clamp 1.0 mm: finer than that re-introduces a little
            # wobble (gate-radius vs hit-spacing) AND blows up runtime under
            # the step cap, with no accuracy gain. Upper clamp 5 mm.
            return float(min(max(self.footprint_dia() / 4.0, 1.0), 5.0))
        return float(self.cell_size_mm)


def scenario_key(s: Scenario) -> tuple:
    """Hashable key for caching — only geometry-affecting params."""
    return (s.array_width_mm, s.n_row1, s.n_row2, s.disc_pitch_mm,
            s.row_pitch_mm, s.yaw_deg, s.disc_diameter_mm, s.n_nozzles,
            s.nozzle_radius_mm, s.nozzle_cant_deg, s.standoff_mm,
            s.counter_rotate, s.pressure_bar, s.footprint_mode,
            s.footprint_dia_mm_override, s.nozzle_exit_mm, s.jet_spread_deg,
            s.resolved_cell_mm, s.total_flow_lpm, s.pump_flow_cap_lpm,
            s.pressure_transmission_ratio, s.hose_pressure_ceiling_bar)


def scenario_full_key(s: Scenario) -> tuple:
    """Full hashable key — all params that affect simulation output."""
    return (*scenario_key(s), s.rpm, s.rov_speed_kn, s.pressure_profile,
            s.sim_length_mm, s.clean_threshold, s.steady_state_only,
            s.removal_pressure_bar, s.min_passes, s.core_factor,
            s.cleaning_measure, s.nozzle_cd, s.decay_K, s.jet_half_angle_deg,
            s.skin_friction_cf, s.water_density)


# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------
@dataclass
class Disc:
    cx_mm: float
    cy_mm: float
    direction: int


def disc_layout(s: Scenario) -> list[Disc]:
    discs: list[Disc] = []
    p = s.disc_pitch_mm
    for i in range(s.n_row1):
        x = (i - (s.n_row1 - 1) / 2.0) * p
        d = +1 if (i % 2 == 0 or not s.counter_rotate) else -1
        discs.append(Disc(x, 0.0, d))
    needs_half_shift = (s.n_row1 % 2) == (s.n_row2 % 2)
    shift = (p / 2.0) if needs_half_shift else 0.0
    for j in range(s.n_row2):
        x = (j - (s.n_row2 - 1) / 2.0) * p + shift
        d = -1 if (j % 2 == 0 or not s.counter_rotate) else +1
        discs.append(Disc(x, float(s.row_pitch_mm), d))
    return discs


def rotate_point(x: float, y: float, cos_t: float, sin_t: float,
                 cx: float, cy: float) -> tuple[float, float]:
    dx, dy = x - cx, y - cy
    return cx + dx * cos_t - dy * sin_t, cy + dx * sin_t + dy * cos_t


def compute_rotated_discs(s: Scenario) -> list[tuple[float, float, int]]:
    """Disc centres after array yaw, in the array-local frame."""
    discs = disc_layout(s)
    yaw_rad = math.radians(s.yaw_deg)
    cos_t, sin_t = math.cos(yaw_rad), math.sin(yaw_rad)
    cx_a, cy_a = 0.0, s.row_pitch_mm / 2.0
    out = []
    for d in discs:
        rx, ry = rotate_point(d.cx_mm, d.cy_mm, cos_t, sin_t, cx_a, cy_a)
        out.append((rx, ry, d.direction))
    return out


# -----------------------------------------------------------------------------
# Footprint stencil
# -----------------------------------------------------------------------------
def footprint_stencil(s: Scenario) -> np.ndarray:
    """
    Weight per cell for one nozzle deposit. The UNIFORM profile deposits a
    weight of 1.0 into every cell inside the footprint, so a pass deposits
    P·dt into each covered cell — total energy ≈ P·dt·footprint_area.

    The GAUSSIAN profile models the same jet with a peaked (centre-heavy)
    intensity rather than a flat one. It is normalised to deposit the SAME
    total energy as the uniform disc (it redistributes energy toward the
    centre, raising the peak, rather than discarding it). Without this
    normalisation a Gaussian footprint silently deposits only ~half the
    uniform energy, collapsing the bar·s KPIs.
    """
    r_mm = s.footprint_dia() / 2.0
    r_cells = r_mm / s.resolved_cell_mm
    half = int(math.ceil(r_cells)) + 1
    yy, xx = np.ogrid[-half:half + 1, -half:half + 1]
    r2 = (xx ** 2 + yy ** 2).astype(np.float32)
    uniform = (r2 <= r_cells ** 2).astype(np.float32)
    if s.pressure_profile.startswith("Uniform"):
        return uniform
    sigma = r_cells / 2.0
    stencil = np.exp(-r2 / (2 * sigma ** 2)).astype(np.float32)
    stencil[r2 > (r_cells * 1.5) ** 2] = 0.0
    # Normalise so the Gaussian deposits the same TOTAL energy as the
    # uniform disc of this footprint (same P·dt·area; centre-weighted).
    total = float(stencil.sum())
    if total > 0.0:
        stencil *= float(uniform.sum()) / total
    return stencil


def footprint_profile_peaknorm(s: Scenario) -> np.ndarray:
    """
    Footprint intensity profile normalised so its PEAK = 1.0. Used to map the
    jet stagnation pressure ACROSS the footprint: a cell at the footprint
    centre receives the full delivered pressure (profile = 1) and the edges a
    fraction. This is the spatial basis for the delivered-pressure heatmap and
    the per-cell intensity gate (distinct from the energy-conserving stencil
    above, which is for the bar·s dose).
    """
    r_mm = s.footprint_dia() / 2.0
    r_cells = r_mm / s.resolved_cell_mm
    half = int(math.ceil(r_cells)) + 1
    yy, xx = np.ogrid[-half:half + 1, -half:half + 1]
    r2 = (xx ** 2 + yy ** 2).astype(np.float32)
    if s.pressure_profile.startswith("Uniform"):
        return (r2 <= r_cells ** 2).astype(np.float32)
    sigma = r_cells / 2.0
    prof = np.exp(-r2 / (2 * sigma ** 2)).astype(np.float32)
    prof[r2 > (r_cells * 1.5) ** 2] = 0.0
    pk = float(prof.max())
    if pk > 0.0:
        prof /= pk        # peak = 1.0
    return prof


# -----------------------------------------------------------------------------
# Hull simulation helpers
# -----------------------------------------------------------------------------
def hull_section_perimeter_mm(beam_mm: float, draft_mm: float,
                              shape_key: str) -> float:
    """
    Approximate perimeter (mm) of the midship cross-section's bottom
    (i.e. everything from port waterline, down around the keel, up to
    starboard waterline, excluding the two vertical side segments).

    This is what the cleaning robot traverses across the bottom of the
    hull on a single sweep at midship.
    """
    shape = HULL_SHAPES[shape_key]
    r = shape["r_rel"] * beam_mm
    if shape["deadrise"] <= 0.0:
        # Rectangular bottom with two quarter-circle bilges.
        flat = max(beam_mm - 2 * r, 0.0)
        # two quarter arcs = half circumference
        arcs = math.pi * r
        return flat + arcs
    # V-bottom with bilge radius r and deadrise α
    alpha = math.radians(shape["deadrise"])
    half_flat = max(beam_mm / 2 - r, 0.0)
    v_leg = half_flat / max(math.cos(alpha), 1e-6)
    arcs = math.pi * r  # two quarter bilges
    return 2 * v_leg + arcs


def hull_wetted_areas(loa_m: float, beam_m: float, draft_m: float,
                      shape_key: str) -> dict:
    """
    First-order wetted-surface-area model for cleaning planning.

    Returns a dict with ``side_port``, ``side_starboard``, ``bottom``
    and ``total`` in m².  LOA, beam and draft are in metres.

    Assumptions:
      - Side area per side ≈ LOA × draft × k_side
      - Bottom area ≈ LOA × midship_section_perimeter
      - End-effects (bow/stern curvature) are absorbed into a
        longitudinal correction factor Cl = 0.96 (fine), 1.00 (typical),
        1.02 (full) — small tweak that captures plumb vs raked bow.
    """
    shape = HULL_SHAPES[shape_key]
    C_l = {"full": 1.02, "typical": 1.00, "fine": 0.96}[shape_key]
    side_area = loa_m * draft_m * shape["k_side"] * C_l
    perim_m = hull_section_perimeter_mm(beam_m * 1000, draft_m * 1000,
                                        shape_key) / 1000.0
    bottom_area = loa_m * perim_m * C_l
    return {
        "side_port": side_area,
        "side_starboard": side_area,
        "bottom": bottom_area,
        "total": 2 * side_area + bottom_area,
        "section_perim_m": perim_m,
        "C_l": C_l,
    }
