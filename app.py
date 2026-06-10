"""
FWU Coverage Simulator — performance-optimised motion viz.

Key changes vs the previous version:
  - plot_motion is fully vectorised. Trails are computed with NumPy
    broadcasting instead of Python loops over time samples × discs × nozzles.
  - Trails are drawn as a single LineCollection instead of per-segment plot()
    calls (~50-100x faster for long trails).
  - Current footprints are drawn as a single PatchCollection.
  - The cumulative cleaning underlay is gated behind a button and cached in
    session_state, so scrubbing the time slider doesn't re-run the pressure
    simulation each time.
  - Top-down and side schematics are cached via st.cache_data so they only
    re-render when the scenario geometry changes, not on every slider move.
  - Trail length cap raised to 20 revolutions.
  - Motion animation uses a "Prepare → Play" pattern: all frames are pre-
    rendered to PNGs into st.session_state, then played back via st.image()
    for instant, flicker-free playback. Stop button interrupts either phase.
  - Hull-frame axis limits are fixed over the full traversal so the camera
    doesn't shift from frame to frame during playback.
  - A real-time vs wall-clock scale label tells you how much slower the
    animation is compared to reality (rotation is ~800 RPM).

Run with:
    streamlit run app.py
    # or, if the streamlit launcher is not on PATH:
    python -m streamlit run app.py
"""

from __future__ import annotations

import io
import math
import time
from dataclasses import dataclass, field, asdict
from copy import deepcopy

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.collections import LineCollection, PatchCollection
from PIL import Image

# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FWU Coverage Simulator",
    layout="wide",
    page_icon="🛠️",
)

st.title("FWU Coverage Simulator")
st.caption(
    "Simulates integrated jet pressure exposure on the hull (1x1 cm grid). "
    "Adjust geometry, rotation, pressure and traverse speed, then run."
)

KNOTS_TO_MPS = 0.514444
# Hull-grid resolution is now per-scenario (Scenario.cell_size_mm). This
# constant is kept only as a legacy default; the sim/render paths read the
# scenario value so the grid can be refined to 1–2 mm for small footprints.
CELL_SIZE_MM = 10.0


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
    disc_pitch_mm: int = 380
    row_pitch_mm: int = 320
    yaw_deg: float = 0.0

    # Disc & nozzles
    disc_diameter_mm: int = 360
    n_nozzles: int = 3
    nozzle_radius_mm: int = 140
    nozzle_cant_deg: int = 10
    standoff_mm: int = 18
    counter_rotate: bool = True

    # Operating point
    rpm: int = 600
    rov_speed_kn: float = 1.5
    pressure_bar: int = 200
    nozzle_exit_mm: float = 1.5

    # Jet footprint model
    footprint_mode: str = "Physical jet (exit dia + standoff)"
    footprint_dia_mm_override: int = 70
    jet_spread_deg: float = 8.0     # free-jet half-cone spread angle
    pressure_profile: str = "Gaussian (peak at centre)"

    # Hull strip & threshold
    sim_length_mm: int = 2000
    clean_threshold: float = 0.5    # bar·s — dose heatmap threshold only
    cell_size_mm: float = 5.0       # hull-grid resolution

    # Cleaning criterion: an intensity GATE (jet stagnation pressure at the
    # hull must exceed a removal threshold for the fouling type) plus a DOSE
    # gate (a minimum number of nozzle passes over the cell).
    removal_pressure_bar: float = 30.0   # P_stag at hull needed to lift fouling
    min_passes: int = 2                  # passes required once above the gate
    jet_core_factor: float = 6.0         # potential-core length = factor · exit_dia

    # Post-processing
    steady_state_only: bool = True

    def stagnation_pressure_bar(self) -> float:
        """
        Jet stagnation pressure delivered at the hull. A submerged free jet
        holds its nozzle pressure within the potential core (length ≈
        core_factor · exit_dia), then its centreline dynamic pressure decays
        as (core_len / standoff)² beyond it. This is the physical 'cleaning
        intensity' — what actually lifts fouling — and it is what makes
        standoff and exit diameter the dominant power levers.
        """
        d0 = max(self.nozzle_exit_mm, 1e-6)
        core_len = self.jet_core_factor * d0
        # Jet path length to the hull (standoff along the canted axis).
        L = self.standoff_mm / max(math.cos(math.radians(self.nozzle_cant_deg)),
                                   1e-6)
        if L <= core_len:
            return float(self.pressure_bar)
        return float(self.pressure_bar) * (core_len / L) ** 2

    def footprint_dia(self) -> float:
        if self.footprint_mode == "Manual override":
            return float(self.footprint_dia_mm_override)
        if self.footprint_mode.startswith("Linear with pressure"):
            return 60.0 + (self.pressure_bar - 50.0) / (600.0 - 50.0) * 20.0
        # Physical jet: a free jet of exit diameter d0 spreads as a cone of
        # half-angle θ over the standoff L, so its impingement footprint is
        #   d_fp = d0 + 2·L·tan(θ).
        # Both the nozzle exit diameter and the distance to the hull drive it.
        return (self.nozzle_exit_mm
                + 2.0 * self.standoff_mm
                * math.tan(math.radians(self.jet_spread_deg)))

    def impact_radius_mm(self) -> float:
        return max(
            0.0,
            self.nozzle_radius_mm
            - self.standoff_mm * math.tan(math.radians(self.nozzle_cant_deg)),
        )


def scenario_key(s: Scenario) -> tuple:
    """Hashable key for caching — only geometry-affecting params."""
    return (s.array_width_mm, s.n_row1, s.n_row2, s.disc_pitch_mm,
            s.row_pitch_mm, s.yaw_deg, s.disc_diameter_mm, s.n_nozzles,
            s.nozzle_radius_mm, s.nozzle_cant_deg, s.standoff_mm,
            s.counter_rotate, s.pressure_bar, s.footprint_mode,
            s.footprint_dia_mm_override, s.nozzle_exit_mm, s.jet_spread_deg,
            s.cell_size_mm)


def scenario_full_key(s: Scenario) -> tuple:
    """Full hashable key — all params that affect simulation output."""
    return (*scenario_key(s), s.rpm, s.rov_speed_kn, s.pressure_profile,
            s.sim_length_mm, s.clean_threshold, s.steady_state_only,
            s.removal_pressure_bar, s.min_passes, s.jet_core_factor)


# -----------------------------------------------------------------------------
# Sidebar parameter builder
# -----------------------------------------------------------------------------
def scenario_controls(prefix: str, defaults: Scenario, container) -> Scenario:
    s = deepcopy(defaults)
    k = lambda name: f"{prefix}_{name}"  # noqa: E731

    container.subheader("Array geometry")
    s.array_width_mm = container.slider(
        "Array width (mm)", 1000, 2400, s.array_width_mm, step=10,
        key=k("array_width_mm"),
        help="Total width of the array enclosure across the ROV.",
    )
    s.n_row1 = container.slider("Discs in front row", 2, 6, s.n_row1, key=k("n_row1"))
    s.n_row2 = container.slider("Discs in back row", 2, 6, s.n_row2, key=k("n_row2"))
    s.disc_pitch_mm = container.slider(
        "Disc pitch (mm)", 200, 700, s.disc_pitch_mm, step=5,
        key=k("disc_pitch_mm"),
        help="Centre-to-centre distance between adjacent discs in the SAME row. "
             "Both rows share this pitch; the back row is automatically "
             "interlocked into the front-row gaps.",
    )
    s.row_pitch_mm = container.slider(
        "Row pitch (mm)", 100, 700, s.row_pitch_mm, step=5,
        key=k("row_pitch_mm"),
        help="Centre-to-centre distance between the two disc rows (along ROV travel).",
    )
    s.yaw_deg = container.slider(
        "Array yaw vs travel direction (°)",
        -45.0, 45.0, s.yaw_deg, step=1.0,
        key=k("yaw_deg"),
        help="Rotation of the disc array about its centroid relative to the "
             "ROV's travel direction. 0° = array perpendicular to travel.",
    )

    container.subheader("Disc & nozzles")
    s.disc_diameter_mm = container.slider(
        "Disc diameter (mm)", 200, 500, s.disc_diameter_mm, step=5, key=k("disc_diameter_mm"))
    s.n_nozzles = container.slider(
        "Nozzles per disc", 1, 6, s.n_nozzles, key=k("n_nozzles"))
    s.nozzle_radius_mm = container.slider(
        "Nozzle radius from disc centre (mm)", 20, 200, s.nozzle_radius_mm, step=1,
        key=k("nozzle_radius_mm"))
    s.nozzle_cant_deg = container.slider(
        "Nozzle cant towards centre (°)", 0, 30, s.nozzle_cant_deg, key=k("nozzle_cant_deg"))
    s.standoff_mm = container.slider(
        "Nozzle standoff to hull (mm)", 5, 60, s.standoff_mm, key=k("standoff_mm"))
    s.counter_rotate = container.checkbox(
        "Adjacent discs counter-rotate", value=s.counter_rotate, key=k("counter_rotate"))

    container.subheader("Operating point")
    s.rpm = container.slider(
        "Disc rotation (RPM)", 50, 1500, s.rpm, step=10, key=k("rpm"))
    s.rov_speed_kn = container.slider(
        "ROV traverse speed (knots)", 0.1, 4.0, s.rov_speed_kn, step=0.1,
        key=k("rov_speed_kn"))
    s.pressure_bar = container.slider(
        "Jet pressure at nozzle (bar)", 50, 600, s.pressure_bar, step=10,
        key=k("pressure_bar"))
    s.nozzle_exit_mm = container.slider(
        "Nozzle exit diameter (mm)", 0.5, 5.0, s.nozzle_exit_mm, step=0.1,
        key=k("nozzle_exit_mm"))

    container.subheader("Jet footprint model")
    _grid_opts = [10.0, 5.0, 2.0, 1.0]
    _grid_idx = (_grid_opts.index(s.cell_size_mm)
                 if s.cell_size_mm in _grid_opts else 1)
    s.cell_size_mm = container.selectbox(
        "Hull grid resolution (mm/cell)", _grid_opts, index=_grid_idx,
        format_func=lambda v: f"{v:g} mm"
        + {10.0: "  (fast)", 1.0: "  (true footprint, slower)"}.get(v, ""),
        key=k("cell_size_mm"),
        help="Cell size of the hull grid. Finer grids resolve small "
             "(sub-cm) physical footprints but cost ~1/cell² in memory "
             "and up to ~1/cell⁴ in compute. The deposit loop is "
             "vectorised, so 1–2 mm stays responsive for the small "
             "physical-jet footprint; the legacy large footprint is "
             "heavier at 1 mm.")
    _fp_modes = [
        "Physical jet (exit dia + standoff)",
        "Linear with pressure (60->80 mm)",
        "Manual override",
    ]
    _fp_index = (_fp_modes.index(s.footprint_mode)
                 if s.footprint_mode in _fp_modes else 0)
    s.footprint_mode = container.radio(
        "Footprint diameter on hull",
        _fp_modes,
        index=_fp_index,
        key=k("footprint_mode"),
        help="Physical jet: the impact zone grows from the nozzle exit "
             "diameter as the jet spreads over the standoff distance to "
             "the hull — d_fp = exit_dia + 2·standoff·tan(spread). "
             "Linear: a legacy curve tied only to pressure. "
             "Manual: a fixed diameter.",
    )
    if s.footprint_mode == "Manual override":
        s.footprint_dia_mm_override = container.slider(
            "Footprint diameter (mm)", 20, 120, s.footprint_dia_mm_override,
            key=k("footprint_dia_mm_override"))
    elif s.footprint_mode.startswith("Physical jet"):
        s.jet_spread_deg = container.slider(
            "Jet spread half-angle (°)", 0.0, 20.0, float(s.jet_spread_deg),
            step=0.5, key=k("jet_spread_deg"),
            help="Half-cone angle of the spreading free jet. Submerged "
                 "water jets typically spread at 5–15°; calibrate against "
                 "a measured spot size if you have one.")
        _spread_mm = (s.footprint_dia() - s.nozzle_exit_mm) / 2.0
        container.caption(
            f"Footprint diameter on hull: **{s.footprint_dia():.1f} mm** "
            f"(= {s.nozzle_exit_mm:.1f} mm exit + 2 × {_spread_mm:.1f} mm "
            f"spread over {s.standoff_mm} mm standoff)")
        if s.footprint_dia() < s.cell_size_mm:
            container.warning(
                f"Footprint ({s.footprint_dia():.1f} mm) is below the "
                f"{s.cell_size_mm:g} mm hull-grid resolution, so the impact "
                "sim deposits onto a single cell. The sensitivity chart "
                "above still reflects the true sub-mm footprint; lower the "
                "grid resolution, or raise the spread angle / standoff, to "
                "resolve it on the grid.")
    else:
        container.caption(f"Footprint diameter on hull: **{s.footprint_dia():.1f} mm**")
    s.pressure_profile = container.radio(
        "Pressure distribution within footprint",
        ["Uniform", "Gaussian (peak at centre)"],
        index=0 if s.pressure_profile.startswith("Uniform") else 1,
        key=k("pressure_profile"),
    )

    container.subheader("Cleaning criterion")
    container.caption(
        "Cleaning needs enough **intensity** (jet pressure delivered at the "
        "hull) AND enough **dose** (passes). Intensity is the gate: below it, "
        "no amount of dwell removes fouling.")
    _foul_presets = {
        "Soft biofilm / slime": 20.0,
        "Light weed / early growth": 60.0,
        "Hard calcareous / barnacle": 150.0,
        "Custom": None,
    }
    _foul_choice = container.selectbox(
        "Fouling type (sets removal pressure)",
        list(_foul_presets.keys()),
        index=0,
        key=k("foul_preset"),
        help="Minimum jet stagnation pressure at the hull needed to lift "
             "this fouling. Calibrate against a known field result — set it "
             "to the delivered pressure of a run you KNOW cleans your "
             "fouling (shown live below).")
    if _foul_presets[_foul_choice] is not None:
        s.removal_pressure_bar = _foul_presets[_foul_choice]
    else:
        s.removal_pressure_bar = container.slider(
            "Removal pressure at hull (bar)", 1.0, 400.0,
            float(s.removal_pressure_bar), step=1.0,
            key=k("removal_pressure_bar"))
    # Live readout of delivered (stagnation) pressure for the current jet.
    _pstag = s.stagnation_pressure_bar()
    _gate = "✅ above" if _pstag >= s.removal_pressure_bar else "❌ BELOW"
    container.caption(
        f"Jet delivers **{_pstag:.0f} bar** at the hull "
        f"({s.pressure_bar} bar at nozzle, decayed over "
        f"{s.standoff_mm} mm standoff with {s.nozzle_exit_mm:.1f} mm exit) "
        f"— {_gate} the {s.removal_pressure_bar:.0f} bar removal threshold.")
    s.min_passes = container.slider(
        "Minimum passes to clean", 1, 10, int(s.min_passes), key=k("min_passes"),
        help="Once the intensity gate is cleared, a cell must be struck at "
             "least this many times to count as cleaned.")

    container.subheader("Hull strip & KPIs")
    s.sim_length_mm = container.slider(
        "Hull strip length to simulate (mm)", 500, 10000, s.sim_length_mm, step=100,
        key=k("sim_length_mm"))
    s.clean_threshold = container.slider(
        "Dose heatmap threshold (bar*s)", 0.05, 50.0, float(s.clean_threshold), step=0.05,
        key=k("clean_threshold"),
        help="Only affects the secondary bar·s 'dose' heatmap, not the "
             "cleaned-area KPI (which uses the intensity+passes criterion "
             "above). Lower this to match the now-smaller bar·s values.")
    s.steady_state_only = container.checkbox(
        "Report KPIs on steady-state core only",
        value=s.steady_state_only,
        key=k("steady_state_only"),
        help="Excludes the array's entry and exit transients from the KPIs and "
             "the bottom heatmap. Transient regions stay visible in the main "
             "heatmap but are dimmed.")
    return s


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
# Top-down schematic (cached — regenerated only when geometry changes)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def plot_topdown_cached(key: tuple, _scen_bytes: bytes,
                        title: str = "Top-down view") -> plt.Figure:
    import pickle
    s: Scenario = pickle.loads(_scen_bytes)
    return _plot_topdown_impl(s, title)


def _plot_topdown_impl(s: Scenario, title: str) -> plt.Figure:
    discs = disc_layout(s)
    cx, cy = 0.0, s.row_pitch_mm / 2.0
    yaw_rad = math.radians(s.yaw_deg)
    cos_t, sin_t = math.cos(yaw_rad), math.sin(yaw_rad)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    xs, ys = [], []
    for d in discs:
        rx, ry = rotate_point(d.cx_mm, d.cy_mm, cos_t, sin_t, cx, cy)
        xs.append(rx); ys.append(ry)

    half_w = max(abs(min(xs)), abs(max(xs))) + s.disc_diameter_mm / 2 + 40
    y_top = min(ys) - s.disc_diameter_mm / 2 - 40
    y_bot = max(ys) + s.disc_diameter_mm / 2 + 40

    corners = [(-half_w, y_top), (half_w, y_top),
               (half_w, y_bot), (-half_w, y_bot)]
    frame_poly = mpatches.Polygon(
        corners, closed=True, fill=False, edgecolor="#444", linewidth=1.5)
    ax.add_patch(frame_poly)

    impact_r = s.impact_radius_mm()
    fp_d = s.footprint_dia()
    for d, rx, ry in zip(discs, xs, ys):
        ax.add_patch(mpatches.Circle((rx, ry), s.disc_diameter_mm / 2,
                                     fill=False, edgecolor="#1f77b4", linewidth=1.2))
        ax.add_patch(mpatches.Circle((rx, ry), impact_r,
                                     fill=False, linestyle="--",
                                     edgecolor="#888", linewidth=0.6))
        for kn in range(s.n_nozzles):
            theta = 2 * math.pi * kn / s.n_nozzles
            nx0 = d.cx_mm + impact_r * math.cos(theta)
            ny0 = d.cy_mm + impact_r * math.sin(theta)
            nx, ny = rotate_point(nx0, ny0, cos_t, sin_t, cx, cy)
            ax.add_patch(mpatches.Circle((nx, ny), fp_d / 2,
                                         facecolor="#ff7f0e", alpha=0.30,
                                         edgecolor="none"))

    ax.annotate("ROV travel", xy=(half_w + 40, y_bot), xytext=(half_w + 40, y_top),
                arrowprops=dict(arrowstyle="->", color="#555"),
                ha="center", va="center", fontsize=9, color="#555")

    ax.set_xlim(-half_w - 120, half_w + 180)
    ax.set_ylim(y_top - 80, y_bot + 80)
    ax.set_aspect("equal")
    ax.set_xlabel("Across-track (mm)")
    ax.set_ylabel("Along-track (mm)")
    ax.set_title(f"{title} — yaw {s.yaw_deg:+.0f}°")
    ax.grid(True, alpha=0.3)
    return fig


def plot_topdown(s: Scenario, title: str = "Top-down view") -> plt.Figure:
    import pickle
    return plot_topdown_cached(scenario_key(s), pickle.dumps(s), title)


# -----------------------------------------------------------------------------
# Side cross-section (cached)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def plot_side_cached(key: tuple, _scen_bytes: bytes) -> plt.Figure:
    import pickle
    return _plot_side_impl(pickle.loads(_scen_bytes))


def _plot_side_impl(s: Scenario) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.axhline(0, color="#555", lw=1.5)
    ax.text(s.impact_radius_mm() * 1.1, -3, "hull", color="#555", fontsize=8)
    disc_h = 60
    ax.add_patch(mpatches.Rectangle(
        (-s.disc_diameter_mm / 2, s.standoff_mm),
        s.disc_diameter_mm, disc_h,
        facecolor="#cfe3f5", edgecolor="#1f77b4",
    ))
    ax.text(0, s.standoff_mm + disc_h / 2, "disc", ha="center", va="center", fontsize=9)
    fp_d = s.footprint_dia()
    d0 = s.nozzle_exit_mm
    for sign in (-1, +1):
        nx = sign * s.nozzle_radius_mm
        end_x = nx - sign * s.standoff_mm * math.tan(math.radians(s.nozzle_cant_deg))
        # Spreading jet cone: starts at the nozzle exit diameter d0 at the
        # disc face and widens to the footprint diameter at the hull. The
        # cone is built around the (canted) jet axis from exit to footprint.
        cone = mpatches.Polygon(
            [(nx - d0 / 2, s.standoff_mm), (nx + d0 / 2, s.standoff_mm),
             (end_x + fp_d / 2, 0), (end_x - fp_d / 2, 0)],
            closed=True, facecolor="#ff7f0e", alpha=0.22, edgecolor="none",
        )
        ax.add_patch(cone)
        # Jet axis (nozzle exit centre -> footprint centre).
        ax.plot([nx, end_x], [s.standoff_mm, 0], color="#ff7f0e", lw=1.5)
        # Footprint on the hull.
        ax.add_patch(mpatches.Ellipse(
            (end_x, 0), fp_d, 4,
            facecolor="#ff7f0e", alpha=0.55, edgecolor="none",
        ))
    # Label the footprint diameter under the right-hand jet.
    fp_cx = s.nozzle_radius_mm - s.standoff_mm * math.tan(math.radians(s.nozzle_cant_deg))
    ax.annotate("", xy=(fp_cx - fp_d / 2, -8), xytext=(fp_cx + fp_d / 2, -8),
                arrowprops=dict(arrowstyle="<->", color="#ff7f0e", lw=1.0))
    ax.text(fp_cx, -12, f"footprint {fp_d:.0f} mm",
            color="#d9660a", ha="center", va="top", fontsize=8)
    ax.annotate("", xy=(-s.disc_diameter_mm / 2 - 20, 0),
                xytext=(-s.disc_diameter_mm / 2 - 20, s.standoff_mm),
                arrowprops=dict(arrowstyle="<->", color="#333"))
    ax.text(-s.disc_diameter_mm / 2 - 25, s.standoff_mm / 2,
            f"{s.standoff_mm} mm", ha="right", va="center", fontsize=8)
    ax.text(s.nozzle_radius_mm + 20, s.standoff_mm * 0.6,
            f"{s.nozzle_cant_deg}°", color="#ff7f0e", fontsize=9)
    ax.set_xlim(-s.disc_diameter_mm / 2 - 80, s.disc_diameter_mm / 2 + 80)
    ax.set_ylim(-26, s.standoff_mm + disc_h + 20)
    ax.set_aspect("equal")
    ax.set_xlabel("Across disc (mm)")
    ax.set_ylabel("Height (mm)")
    ax.set_title("Side cross-section — spreading jet")
    ax.grid(True, alpha=0.3)
    return fig


def plot_side(s: Scenario) -> plt.Figure:
    import pickle
    return plot_side_cached(scenario_key(s), pickle.dumps(s))


# -----------------------------------------------------------------------------
# Zoomed spray profile (single nozzle, close-up of exit -> hull)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def plot_spray_profile_cached(key: tuple, _scen_bytes: bytes) -> plt.Figure:
    import pickle
    return _plot_spray_profile_impl(pickle.loads(_scen_bytes))


def _plot_spray_profile_impl(s: Scenario) -> plt.Figure:
    """
    Close-up of a SINGLE nozzle jet from the exit orifice down to the hull,
    at the true mm scale. The main side view is dominated by the ~360 mm
    disc, so the mm-scale jet (exit dia, spread, footprint) is invisible
    there; this panel zooms in on just the jet column.
    """
    fp_d = s.footprint_dia()
    d0 = s.nozzle_exit_mm
    L = float(s.standoff_mm)
    cant = math.radians(s.nozzle_cant_deg)
    dx_hull = L * math.tan(cant)           # horizontal cant shift over standoff

    fig, ax = plt.subplots(figsize=(5.5, 3.4))

    # Hull line.
    ax.axhline(0, color="#555", lw=2.0, zorder=1)
    # Nozzle body block at the top (exit at y = L).
    body_h = max(L * 0.35, 4.0)
    ax.add_patch(mpatches.Rectangle(
        (-max(d0, 2.0) * 2.0, L), max(d0, 2.0) * 4.0, body_h,
        facecolor="#cfe3f5", edgecolor="#1f77b4", zorder=3))
    ax.text(0, L + body_h / 2, "nozzle", ha="center", va="center", fontsize=8)

    # Spreading jet cone: exit (width d0) at top, footprint (width fp_d) at hull,
    # centre line canted by dx_hull.
    cone = mpatches.Polygon(
        [(-d0 / 2, L), (d0 / 2, L),
         (dx_hull + fp_d / 2, 0), (dx_hull - fp_d / 2, 0)],
        closed=True, facecolor="#ff7f0e", alpha=0.25, edgecolor="#ff7f0e",
        linewidth=1.0, zorder=2)
    ax.add_patch(cone)
    ax.plot([0, dx_hull], [L, 0], color="#ff7f0e", lw=1.2, ls="--", zorder=4)

    # Footprint mark on the hull.
    ax.add_patch(mpatches.Ellipse(
        (dx_hull, 0), fp_d, max(fp_d * 0.12, 1.0),
        facecolor="#ff7f0e", alpha=0.7, edgecolor="none", zorder=4))

    # Dimension: exit diameter (top).
    ax.annotate("", xy=(-d0 / 2, L + body_h + 2), xytext=(d0 / 2, L + body_h + 2),
                arrowprops=dict(arrowstyle="<->", color="#1f77b4", lw=1.0))
    ax.text(0, L + body_h + 3, f"exit {d0:.1f} mm",
            color="#1f77b4", ha="center", va="bottom", fontsize=8)

    # Dimension: standoff (left).
    x_so = -fp_d / 2 - max(fp_d * 0.4, 4.0)
    ax.annotate("", xy=(x_so, 0), xytext=(x_so, L),
                arrowprops=dict(arrowstyle="<->", color="#333", lw=1.0))
    ax.text(x_so - 1, L / 2, f"standoff\n{L:.0f} mm",
            color="#333", ha="right", va="center", fontsize=8)

    # Dimension: footprint diameter (bottom).
    ax.annotate("", xy=(dx_hull - fp_d / 2, -max(fp_d * 0.5, 4.0)),
                xytext=(dx_hull + fp_d / 2, -max(fp_d * 0.5, 4.0)),
                arrowprops=dict(arrowstyle="<->", color="#d9660a", lw=1.2))
    ax.text(dx_hull, -max(fp_d * 0.5, 4.0) - 1, f"footprint {fp_d:.1f} mm",
            color="#d9660a", ha="center", va="top", fontsize=8, fontweight="bold")

    # Cant angle label near the exit.
    if s.nozzle_cant_deg > 0:
        ax.text(d0 / 2 + 1, L * 0.75, f"{s.nozzle_cant_deg}° cant",
                color="#ff7f0e", ha="left", va="center", fontsize=8)

    span = max(fp_d, d0, dx_hull * 2) * 1.6 + 8
    ax.set_xlim(-span, span)
    ax.set_ylim(-max(fp_d * 0.5, 4.0) - 8, L + body_h + 10)
    ax.set_aspect("equal")
    ax.set_xlabel("Across jet axis (mm)")
    ax.set_ylabel("Height above hull (mm)")
    ax.set_title("Spray profile (one nozzle, zoomed)")
    ax.grid(True, alpha=0.25)
    return fig


def plot_spray_profile(s: Scenario) -> plt.Figure:
    import pickle
    return plot_spray_profile_cached(scenario_key(s), pickle.dumps(s))


# -----------------------------------------------------------------------------
# Single-disc coverage diagnostic (ring gap detector)
# -----------------------------------------------------------------------------
def single_disc_coverage(s: Scenario) -> dict:
    """
    Run the pressure sim for a SINGLE disc on a fine grid and measure whether
    its swept annulus is continuous (no gaps) as the disc advances.

    Returns the touched/untouched map plus the key geometric numbers:
      - forward advance per revolution and the effective forward pitch
        (advance / n_nozzles), vs the footprint diameter — the overlap
        condition for a gap-free along-track track;
      - annulus fill fraction: of the cells inside the swept ring band
        (radius in [r_ring - fp, r_ring + fp]), how many got any exposure.
    """
    one = deepcopy(s)
    one.n_row1 = 1
    one.n_row2 = 1
    one.array_width_mm = max(one.disc_diameter_mm + 40, 400)
    # Fine grid so a mm-scale footprint is resolved; short strip = a few revs.
    one.cell_size_mm = min(s.cell_size_mm, 2.0)
    rov_mm_s = max(one.rov_speed_kn * KNOTS_TO_MPS * 1000.0, 1e-6)
    rev_s = one.rpm / 60.0
    advance_per_rev = rov_mm_s / max(rev_s, 1e-6)
    # Simulate enough length to see several full revolutions sweep past.
    one.sim_length_mm = int(max(6 * advance_per_rev, 2 * one.disc_diameter_mm, 400))
    one.steady_state_only = False

    strip, m, box = simulate_pressure(one)
    cell = one.cell_size_mm
    touched = strip > 0.0

    fp_d = one.footprint_dia()
    n_noz = max(one.n_nozzles, 1)
    eff_pitch = advance_per_rev / n_noz
    overlap = eff_pitch < fp_d

    # The reliable gap signal is the analytical overlap condition
    # (eff_pitch vs footprint); the green map below shows it visually. We
    # avoid measuring a gap from pixels — a vertical cut through the ring's
    # near-tangent legs aliases the discrete blobs and gives a noisy number.
    ir = one.impact_radius_mm()
    ny, nx = strip.shape
    cx = nx / 2.0
    overlap_margin = fp_d - eff_pitch   # >0 = overlap, <0 = gap of this width

    return {
        "strip": strip,
        "touched": touched,
        "cell": cell,
        "box": box,
        "advance_per_rev_mm": advance_per_rev,
        "eff_pitch_mm": eff_pitch,
        "footprint_mm": fp_d,
        "overlap": bool(overlap),
        "overlap_margin_mm": overlap_margin,
        "n_nozzles": n_noz,
        "ring_r_mm": ir,
        "cx": cx,
    }


def plot_single_disc_coverage(s: Scenario) -> tuple[plt.Figure, dict]:
    d = single_disc_coverage(s)
    touched = d["touched"]
    cell = d["cell"]
    ny, nx = touched.shape
    extent = [-nx / 2 * cell, nx / 2 * cell, ny * cell, 0]

    fig, ax = plt.subplots(figsize=(5.5, 4.4))
    # Touched cells in solid colour over an untouched background.
    ax.imshow(touched.astype(float), extent=extent, aspect="equal",
              cmap="Greens", vmin=0, vmax=1.4, interpolation="nearest")
    # Overlay the theoretical ring-band edges.
    ir = d["ring_r_mm"]
    fp = d["footprint_mm"]
    for xr in (ir, -ir):
        ax.axvline(xr, color="#1f77b4", lw=0.7, ls=":")
    ax.set_xlabel("Across-track (mm)")
    ax.set_ylabel("Along-track (mm)")
    status = "OVERLAP — gap-free" if d["overlap"] else "GAP RISK"
    ax.set_title(f"Single-disc swept coverage — {status}")
    return fig, d


# -----------------------------------------------------------------------------
# Footprint sensitivity: impact-zone diameter vs standoff & nozzle exit dia
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def plot_footprint_sensitivity_cached(key: tuple, _scen_bytes: bytes) -> plt.Figure:
    import pickle
    return _plot_footprint_sensitivity_impl(pickle.loads(_scen_bytes))


def _plot_footprint_sensitivity_impl(s: Scenario) -> plt.Figure:
    """
    Capture how the impact-zone diameter responds to the two physical
    drivers: nozzle exit diameter and standoff distance to the hull.
    Footprint area ∝ diameter², so the right panel reports the relative
    cleaning-energy density (∝ 1/area) the jet delivers.
    """
    spread = math.tan(math.radians(s.jet_spread_deg))

    def fp(d0: float, standoff: float) -> float:
        return d0 + 2.0 * standoff * spread

    standoffs = np.linspace(5, 60, 80)
    exit_dias = sorted({0.5, 1.0, 1.5, 2.5, 4.0, float(s.nozzle_exit_mm)})

    fig, (axd, axe) = plt.subplots(1, 2, figsize=(9.5, 3.8))

    # Left: footprint diameter vs standoff, one curve per exit diameter.
    for d0 in exit_dias:
        is_current = abs(d0 - s.nozzle_exit_mm) < 1e-6
        axd.plot(standoffs, [fp(d0, L) for L in standoffs],
                 lw=2.2 if is_current else 1.2,
                 color="#d9660a" if is_current else "#9aa0a6",
                 label=f"exit {d0:.1f} mm" + (" (current)" if is_current else ""),
                 zorder=3 if is_current else 2)
    cur_fp = fp(s.nozzle_exit_mm, s.standoff_mm)
    axd.scatter([s.standoff_mm], [cur_fp], color="#d9660a", s=55, zorder=4,
                edgecolor="white", linewidth=1.0)
    axd.annotate(f"{cur_fp:.0f} mm",
                 xy=(s.standoff_mm, cur_fp),
                 xytext=(6, 8), textcoords="offset points",
                 color="#d9660a", fontsize=9, fontweight="bold")
    axd.axvline(s.standoff_mm, color="#d9660a", ls=":", lw=0.8, alpha=0.6)
    axd.set_xlabel("Standoff to hull (mm)")
    axd.set_ylabel("Impact-zone diameter (mm)")
    axd.set_title("Footprint vs standoff")
    axd.legend(fontsize=7, loc="upper left", framealpha=0.9)
    axd.grid(True, alpha=0.3)

    # Right: relative energy density (∝ 1/footprint area) vs standoff.
    base = fp(s.nozzle_exit_mm, s.standoff_mm)
    for d0 in exit_dias:
        is_current = abs(d0 - s.nozzle_exit_mm) < 1e-6
        dens = [(base / fp(d0, L)) ** 2 for L in standoffs]
        axe.plot(standoffs, dens,
                 lw=2.2 if is_current else 1.2,
                 color="#1f77b4" if is_current else "#bcd2e8",
                 label=f"exit {d0:.1f} mm" + (" (current)" if is_current else ""),
                 zorder=3 if is_current else 2)
    axe.axhline(1.0, color="#888", ls="--", lw=0.8)
    axe.axvline(s.standoff_mm, color="#1f77b4", ls=":", lw=0.8, alpha=0.6)
    axe.scatter([s.standoff_mm], [1.0], color="#1f77b4", s=55, zorder=4,
                edgecolor="white", linewidth=1.0)
    axe.set_xlabel("Standoff to hull (mm)")
    axe.set_ylabel("Relative energy density (× current)")
    axe.set_title("Energy density ∝ 1 / area")
    axe.legend(fontsize=7, loc="upper right", framealpha=0.9)
    axe.grid(True, alpha=0.3)

    fig.suptitle(
        f"Impact zone driven by exit dia & standoff "
        f"(spread half-angle {s.jet_spread_deg:.1f}°)",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def plot_footprint_sensitivity(s: Scenario) -> plt.Figure:
    import pickle
    return plot_footprint_sensitivity_cached(scenario_key(s), pickle.dumps(s))


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
    r_cells = r_mm / s.cell_size_mm
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


# -----------------------------------------------------------------------------
# Core nozzle-position helper (scalar version — kept for simulate_pressure)
# -----------------------------------------------------------------------------
def nozzle_positions_hull(s: Scenario, t: float,
                          rotated: list[tuple[float, float, int]],
                          phases: list[float]
                          ) -> list[list[tuple[float, float]]]:
    """
    Nozzle positions in the hull frame at time t.
    Output: list per disc -> list of (x, y) positions, one per nozzle.
    """
    rov_speed_mm_s = s.rov_speed_kn * KNOTS_TO_MPS * 1000.0
    omega = s.rpm * 2 * math.pi / 60.0
    yaw_rad = math.radians(s.yaw_deg)
    cos_t, sin_t = math.cos(yaw_rad), math.sin(yaw_rad)
    ir = s.impact_radius_mm()

    rys = [r[1] for r in rotated]
    array_leading_y = min(rys) - s.disc_diameter_mm / 2
    margin_mm = 120
    array_y_offset_init = -margin_mm - array_leading_y
    array_y = array_y_offset_init + rov_speed_mm_s * t

    out = []
    for di, (rx, ry, direction) in enumerate(rotated):
        phase = phases[di] + direction * omega * t
        dcx = rx
        dcy = ry + array_y
        per_disc = []
        for kn in range(s.n_nozzles):
            theta = phase + 2 * math.pi * kn / s.n_nozzles
            lx = ir * math.cos(theta)
            ly = ir * math.sin(theta)
            gx = lx * cos_t - ly * sin_t
            gy = lx * sin_t + ly * cos_t
            per_disc.append((dcx + gx, dcy + gy))
        out.append(per_disc)
    return out


def disc_centres_hull(s: Scenario, t: float,
                      rotated: list[tuple[float, float, int]]
                      ) -> list[tuple[float, float, int]]:
    """Disc centres in the hull frame at time t."""
    rov_speed_mm_s = s.rov_speed_kn * KNOTS_TO_MPS * 1000.0
    rys = [r[1] for r in rotated]
    array_leading_y = min(rys) - s.disc_diameter_mm / 2
    margin_mm = 120
    array_y_offset_init = -margin_mm - array_leading_y
    array_y = array_y_offset_init + rov_speed_mm_s * t
    return [(rx, ry + array_y, dirn) for (rx, ry, dirn) in rotated]


# -----------------------------------------------------------------------------
# Vectorised trail computation
# -----------------------------------------------------------------------------
def nozzle_trails_vec(s: Scenario, ts: np.ndarray,
                      rotated: list[tuple[float, float, int]],
                      phases: list[float]
                      ) -> np.ndarray:
    """
    Vectorised nozzle positions for all (time, disc, nozzle) tuples.

    Returns array of shape (T, D, N, 2) with (x, y) in hull frame, where
      T = len(ts), D = #discs, N = n_nozzles.
    """
    ts = np.asarray(ts, dtype=np.float64)
    T = ts.size
    D = len(rotated)
    N = s.n_nozzles

    rov_speed_mm_s = s.rov_speed_kn * KNOTS_TO_MPS * 1000.0
    omega = s.rpm * 2 * math.pi / 60.0
    yaw_rad = math.radians(s.yaw_deg)
    cos_yaw, sin_yaw = math.cos(yaw_rad), math.sin(yaw_rad)
    ir = s.impact_radius_mm()

    rys = np.array([r[1] for r in rotated])
    array_leading_y = rys.min() - s.disc_diameter_mm / 2
    margin_mm = 120
    array_y_offset_init = -margin_mm - array_leading_y
    array_y_t = array_y_offset_init + rov_speed_mm_s * ts      # (T,)

    disc_rx = np.array([r[0] for r in rotated])                # (D,)
    disc_ry = np.array([r[1] for r in rotated])                # (D,)
    disc_dir = np.array([r[2] for r in rotated], dtype=np.float64)  # (D,)
    phases_arr = np.asarray(phases, dtype=np.float64)          # (D,)

    # nozzle index offsets (N,)
    nk = np.arange(N, dtype=np.float64) * (2 * math.pi / N)

    # phase[T, D, N] = phases[D] + dir[D] * omega * ts[T] + 2πk/N [N]
    phase = (phases_arr[None, :, None]
             + disc_dir[None, :, None] * omega * ts[:, None, None]
             + nk[None, None, :])

    lx = ir * np.cos(phase)                                     # (T, D, N)
    ly = ir * np.sin(phase)
    gx = lx * cos_yaw - ly * sin_yaw
    gy = lx * sin_yaw + ly * cos_yaw

    x = disc_rx[None, :, None] + gx                              # (T, D, N)
    y = disc_ry[None, :, None] + array_y_t[:, None, None] + gy   # (T, D, N)

    return np.stack([x, y], axis=-1)   # (T, D, N, 2)


# -----------------------------------------------------------------------------
# Pressure simulation (unchanged — supports optional t_stop_s)
# -----------------------------------------------------------------------------
def simulate_pressure(s: Scenario, t_stop_s: float | None = None
                      ) -> tuple[np.ndarray, dict, tuple[int, int, int, int]]:
    discs = disc_layout(s)
    rov_speed_mm_s = s.rov_speed_kn * KNOTS_TO_MPS * 1000.0
    omega_rad_s = s.rpm * 2 * math.pi / 60.0
    yaw_rad = math.radians(s.yaw_deg)
    cos_t, sin_t = math.cos(yaw_rad), math.sin(yaw_rad)
    cx_array, cy_array = 0.0, s.row_pitch_mm / 2.0

    rotated: list[tuple[float, float, int]] = []
    for d in discs:
        rx, ry = rotate_point(d.cx_mm, d.cy_mm, cos_t, sin_t, cx_array, cy_array)
        rotated.append((rx, ry, d.direction))

    rxs = [r[0] for r in rotated]
    rys = [r[1] for r in rotated]
    array_span_x = max(rxs) - min(rxs) + s.disc_diameter_mm
    array_span_y = max(rys) - min(rys) + s.disc_diameter_mm

    margin_mm = 120
    full_extent = s.sim_length_mm + array_span_y + 2 * margin_mm
    width_extent = array_span_x + 2 * margin_mm

    cell = s.cell_size_mm
    nx = int(width_extent / cell)
    ny = int(full_extent / cell)
    grid = np.zeros((ny, nx), dtype=np.float32)
    passes = np.zeros((ny, nx), dtype=np.float32)  # nozzle passes per cell

    # Time-step constraints. The integrated exposure is Σ p·dt, which only
    # tracks the true dwell time if the jet's footprint cannot SKIP past
    # cells between samples. The binding constraint is the tangential speed
    # of the nozzle along its impact ring (omega·r), which at high RPM is
    # far faster than the ROV traverse: the nozzle must not advance more
    # than half a footprint width per step, or the swept track breaks up
    # into a dotted line and the per-cell exposure is undercounted.
    ir = s.impact_radius_mm()
    fp_d = s.footprint_dia()
    v_tan = omega_rad_s * max(ir, 1e-6)            # mm/s along the ring
    dt_rot = math.radians(5.0) / max(omega_rad_s, 1e-6)
    dt_trav = 0.5 * cell / max(rov_speed_mm_s, 1e-6)
    dt_arc = 0.5 * max(fp_d, cell) / max(v_tan, 1e-6)  # footprint can't skip
    dt = min(dt_rot, dt_trav, dt_arc)
    total_time_full = s.sim_length_mm / max(rov_speed_mm_s, 1e-6)
    total_time = total_time_full if t_stop_s is None else min(t_stop_s, total_time_full)
    n_steps_ideal = int(total_time / dt) + 1
    # Cap to bound runtime, but record whether the cap forced a coarser dt
    # than the jet-path constraint wants (i.e. the track is undersampled).
    STEP_CAP = 60000
    n_steps = min(n_steps_ideal, STEP_CAP)
    dt = total_time / max(n_steps, 1)
    arc_per_step = v_tan * dt
    undersampled = arc_per_step > max(fp_d, cell)

    stencil = footprint_stencil(s)
    sh, sw = stencil.shape
    rr = sh // 2
    p_dt = float(s.pressure_bar) * dt
    weighted_stencil = (stencil * p_dt).astype(np.float32)

    x0 = -width_extent / 2
    array_leading_y = min(rys) - s.disc_diameter_mm / 2
    array_y_offset_init = -margin_mm - array_leading_y

    # ---- Vectorised nozzle impact positions -------------------------------
    # Compute every (step, disc, nozzle) impact centre as NumPy arrays, then
    # scatter-add the footprint stencil in one pass per stencil cell. This
    # replaces the former triple Python loop and is what keeps fine grids
    # (1–2 mm) responsive.
    n_discs = len(rotated)
    if not (n_discs == 0 or s.n_nozzles == 0 or n_steps == 0):
        steps = np.arange(n_steps, dtype=np.float64)
        t = steps * dt                                   # (S,)
        array_y = array_y_offset_init + rov_speed_mm_s * t  # (S,)

        rx = np.array([r[0] for r in rotated])           # (D,)
        ry = np.array([r[1] for r in rotated])           # (D,)
        direction = np.array([r[2] for r in rotated])    # (D,)
        phase0 = (2 * np.pi * np.arange(n_discs)
                  / max(n_discs, 1))                      # (D,)
        kn = np.arange(s.n_nozzles)                       # (N,)
        nozzle_off = 2 * np.pi * kn / s.n_nozzles         # (N,)

        # theta[s, d, n]
        phase = (phase0[None, :]
                 + direction[None, :] * omega_rad_s * t[:, None])  # (S, D)
        theta = phase[:, :, None] + nozzle_off[None, None, :]      # (S, D, N)

        lx = ir * np.cos(theta)
        ly = ir * np.sin(theta)
        gx = lx * cos_t - ly * sin_t
        gy = lx * sin_t + ly * cos_t

        nx_mm = rx[None, :, None] + gx                            # (S, D, N)
        ny_mm = (ry[None, :, None] + array_y[:, None, None] + gy)  # (S, D, N)

        ix = np.round((nx_mm - x0) / cell).astype(np.int64).ravel()
        iy = np.round(ny_mm / cell).astype(np.int64).ravel()

        # Every nozzle hit deposits the SAME weighted stencil, so the total
        # exposure is the 2-D convolution of a "hit-count" grid with that
        # stencil — independent of the number of timesteps and the stencil
        # size, which is what keeps fine grids fast.
        #
        # We bin hits into a grid padded by rr on every side so that a hit
        # whose centre sits just off the real grid still contributes its
        # overlapping stencil tail (matching the original clipped-deposit
        # loop). Hits whose stencil cannot touch the grid at all are dropped.
        pix = ix + rr            # column in the padded grid
        piy = iy + rr            # row in the padded grid
        pny = ny + 2 * rr
        pnx = nx + 2 * rr
        keep = ((pix >= 0) & (pix < pnx) & (piy >= 0) & (piy < pny))
        hit = np.bincount((piy[keep] * pnx + pix[keep]),
                          minlength=pny * pnx).astype(np.float32).reshape(pny, pnx)

        # Binary footprint (presence): convolving the hit grid with it counts
        # how many nozzle PASSES covered each cell — the dose for the cleaning
        # criterion, distinct from the energy (bar·s) deposit above.
        binary_stencil = (stencil > 0.0).astype(np.float32)

        if sh == 1 and sw == 1:
            # Single-cell footprint (sub-resolution jet): no spread to apply.
            hcore = hit[rr:rr + ny, rr:rr + nx]
            grid += hcore * weighted_stencil[0, 0]
            passes += hcore * binary_stencil[0, 0]
        else:
            # FFT convolution: O(MN log MN), stencil-size independent. The
            # hit grid's FFT is shared by the energy and pass-count deposits.
            fy = pny + sh - 1
            fx = pnx + sw - 1
            H = np.fft.rfft2(hit, s=(fy, fx))
            K = np.fft.rfft2(weighted_stencil, s=(fy, fx))
            conv = np.fft.irfft2(H * K, s=(fy, fx))
            # The hit at padded (piy, pix) deposits stencil cell (dy, dx) at
            # padded (piy+dy-rr, pix+dx-rr); the full convolution places it at
            # conv[piy+dy, pix+dx]. The real grid cell (i, j) corresponds to
            # padded (i+rr, j+rr), so it reads conv[i+2*rr, j+2*rr].
            grid += conv[2 * rr:2 * rr + ny, 2 * rr:2 * rr + nx].astype(np.float32)
            Kb = np.fft.rfft2(binary_stencil, s=(fy, fx))
            convb = np.fft.irfft2(H * Kb, s=(fy, fx))
            passes += convb[2 * rr:2 * rr + ny,
                            2 * rr:2 * rr + nx].astype(np.float32)

    y_strip_start = 0
    y_strip_end = int((s.sim_length_mm + array_span_y) / cell)
    x_strip_start = int(margin_mm / cell)
    x_strip_end = int((margin_mm + array_span_x) / cell)
    strip = grid[y_strip_start:y_strip_end, x_strip_start:x_strip_end]
    passes_strip = passes[y_strip_start:y_strip_end, x_strip_start:x_strip_end]

    core_y0 = int(array_span_y / cell)
    core_y1 = int(s.sim_length_mm / cell)
    if core_y1 <= core_y0:
        core_y0, core_y1 = 0, strip.shape[0]
    core_x0 = 0
    core_x1 = strip.shape[1]
    core_box = (core_y0, core_y1, core_x0, core_x1)

    if s.steady_state_only:
        region = strip[core_y0:core_y1, core_x0:core_x1]
        region_passes = passes_strip[core_y0:core_y1, core_x0:core_x1]
    else:
        region = strip
        region_passes = passes_strip

    # --- Cleaning criterion: intensity gate × dose gate -------------------
    # Intensity: the jet must deliver enough stagnation pressure at the hull
    # to lift the fouling. This is per-scenario (same jet everywhere), so it
    # gates the WHOLE field on or off. Dose: a cell also needs a minimum
    # number of passes. A cell is cleaned iff both hold.
    p_stag = s.stagnation_pressure_bar()
    intensity_ok = p_stag >= s.removal_pressure_bar
    if region_passes.size and intensity_ok:
        cleaned_mask = region_passes >= s.min_passes
        cleaned_pct = float(cleaned_mask.mean() * 100.0)
    else:
        cleaned_pct = 0.0

    metrics = {
        "rov_speed_mm_s": rov_speed_mm_s,
        "dt_ms": dt * 1000.0,
        "n_steps": n_steps,
        "array_span_x_mm": array_span_x,
        "array_span_y_mm": array_span_y,
        "mean_bs": float(region.mean()) if region.size else 0.0,
        "p10_bs": float(np.percentile(region, 10)) if region.size else 0.0,
        "p50_bs": float(np.percentile(region, 50)) if region.size else 0.0,
        "p90_bs": float(np.percentile(region, 90)) if region.size else 0.0,
        "min_bs": float(region.min()) if region.size else 0.0,
        "max_bs": float(region.max()) if region.size else 0.0,
        # Legacy bar·s coverage (kept for continuity / the dose heatmap).
        "coverage_pct": float((region >= s.clean_threshold).mean() * 100.0)
                       if region.size else 0.0,
        # New physically-gated cleaning coverage.
        "cleaned_pct": cleaned_pct,
        "stagnation_pressure_bar": float(p_stag),
        "intensity_ok": bool(intensity_ok),
        "median_passes": float(np.median(region_passes)) if region_passes.size else 0.0,
        "max_passes": float(region_passes.max()) if region_passes.size else 0.0,
        "missed_pct": float((region == 0).mean() * 100.0) if region.size else 0.0,
        "region_area_mm2": float(region.size * cell * cell),
        "total_time_s": total_time_full,
        "arc_per_step_mm": float(arc_per_step),
        "footprint_mm": float(fp_d),
        "undersampled": bool(undersampled),
    }
    return strip, metrics, core_box


# -----------------------------------------------------------------------------
# Fast motion-viz renderer
# -----------------------------------------------------------------------------
def plot_motion_fast(s: Scenario, t_now_s: float,
                     trail_revolutions: float = 1.0,
                     frame: str = "Hull frame",
                     cumulative_strip: np.ndarray | None = None,
                     fixed_xlim: tuple[float, float] | None = None,
                     fixed_ylim: tuple[float, float] | None = None,
                     ) -> plt.Figure:
    """
    Fast render: trails via LineCollection, current footprints via
    PatchCollection. Trails computed by vectorised NumPy.

    If ``fixed_xlim``/``fixed_ylim`` are provided, they override the
    auto-fit axis limits. This is essential for the pre-rendered
    animation so the camera doesn't shift from frame to frame.
    """
    rotated = compute_rotated_discs(s)
    phases = [2 * math.pi * i / max(len(rotated), 1) for i in range(len(rotated))]
    D = len(rotated)
    N = s.n_nozzles

    omega = s.rpm * 2 * math.pi / 60.0
    trail_duration_s = trail_revolutions * (2 * math.pi / max(omega, 1e-6))
    t_start = max(0.0, t_now_s - trail_duration_s)
    rov_speed_mm_s = s.rov_speed_kn * KNOTS_TO_MPS * 1000.0

    # Sample density: keep segment length in physical units roughly constant.
    # Target ~4° of rotation per sample.
    trail_n = max(20, int(trail_revolutions * 90))
    trail_n = min(trail_n, 2000)  # hard cap
    if trail_duration_s > 0:
        ts = np.linspace(t_start, t_now_s, trail_n)
    else:
        ts = np.array([t_now_s])

    pts = nozzle_trails_vec(s, ts, rotated, phases)   # (T, D, N, 2)

    # Apply ROV-frame shift if needed
    if frame == "ROV frame":
        rys_rot = np.array([r[1] for r in rotated])
        array_leading_y = rys_rot.min() - s.disc_diameter_mm / 2
        margin_mm = 120
        array_y_offset_init = -margin_mm - array_leading_y
        array_y_t = array_y_offset_init + rov_speed_mm_s * ts
        # Subtract each row's array_y from each time slice
        pts = pts.copy()
        pts[..., 1] -= array_y_t[:, None, None]
        array_y_now = array_y_offset_init + rov_speed_mm_s * t_now_s
        y_shift_now = -array_y_now
    else:
        array_y_now = 0.0  # ignored
        y_shift_now = 0.0

    # Build line segments for LineCollection.
    # For each (disc, nozzle) series of T points -> (T-1) line segments.
    # Concatenate all DNx (T-1) segments into one array of shape (S, 2, 2).
    T = pts.shape[0]

    # Pick a figure size so the equal-aspect plot region fills most of
    # the canvas without excessive whitespace. Use fixed_xlim/fixed_ylim
    # if provided (they define the real extent); otherwise estimate
    # from current-frame pts.
    if fixed_xlim is not None and fixed_ylim is not None:
        x_span = fixed_xlim[1] - fixed_xlim[0]
        y_span = fixed_ylim[1] - fixed_ylim[0]
    else:
        # np.ptp() is the NumPy-2.0-safe form; .ptp() on arrays was removed.
        x_span = max(60.0, float(np.ptp(pts[..., 0])) + 300)
        y_span = max(60.0, float(np.ptp(pts[..., 1])) + 300)

    # Plot area target: short side ≈ 4", long side scales with data aspect
    # but capped so the figure stays reasonable in the Streamlit column.
    data_aspect = y_span / max(x_span, 1e-6)
    # Axes margin for title/labels/ticks (≈ 1.5" total in each dimension)
    axes_margin_in = 1.5
    if data_aspect >= 1.0:
        # Tall plot (Hull frame typical): fix width at 6", scale height
        plot_w = 6.0
        plot_h = min(plot_w * data_aspect, 12.0)  # cap at 12"
    else:
        # Wide plot (ROV frame typical): fix height at 4", scale width
        plot_h = 4.0
        plot_w = min(plot_h / data_aspect, 14.0)
    fig_w = plot_w + axes_margin_in
    fig_h = plot_h + axes_margin_in
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Cumulative heatmap underlay (hull frame only)
    if cumulative_strip is not None and frame == "Hull frame":
        rxs_rot = [r[0] for r in rotated]
        array_span_x = max(rxs_rot) - min(rxs_rot) + s.disc_diameter_mm
        extent = [-array_span_x / 2, array_span_x / 2,
                  cumulative_strip.shape[0] * s.cell_size_mm, 0]
        vmax = max(float(cumulative_strip.max()), 1e-6)
        ax.imshow(cumulative_strip, extent=extent, aspect="auto",
                  cmap="viridis", vmin=0, vmax=vmax, alpha=0.7, zorder=0)

    # Trails via LineCollection with per-segment alpha
    if T >= 2:
        # Pts for each nozzle: shape (T, 2). We need consecutive pairs.
        # Reshape pts -> (D*N, T, 2)
        pts_flat = pts.transpose(1, 2, 0, 3).reshape(D * N, T, 2)
        # Segments: (D*N, T-1, 2, 2)
        segs = np.stack([pts_flat[:, :-1, :], pts_flat[:, 1:, :]], axis=2)
        segs = segs.reshape(-1, 2, 2)                         # (D*N*(T-1), 2, 2)
        # Alpha gradient — newer samples more opaque
        frac = np.linspace(0.0, 1.0, T - 1)                    # (T-1,)
        alphas = 0.15 + 0.55 * frac                            # (T-1,)
        alphas_all = np.tile(alphas, D * N)                    # (D*N*(T-1),)
        # Build RGBA colours (all orange, varying alpha)
        base_rgb = np.array([1.0, 127/255, 14/255])
        colors = np.empty((alphas_all.size, 4))
        colors[:, :3] = base_rgb[None, :]
        colors[:, 3] = alphas_all
        lc = LineCollection(segs, colors=colors, linewidths=0.9, zorder=2)
        ax.add_collection(lc)

    # Current nozzle positions (last time slice of pts, but we didn't include
    # t_now in pts if trail_duration=0; recompute cleanly)
    pts_now = nozzle_trails_vec(s, np.array([t_now_s]),
                                rotated, phases)[0]            # (D, N, 2)
    # Apply ROV-frame shift at t_now
    if frame == "ROV frame":
        pts_now = pts_now.copy()
        pts_now[..., 1] += y_shift_now

    fp_r = s.footprint_dia() / 2.0
    impact_r = s.impact_radius_mm()

    # Disc outlines + impact rings via PatchCollections (2 calls total)
    centres_now = disc_centres_hull(s, t_now_s, rotated)
    disc_patches = []
    ring_patches = []
    disc_xs, disc_ys = [], []
    for (cx_d, cy_d, _) in centres_now:
        cy_s = cy_d + y_shift_now
        disc_xs.append(cx_d); disc_ys.append(cy_s)
        disc_patches.append(mpatches.Circle(
            (cx_d, cy_s), s.disc_diameter_mm / 2))
        ring_patches.append(mpatches.Circle(
            (cx_d, cy_s), impact_r))
    ax.add_collection(PatchCollection(
        disc_patches, facecolor="none", edgecolor="#1f77b4",
        linewidths=1.2, zorder=3))
    ax.add_collection(PatchCollection(
        ring_patches, facecolor="none", edgecolor="#888",
        linewidths=0.6, linestyles="--", zorder=3))

    # Current footprints as a single PatchCollection
    foot_patches = [mpatches.Circle((pts_now[di, kn, 0], pts_now[di, kn, 1]),
                                     fp_r)
                    for di in range(D) for kn in range(N)]
    ax.add_collection(PatchCollection(
        foot_patches, facecolor="#ff7f0e", alpha=0.55,
        edgecolor="#c65500", linewidths=0.6, zorder=4))
    # Current nozzle points (scatter)
    ax.scatter(pts_now[..., 0].ravel(), pts_now[..., 1].ravel(),
               c="#c65500", s=10, zorder=5)

    # Array frame rectangle (rotated)
    rys_rot = [r[1] for r in rotated]
    rxs_rot = [r[0] for r in rotated]
    half_fw = max(abs(min(rxs_rot)), abs(max(rxs_rot))) + s.disc_diameter_mm / 2 + 30
    y_top_rel = min(rys_rot) - s.disc_diameter_mm / 2 - 30
    y_bot_rel = max(rys_rot) + s.disc_diameter_mm / 2 + 30
    if frame == "ROV frame":
        y_top, y_bot = y_top_rel, y_bot_rel
    else:
        array_leading_y = min(rys_rot) - s.disc_diameter_mm / 2
        margin_mm = 120
        array_y_offset_init = -margin_mm - array_leading_y
        array_y_now_h = array_y_offset_init + rov_speed_mm_s * t_now_s
        y_top = y_top_rel + array_y_now_h
        y_bot = y_bot_rel + array_y_now_h
    ax.add_patch(mpatches.Polygon(
        [(-half_fw, y_top), (half_fw, y_top),
         (half_fw, y_bot), (-half_fw, y_bot)],
        closed=True, fill=False, edgecolor="#444", linewidth=1.2, zorder=2))

    # Travel arrow
    if frame == "Hull frame":
        ax.annotate("ROV travel",
                    xy=(half_fw + 40, y_bot), xytext=(half_fw + 40, y_top),
                    arrowprops=dict(arrowstyle="->", color="#555"),
                    ha="center", va="center", fontsize=9, color="#555")
    else:
        ax.annotate("hull scrolls",
                    xy=(half_fw + 40, y_top), xytext=(half_fw + 40, y_bot),
                    arrowprops=dict(arrowstyle="->", color="#555"),
                    ha="center", va="center", fontsize=9, color="#555")

    # Axis limits
    if fixed_xlim is not None:
        x_lo, x_hi = fixed_xlim
    else:
        all_x = list(disc_xs) + [float(pts[..., 0].min()),
                                 float(pts[..., 0].max())]
        x_lo = min(min(all_x), -half_fw) - 60
        x_hi = max(max(all_x), half_fw) + 180

    if fixed_ylim is not None:
        y_lo, y_hi = fixed_ylim
    else:
        all_y = list(disc_ys) + [float(pts[..., 1].min()),
                                 float(pts[..., 1].max())]
        y_lo = min(min(all_y), y_top) - 60
        y_hi = max(max(all_y), y_bot) + 60

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Across-track (mm)")
    ax.set_ylabel("Along-track (mm)")
    ax.set_title(
        f"Motion snapshot at t = {t_now_s*1000:.0f} ms  "
        f"(trail = {trail_revolutions:.2f} rev = "
        f"{trail_duration_s*1000:.0f} ms), {frame}"
    )
    ax.grid(True, alpha=0.25)
    return fig


def full_traversal_limits(s: Scenario, trail_revolutions: float,
                          frame: str,
                          n_samples: int = 60,
                          pad_mm: float = 60.0,
                          t_start_s: float = 0.0,
                          t_end_s: float | None = None,
                          ) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Compute the Hull-frame (or ROV-frame) axis window so the array and
    trails remain visible for ALL t in ``[t_start_s, t_end_s]``.

    Default range is the full traversal (``t_end_s=None`` → total time).
    Pass a narrower window to get a tighter camera for a segment-only
    playback.
    """
    rotated = compute_rotated_discs(s)
    phases = [2 * math.pi * i / max(len(rotated), 1) for i in range(len(rotated))]
    rov_speed_mm_s = s.rov_speed_kn * KNOTS_TO_MPS * 1000.0
    total_time_s = s.sim_length_mm / max(rov_speed_mm_s, 1e-6)
    if t_end_s is None:
        t_end_s = total_time_s
    t_start_s = max(0.0, float(t_start_s))
    t_end_s = min(float(t_end_s), total_time_s)
    if t_end_s <= t_start_s:
        t_end_s = t_start_s + 1e-3

    rxs_rot = [r[0] for r in rotated]
    rys_rot = [r[1] for r in rotated]
    half_fw = max(abs(min(rxs_rot)), abs(max(rxs_rot))) + s.disc_diameter_mm / 2 + 30
    y_top_rel = min(rys_rot) - s.disc_diameter_mm / 2 - 30
    y_bot_rel = max(rys_rot) + s.disc_diameter_mm / 2 + 30

    omega = s.rpm * 2 * math.pi / 60.0
    trail_dur = trail_revolutions * (2 * math.pi / max(omega, 1e-6))

    ts = np.linspace(t_start_s, t_end_s, max(n_samples, 2))
    pts = nozzle_trails_vec(s, ts, rotated, phases)   # (T, D, N, 2)

    if frame == "ROV frame":
        # Trails collapse to a stationary rosette; also include hull array box.
        rys_np = np.array(rys_rot)
        aly0 = rys_np.min() - s.disc_diameter_mm / 2
        array_y_offset_init = -120.0 - aly0
        array_y_t = array_y_offset_init + rov_speed_mm_s * ts
        pts = pts.copy()
        pts[..., 1] -= array_y_t[:, None, None]
        y_lo = min(float(pts[..., 1].min()), y_top_rel) - pad_mm
        y_hi = max(float(pts[..., 1].max()), y_bot_rel) + pad_mm
    else:
        # Hull frame: array slides from t_start..t_end; include the
        # trail extent too (which lags behind by up to trail_dur).
        array_leading_y = min(rys_rot) - s.disc_diameter_mm / 2
        margin_mm = 120.0
        array_y_offset_init = -margin_mm - array_leading_y
        y_top_tstart = y_top_rel + array_y_offset_init \
            + rov_speed_mm_s * t_start_s
        y_bot_tend = y_bot_rel + array_y_offset_init \
            + rov_speed_mm_s * t_end_s
        # trails extend up to trail_dur behind the current array position
        trail_back = rov_speed_mm_s * trail_dur
        y_lo = min(y_top_tstart, y_top_tstart - trail_back) - pad_mm
        y_hi = y_bot_tend + pad_mm

    all_x = list(rxs_rot) + [float(pts[..., 0].min()), float(pts[..., 0].max())]
    x_lo = min(min(all_x), -half_fw) - pad_mm
    x_hi = max(max(all_x), half_fw) + (180.0 if frame == "Hull frame" else pad_mm)

    return (x_lo, x_hi), (y_lo, y_hi)


# -----------------------------------------------------------------------------
# Hull simulation helpers
# -----------------------------------------------------------------------------
# Three hull-shape presets parametrising the midship section. Each entry
# gives:
#   Cm         — midship coefficient (ratio of midship section area to B*T)
#   r_rel      — bilge radius as a fraction of beam B
#   deadrise   — V-bottom deadrise angle in degrees (0 for flat bottoms)
#   k_side     — side-surface correction factor (≈ 1 for flat sides,
#                >1 for flared/curved sides)
#   label      — short UI label
#   blurb      — one-line description
HULL_SHAPES = {
    "full": {
        "Cm": 0.98, "r_rel": 0.10, "deadrise": 0.0,
        "k_side": 1.00,
        "label": "Full (block)",
        "blurb": "Flat bottom, boxy — container, bulk carrier, large tanker.",
    },
    "typical": {
        "Cm": 0.88, "r_rel": 0.30, "deadrise": 0.0,
        "k_side": 1.02,
        "label": "Typical",
        "blurb": "Rounded bilge — general cargo, product tanker, ferry.",
    },
    "fine": {
        "Cm": 0.72, "r_rel": 0.15, "deadrise": 20.0,
        "k_side": 1.05,
        "label": "Fine (V-keel)",
        "blurb": "Deep-V with sharp keel — naval, fast ferry, yacht.",
    },
}


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


def plot_hull_section(shape_key: str, beam_m: float, draft_m: float,
                      title: str | None = None,
                      figsize: tuple[float, float] = (3.0, 2.4)
                      ) -> plt.Figure:
    """
    Draw the midship cross-section of a hull shape, with waterline.
    Used as a visual selection card in the Hull simulation tab.
    """
    shape = HULL_SHAPES[shape_key]
    fig, ax = plt.subplots(figsize=figsize)
    B = beam_m
    T = draft_m
    r = shape["r_rel"] * B

    # Build outline as a list of (x, y) points starting at port waterline
    # (x = -B/2, y = 0), going DOWN the port side, around the bottom,
    # up the starboard side, and closing at (B/2, 0). Waterline = y=0;
    # depth is positive downward in this plot (we'll invert y-axis).
    # Coordinates: x horizontal (port negative, stbd positive),
    # y = depth-below-waterline (positive downward). ax.invert_yaxis()
    # later flips this so that on-screen "up" corresponds to "above
    # waterline".  We trace the outline counter-clockwise in (x, y)
    # space, starting at port waterline:
    #   port WL → down port side → port bilge → bottom → stbd bilge →
    #   up stbd side → stbd WL, closing back to port WL via the top.
    pts = []
    if shape["deadrise"] <= 0.0:
        # ------------- Flat bottom with rounded bilges -------------
        # Port bilge: quarter circle centred at (-B/2 + r, T - r).
        # Start tangent-to-side: radius points in -x direction → θ=π.
        # End tangent-to-bottom: radius points in +y direction → θ=π/2.
        # Sweep θ from π DOWN to π/2 (angle decreases).
        pts.append((-B / 2, 0))
        pts.append((-B / 2, T - r))
        thetas = np.linspace(math.pi, 0.5 * math.pi, 30)
        for th in thetas:
            pts.append((-B / 2 + r + r * math.cos(th),
                        T - r + r * math.sin(th)))
        # Flat bottom
        pts.append((B / 2 - r, T))
        # Starboard bilge: centred at (B/2 - r, T - r).
        # Start tangent-to-bottom: θ = π/2 (radius → +y).
        # End tangent-to-side: θ = 0 (radius → +x).
        thetas = np.linspace(0.5 * math.pi, 0.0, 30)
        for th in thetas:
            pts.append((B / 2 - r + r * math.cos(th),
                        T - r + r * math.sin(th)))
        pts.append((B / 2, T - r))
        pts.append((B / 2, 0))
    else:
        # ------------- V-bottom with bilge radius ------------------
        alpha = math.radians(shape["deadrise"])
        # Half-flat is NOT a flat bottom — it's the horizontal distance
        # from keel centreline to the turn-of-bilge on each side.
        half_flat = B / 2 - r
        # Port bilge centre is offset outboard of the side by r, and
        # above (shallower than) the keel. Its centre sits at
        # (-B/2 + r, T - r) — same as flat case — but the arc ends
        # tangent to a V-face sloping downward-and-inboard at angle α
        # below horizontal. The arc therefore sweeps from θ = π
        # (tangent to vertical side) to θ = π/2 + α (tangent to V-face,
        # i.e. radius normal to V-face).
        a0 = math.pi
        a1 = 0.5 * math.pi + alpha
        pts.append((-B / 2, 0))
        pts.append((-B / 2, T - r))
        thetas = np.linspace(a0, a1, 30)
        for th in thetas:
            pts.append((-B / 2 + r + r * math.cos(th),
                        T - r + r * math.sin(th)))
        # V-face from end-of-port-bilge down to keel at x = 0.
        x_end_port = -B / 2 + r + r * math.cos(a1)   # negative
        y_end_port = T - r + r * math.sin(a1)
        y_keel = y_end_port + (-x_end_port) * math.tan(alpha)
        pts.append((0, y_keel))
        # Mirror across x=0 to the starboard turn-of-bilge.
        x_start_stbd = -x_end_port                    # positive
        y_start_stbd = y_end_port
        pts.append((x_start_stbd, y_start_stbd))
        # Starboard bilge: centred at (B/2 - r, T - r). Sweep from the
        # V-tangent angle (π/2 − α) back to θ = 0 (tangent to vertical
        # side), going CLOCKWISE in (x, y) so angles decrease.
        a0s = 0.5 * math.pi - alpha
        a1s = 0.0
        thetas = np.linspace(a0s, a1s, 30)
        for th in thetas:
            pts.append((B / 2 - r + r * math.cos(th),
                        T - r + r * math.sin(th)))
        pts.append((B / 2, T - r))
        pts.append((B / 2, 0))

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.fill(xs, ys, color="#cfd8dc", edgecolor="#263238", linewidth=1.4)

    # Waterline
    wl_x = max(abs(min(xs)), abs(max(xs))) * 1.2
    ax.plot([-wl_x, wl_x], [0, 0], color="#0288d1",
            lw=1.0, ls="--", alpha=0.8)
    ax.text(wl_x * 0.95, -0.02 * max(T, 1), "WL",
            fontsize=7, color="#0288d1", ha="right", va="bottom")

    # Dimensions
    ax.annotate(f"B = {B:.1f} m",
                xy=(0, -0.04 * max(T, 1)), ha="center", va="bottom",
                fontsize=8, color="#333")
    ax.annotate(f"T = {T:.1f} m",
                xy=(-wl_x * 0.98, T / 2), ha="left", va="center",
                fontsize=8, color="#333", rotation=90)

    ax.set_aspect("equal")
    ax.invert_yaxis()       # so draft goes downwards visually
    ax.set_xlim(-wl_x, wl_x)
    ax.set_ylim(T * 1.25, -T * 0.15)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title or shape["label"], fontsize=10)
    fig.tight_layout(pad=0.2)
    return fig


# -----------------------------------------------------------------------------
# Result renderer (KPI cards + heatmaps) — unchanged
# -----------------------------------------------------------------------------
def render_result(s: Scenario, strip: np.ndarray, m: dict,
                  core_box: tuple[int, int, int, int],
                  container, vmax_shared: float | None = None,
                  label: str = "") -> None:
    cy0, cy1, cx0, cx1 = core_box

    c1, c2, c3, c4 = container.columns(4)
    c1.metric(f"{label}Cleaned area", f"{m.get('cleaned_pct', 0):.1f} %",
              help=f"Cells that clear the intensity gate AND get "
                   f"≥ {s.min_passes} passes. The headline cleaning KPI.")
    c2.metric(f"{label}Delivered pressure",
              f"{m.get('stagnation_pressure_bar', 0):.0f} bar",
              delta="clears gate" if m.get("intensity_ok") else "below gate",
              delta_color="normal" if m.get("intensity_ok") else "inverse",
              help=f"Jet stagnation pressure at the hull vs the "
                   f"{s.removal_pressure_bar:.0f} bar removal threshold. "
                   "If below, nothing cleans regardless of passes.")
    c3.metric(f"{label}Median passes", f"{m.get('median_passes', 0):.1f}",
              help="Typical number of nozzle passes per cell over the region.")
    c4.metric(f"{label}Untouched", f"{m['missed_pct']:.2f} %")

    if not m.get("intensity_ok"):
        container.error(
            f"Intensity gate NOT met: the jet delivers only "
            f"{m.get('stagnation_pressure_bar', 0):.0f} bar at the hull, "
            f"below the {s.removal_pressure_bar:.0f} bar needed to lift this "
            "fouling — so cleaned area is 0% no matter how many passes. "
            "Raise nozzle pressure, shorten the standoff, or widen the "
            "nozzle exit to deliver more pressure to the hull.")

    mode = "steady-state core" if s.steady_state_only else "full strip"
    container.caption(
        f"KPIs over {mode}. Δt = {m['dt_ms']:.2f} ms, {m['n_steps']} steps. "
        f"Array span (yaw {s.yaw_deg:+.0f}°) = "
        f"{m['array_span_x_mm']:.0f} mm across × {m['array_span_y_mm']:.0f} mm along. "
        f"Impact ring r = {s.impact_radius_mm():.1f} mm. "
        f"Jet advances {m.get('arc_per_step_mm', 0):.1f} mm/step along its ring "
        f"(footprint {m.get('footprint_mm', 0):.1f} mm)."
    )
    if m.get("undersampled"):
        container.warning(
            f"Time-step undersampled: the jet advances "
            f"{m.get('arc_per_step_mm', 0):.1f} mm per step along its impact "
            f"ring — wider than the {m.get('footprint_mm', 0):.1f} mm "
            "footprint — so the swept track breaks into a dotted line and "
            "the per-cell exposure (median/peak) is undercounted. Total "
            "deposited energy is still conserved. Lower the RPM or the "
            "simulated strip length so the step budget can resolve the "
            "path, or accept that point KPIs read low here.")

    vmax = vmax_shared if vmax_shared is not None else max(strip.max(), 1e-6)
    array_span_x = m["array_span_x_mm"]
    cell = s.cell_size_mm
    extent = [-array_span_x / 2, array_span_x / 2, strip.shape[0] * cell, 0]

    fig, ax = plt.subplots(figsize=(8, 3.6))
    im = ax.imshow(strip, extent=extent, aspect="auto",
                   cmap="viridis", vmin=0, vmax=vmax)
    if s.steady_state_only:
        core_y0_mm = cy0 * cell
        core_y1_mm = cy1 * cell
        ax.add_patch(mpatches.Rectangle(
            (extent[0], 0), extent[1] - extent[0], core_y0_mm,
            facecolor="white", alpha=0.45, edgecolor="none"))
        ax.add_patch(mpatches.Rectangle(
            (extent[0], core_y1_mm), extent[1] - extent[0],
            strip.shape[0] * cell - core_y1_mm,
            facecolor="white", alpha=0.45, edgecolor="none"))
        ax.axhline(core_y0_mm, color="red", lw=0.8, ls="--")
        ax.axhline(core_y1_mm, color="red", lw=0.8, ls="--")
    ax.set_xlabel("Across-track (mm)")
    ax.set_ylabel("Along-track (mm)")
    ax.set_title(f"{label}Exposure heatmap (bar*s per cell)")
    plt.colorbar(im, ax=ax, label="bar*s")
    container.pyplot(fig, clear_figure=True)

    region = strip[cy0:cy1, cx0:cx1] if s.steady_state_only else strip
    region_extent_y = (cy1 - cy0) * cell if s.steady_state_only \
        else strip.shape[0] * cell
    fig_t, ax_t = plt.subplots(figsize=(8, 2.8))
    ax_t.imshow(
        (region >= s.clean_threshold).astype(np.float32),
        extent=[-array_span_x / 2, array_span_x / 2, region_extent_y, 0],
        aspect="auto", cmap="Greens", vmin=0, vmax=1,
    )
    ax_t.set_xlabel("Across-track (mm)")
    ax_t.set_ylabel("Along-track (mm)")
    ax_t.set_title(f"{label}Cleaned cells (green = >= {s.clean_threshold:.1f} bar*s)")
    container.pyplot(fig_t, clear_figure=True)


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Mode")
    compare_mode = st.toggle("Compare two scenarios", value=False)

# -----------------------------------------------------------------------------
# Main layout
# -----------------------------------------------------------------------------
if not compare_mode:
    with st.sidebar:
        st.divider()
        scen = scenario_controls("single", Scenario(), st.sidebar)

    tab_impact, tab_motion, tab_hull = st.tabs(
        ["Impact simulation", "Motion simulation", "Hull simulation"])

    # =============================================================
    # TAB 1 — Impact simulation
    # =============================================================
    with tab_impact:
        left, right = st.columns([1.2, 1.0])
        with left:
            st.subheader("Top-down view")
            st.pyplot(plot_topdown(scen), clear_figure=False)
        with right:
            st.subheader("Side view (one disc)")
            st.pyplot(plot_side(scen), clear_figure=False)
            st.caption(
                "Zoomed spray profile — the same jet at true mm scale "
                "(the side view above is dominated by the disc width).")
            st.pyplot(plot_spray_profile(scen), clear_figure=False)

        st.divider()
        st.subheader("Single-disc coverage — ring gap check")
        st.caption(
            "Does one disc lay down a continuous swept ring as it advances, "
            "or do the rings separate into gaps? Green = swept by this disc. "
            "A single disc only cleans its **annulus** (ring), not a filled "
            "circle — the swath is completed by adjacent discs and rows.")
        fig_cov, cov = plot_single_disc_coverage(scen)
        cov_l, cov_r = st.columns([1.0, 1.0])
        with cov_l:
            st.pyplot(fig_cov, clear_figure=True)
        with cov_r:
            k1, k2 = st.columns(2)
            k1.metric("Forward advance / rev", f"{cov['advance_per_rev_mm']:.1f} mm")
            k2.metric("Effective pitch",
                      f"{cov['eff_pitch_mm']:.1f} mm",
                      help=f"= advance/rev ÷ {cov['n_nozzles']} nozzles. The "
                           "along-track spacing between successive nozzle "
                           "passes over a fixed point.")
            k3, k4 = st.columns(2)
            k3.metric("Footprint", f"{cov['footprint_mm']:.1f} mm")
            _m = cov["overlap_margin_mm"]
            k4.metric("Overlap margin",
                      f"{_m:+.1f} mm",
                      delta="overlap" if _m >= 0 else "gap",
                      delta_color="normal" if _m >= 0 else "inverse",
                      help="footprint − effective pitch. Positive = "
                           "successive passes overlap (gap-free); negative = "
                           "a gap of this width opens between rings.")
            if cov["overlap"]:
                st.success(
                    f"Gap-free along-track: effective pitch "
                    f"{cov['eff_pitch_mm']:.1f} mm < footprint "
                    f"{cov['footprint_mm']:.1f} mm, so successive passes "
                    "overlap. No 'circles' marching forward.")
            else:
                st.warning(
                    f"Gap risk: effective pitch {cov['eff_pitch_mm']:.1f} mm "
                    f"≥ footprint {cov['footprint_mm']:.1f} mm — successive "
                    "passes separate into discrete rings. Slow the traverse, "
                    "add nozzles, or raise RPM to close the gap.")

        if scen.footprint_mode.startswith("Physical jet"):
            st.subheader("Impact zone vs nozzle exit & standoff")
            st.caption(
                "The impact-zone diameter is "
                f"**{scen.footprint_dia():.1f} mm** = "
                f"{scen.nozzle_exit_mm:.1f} mm nozzle exit + "
                f"2 × {scen.standoff_mm} mm standoff × "
                f"tan({scen.jet_spread_deg:.1f}°). A larger standoff or "
                "spread angle widens the footprint and dilutes the "
                "cleaning energy (∝ 1/area); a tighter exit or shorter "
                "standoff concentrates it.")
            st.pyplot(plot_footprint_sensitivity(scen), clear_figure=False)

        st.divider()
        if st.button("Run full simulation", type="primary"):
            with st.spinner("Sweeping the array across the hull…"):
                strip, metrics, core_box = simulate_pressure(scen)
            # Cache result so the Hull tab can read cleaning-rate KPIs.
            st.session_state["last_impact_result"] = {
                "scen_key": scenario_full_key(scen),
                "strip": strip,
                "metrics": metrics,
                "core_box": core_box,
                "rov_speed_kn": scen.rov_speed_kn,
                "array_span_x_mm": metrics["array_span_x_mm"],
                "array_span_y_mm": metrics["array_span_y_mm"],
                # Hull-tab cleaning rate uses the physically-gated cleaned area.
                "coverage_pct": metrics["cleaned_pct"],
            }
            render_result(scen, strip, metrics, core_box, st)
        else:
            cached_impact = st.session_state.get("last_impact_result")
            if (cached_impact and
                    cached_impact.get("scen_key") == scenario_full_key(scen)):
                st.caption(
                    "Showing cached result for current scenario. Click "
                    "**Run full simulation** to re-run.")
                render_result(scen, cached_impact["strip"],
                              cached_impact["metrics"],
                              cached_impact["core_box"], st)
            else:
                st.info(
                    "Adjust parameters in the sidebar, then click "
                    "**Run full simulation**. The cleaned-area KPI here "
                    "feeds the Hull simulation tab.")

    # =============================================================
    # TAB 2 — Motion simulation
    # =============================================================
    with tab_motion:
        rov_speed_mm_s = scen.rov_speed_kn * KNOTS_TO_MPS * 1000.0
        total_time_s = scen.sim_length_mm / max(rov_speed_mm_s, 1e-6)
        total_ms = int(total_time_s * 1000)

        # Use session_state so Play button and slider share t
        if "t_ms" not in st.session_state:
            st.session_state.t_ms = total_ms // 4

        mc1, mc2, mc3 = st.columns([1.6, 1.0, 1.0])
        with mc1:
            t_ms = st.slider(
                "Snapshot time (ms)",
                min_value=0,
                max_value=total_ms,
                value=int(st.session_state.t_ms),
                step=max(1, total_ms // 400),
                key="t_ms_slider",
                help="Scrub through the traversal to see the state at any instant.")
            st.session_state.t_ms = t_ms
        with mc2:
            trail_rev = st.slider(
                "Trail length (disc revolutions)",
                0.0, 20.0, 2.0, step=0.25,
                key="trail_rev_slider",
                help="How far back in time to draw each nozzle's path.")
        with mc3:
            frame = st.radio(
                "Reference frame",
                ["Hull frame", "ROV frame"],
                index=0,
                key="frame_radio",
                help="Hull frame: array translates, hull is stationary. "
                     "ROV frame: array is stationary; trails look like rosettes.")

        # ----- Animation: Prepare → Play pattern ---------------------
        # Render all frames ONCE into PNG bytes cached in session_state,
        # then play them back via st.image() for instant, shift-free
        # playback. Hull-frame axes are fixed over the full traversal.
        pc1, pc2, pc3, pc4 = st.columns([1.2, 1.2, 1.2, 1.3])
        play_frames = pc1.slider(
            "Animation frames", 20, 500, 120, step=10, key="play_frames",
            help="More frames = smoother rotation, but slower to prepare. "
                 "Rule of thumb: aim for ≤ 0.1 disc revolution per frame "
                 "(see strobo warning below).")
        play_speed = pc2.select_slider(
            "Speed", options=["0.25x", "0.5x", "1x", "2x", "4x"],
            value="1x", key="play_speed",
            help="Wall-clock speed multiplier. 1x = one frame every 60 ms.")
        prepare_underlay = pc3.checkbox(
            "Underlay during prepare", value=False, key="prepare_underlay",
            help="Include cumulative cleaning heatmap on each frame. "
                 "Much slower to prepare — only for short segments.")
        clear_anim = pc4.button(
            "Clear animation cache", key="clear_anim_btn",
            help="Remove cached frames (e.g. after changing the scenario).")
        if clear_anim:
            st.session_state.pop("anim_frames", None)

        # Segment selector: animate only a sub-window of the traversal.
        # Useful for smooth rotation — a 500 ms window at 120 frames gives
        # 4.2 ms per frame (0.04 rev at 800 RPM), which spins smoothly.
        sg1, sg2 = st.columns([2, 2])
        seg_start_ms = sg1.slider(
            "Segment start (ms)", 0, total_ms,
            min(int(st.session_state.get("t_ms", total_ms // 4)), total_ms - 100),
            step=max(1, total_ms // 400), key="seg_start_ms",
            help="Animate only from this time onwards.")
        seg_dur_ms = sg2.slider(
            "Segment duration (ms)",
            100, total_ms,
            min(1000, total_ms), step=max(1, total_ms // 100),
            key="seg_dur_ms",
            help="How much of the traversal to animate. Shorter = smoother "
                 "spin for the same frame count.")
        seg_end_ms = min(seg_start_ms + seg_dur_ms, total_ms)

        ac1, ac2, ac3 = st.columns([1.2, 1.2, 2.4])
        prepare = ac1.button(
            "🎞 Prepare animation", key="prepare_btn",
            help="Pre-render all frames into memory. Playback is then instant.")
        play_cached = ac2.button(
            "▶ Play", key="play_btn",
            help="Play the cached frames. Prepare first if no cache exists.")
        stop = ac3.button(
            "■ Stop", key="stop_btn",
            help="Interrupt preparation or playback.")
        if stop:
            st.session_state["stop_play"] = True

        # Build cache key so cache invalidates when any input changes
        cache_key = (
            scenario_full_key(scen),
            round(float(trail_rev), 4),
            frame,
            int(play_frames),
            bool(prepare_underlay),
            int(seg_start_ms),
            int(seg_end_ms),
        )
        cached_anim = st.session_state.get("anim_frames")
        if cached_anim and cached_anim.get("key") != cache_key:
            st.session_state.pop("anim_frames", None)
            cached_anim = None

        # ----- Underlay (static, opt-in) -----------------------------
        bc1, bc2 = st.columns([1, 2])
        compute_underlay = bc1.button(
            "Compute underlay",
            key="compute_underlay_btn",
            help="Run the pressure sim up to the current time and cache it. "
                 "Doesn't re-run when the slider moves (use this button again "
                 "for a new time).")
        if bc2.checkbox(
            "Show cached underlay (hull frame only)",
            value=False, key="show_underlay_cb",
            help="Toggle the cached heatmap on/off without re-running."):
            show_underlay = True
        else:
            show_underlay = False

        if compute_underlay:
            with st.spinner(f"Running cumulative simulation to t = {t_ms} ms…"):
                cum, _m, _c = simulate_pressure(scen, t_stop_s=t_ms / 1000.0)
                st.session_state["underlay"] = {
                    "data": cum,
                    "t_ms": t_ms,
                    "scen_key": scenario_full_key(scen),
                }

        cumulative = None
        cached = st.session_state.get("underlay")
        # Invalidate cache if scenario geometry/operating point changed
        if cached and cached.get("scen_key") != scenario_full_key(scen):
            cached = None
            st.session_state.pop("underlay", None)
        if show_underlay and frame == "Hull frame" and cached is not None:
            cumulative = cached["data"]
            st.caption(
                f"Showing cached underlay computed at t = {cached['t_ms']} ms "
                f"(re-click *Compute underlay* after moving the slider)."
            )

        # ----- Scale info --------------------------------------------
        # The animation covers total_time_s of real time, divided into
        # play_frames steps. Each frame represents total_time_s/frames of
        # reality. Wall-clock playback rate is governed by frame_wall_dt.
        speed_map = {"0.25x": 0.25, "0.5x": 0.5, "1x": 1.0,
                     "2x": 2.0, "4x": 4.0}
        speed = speed_map[play_speed]
        # Baseline: aim for ~60 ms per frame on screen at 1x.
        # Streamlit coalesces DOM updates below ~30 ms, so we enforce a
        # minimum dwell — otherwise at 4x playback looks like an
        # instantaneous jump from start to finish.
        baseline_wall_dt = 0.060
        frame_wall_dt = max(baseline_wall_dt / speed, 0.030)
        seg_dur_s = max((seg_end_ms - seg_start_ms) / 1000.0, 1e-9)
        real_dt_per_frame = seg_dur_s / max(play_frames, 1)
        # Ratio of wall time to real time for the same animation segment.
        #   >1  → animation runs SLOWER than real life
        #   <1  → animation runs FASTER than real life
        ratio = frame_wall_dt / max(real_dt_per_frame, 1e-9)
        if ratio >= 1.0:
            scale_word = f"~{ratio:.1f}× slower than real time"
        else:
            scale_word = f"~{1.0/ratio:.1f}× faster than real time"
        disc_period_ms = 60000.0 / max(scen.rpm, 1)
        rev_per_frame = real_dt_per_frame * 1000.0 / disc_period_ms
        # Aliasing warning if each frame advances by ≥ half a disc revolution
        alias_note = ""
        if rev_per_frame > 0.5:
            alias_note = (f" ⚠ Each frame advances by {rev_per_frame:.2f} "
                          f"disc revolutions — the rotation will look "
                          f"stroboscopic. Increase frames or shorten the "
                          f"traversal for smooth spin.")
        st.caption(
            f"⏱ Playback scale at {play_speed}: **{scale_word}** "
            f"(each frame = {real_dt_per_frame*1000:.1f} ms of reality, "
            f"shown for {frame_wall_dt*1000:.0f} ms on screen). "
            f"Discs spin at {scen.rpm} RPM (one rev every "
            f"{disc_period_ms:.1f} ms)." + alias_note
        )

        # ----- Render current snapshot (always shown) ---------------
        # Render as PNG via the same pipeline used for playback, so the
        # image size matches the cached frames exactly (no "drawing
        # shrinks when you click Play" surprise).
        snapshot_slot = st.empty()
        _init_fig = plot_motion_fast(
            scen, t_ms / 1000.0,
            trail_revolutions=trail_rev,
            frame=frame,
            cumulative_strip=cumulative,
            # If an animation is cached, use its fixed axes so the
            # snapshot and the movie share the same camera; otherwise
            # let the snapshot auto-fit.
            fixed_xlim=(cached_anim["xlim"] if cached_anim else None),
            fixed_ylim=(cached_anim["ylim"] if cached_anim else None),
        )
        _init_buf = io.BytesIO()
        _init_fig.savefig(_init_buf, format="png", dpi=110)
        plt.close(_init_fig)
        snapshot_slot.image(_init_buf.getvalue(), use_container_width=True)

        # ----- Prepare phase ----------------------------------------
        # Strategy: render every frame as an RGBA PIL Image, then
        # assemble them into a single animated GIF held as bytes in
        # session_state. Playback is a single st.image(gif_bytes) call —
        # the browser handles frame timing natively via the GIF's
        # embedded per-frame duration, so we get smooth animation
        # without Streamlit's websocket round-trip per frame (which was
        # causing the "jumps from start to end" behaviour in the
        # previous per-frame loop).
        if prepare:
            st.session_state["stop_play"] = False
            fx, fy = full_traversal_limits(
                scen, trail_rev, frame,
                t_start_s=seg_start_ms / 1000.0,
                t_end_s=seg_end_ms / 1000.0)
            times_ms_arr = np.linspace(seg_start_ms, seg_end_ms,
                                       int(play_frames)).astype(int)

            pil_frames: list = []
            progress = st.progress(0.0, text="Preparing animation…")
            interrupted = False
            t_prep_start = time.perf_counter()
            for i, tm in enumerate(times_ms_arr):
                if st.session_state.get("stop_play"):
                    interrupted = True
                    break
                t_sec = tm / 1000.0
                cum_f = None
                if prepare_underlay and frame == "Hull frame":
                    cum_f, _m, _c = simulate_pressure(
                        scen, t_stop_s=max(t_sec, 1e-6))
                fig_f = plot_motion_fast(
                    scen, t_sec,
                    trail_revolutions=trail_rev,
                    frame=frame,
                    cumulative_strip=cum_f,
                    fixed_xlim=fx, fixed_ylim=fy)
                buf = io.BytesIO()
                # No bbox_inches="tight" — cropping varies per frame and
                # would make the canvas shrink/jump during playback.
                fig_f.savefig(buf, format="png", dpi=110)
                plt.close(fig_f)
                buf.seek(0)
                # Convert to palette-mode (P) with adaptive palette —
                # GIF only supports 256 colours per frame, but we get
                # acceptable quality for line art + trails.
                img = Image.open(buf).convert("RGB").convert(
                    "P", palette=Image.ADAPTIVE, colors=192)
                pil_frames.append(img)
                progress.progress(
                    (i + 1) / len(times_ms_arr),
                    text=f"Preparing animation… {i+1}/{len(times_ms_arr)}")
            prep_wall = time.perf_counter() - t_prep_start
            progress.empty()

            if not interrupted and pil_frames:
                # Assemble GIF. Per-frame duration in ms is
                # frame_wall_dt × 1000, clamped to >= 20 ms (the GIF
                # spec's lower bound — browsers override anything below
                # ~20 ms to 100 ms).
                gif_frame_ms = max(int(round(frame_wall_dt * 1000)), 20)
                gif_buf = io.BytesIO()
                pil_frames[0].save(
                    gif_buf,
                    format="GIF",
                    save_all=True,
                    append_images=pil_frames[1:],
                    duration=gif_frame_ms,
                    loop=0,             # 0 = loop forever
                    disposal=2,         # clear each frame before next
                    optimize=False,
                )
                gif_bytes = gif_buf.getvalue()

                st.session_state["anim_frames"] = {
                    "key": cache_key,
                    "gif": gif_bytes,
                    "n_frames": len(pil_frames),
                    "gif_frame_ms": gif_frame_ms,
                    "times_ms": times_ms_arr.tolist(),
                    "total_time_s": total_time_s,
                    "prep_wall_s": prep_wall,
                    "xlim": fx,
                    "ylim": fy,
                }
                st.success(
                    f"Cached GIF with {len(pil_frames)} frames "
                    f"({len(gif_bytes)/1024:.0f} kB) in {prep_wall:.1f} s. "
                    f"Click ▶ Play.")
            elif interrupted:
                st.warning(
                    f"Preparation stopped after {len(pil_frames)} frames.")
            st.session_state["stop_play"] = False

        # ----- Playback phase ---------------------------------------
        # The GIF loops natively in the browser, so "Play" just means
        # "(re-)display the cached GIF". No per-frame Python loop, no
        # websocket per frame → no start/end jumping.
        cached_anim = st.session_state.get("anim_frames")
        if play_cached:
            if not cached_anim or cached_anim.get("key") != cache_key:
                st.warning(
                    "No cached animation matches the current settings. "
                    "Click *Prepare animation* first.")
            elif "gif" in cached_anim:
                snapshot_slot.image(
                    cached_anim["gif"], use_container_width=True)
                # Leave slider at the final frame
                if cached_anim.get("times_ms"):
                    st.session_state.t_ms = int(cached_anim["times_ms"][-1])

        # Always re-display the cached GIF if it exists and matches —
        # this way the animation persists across reruns (e.g. when
        # other widgets are touched), not only when Play is clicked.
        if (cached_anim and cached_anim.get("key") == cache_key
                and "gif" in cached_anim and not play_cached):
            snapshot_slot.image(
                cached_anim["gif"], use_container_width=True)

        if cached_anim and cached_anim.get("key") == cache_key:
            st.caption(
                f"✅ Animation cached: {cached_anim['n_frames']} frames "
                f"as GIF ({len(cached_anim['gif'])/1024:.0f} kB, "
                f"{cached_anim['gif_frame_ms']} ms/frame). "
                f"Prep took {cached_anim.get('prep_wall_s', 0):.1f} s.")

        st.caption(
            f"Total traversal time: {total_time_s*1000:.0f} ms "
            f"({total_time_s:.2f} s). "
            f"Disc period at {scen.rpm} RPM: "
            f"{60000/max(scen.rpm,1):.1f} ms/rev. "
            f"At 20 rev, trail covers {20*60000/max(scen.rpm,1):.0f} ms."
        )

    # =============================================================
    # TAB 3 — Hull simulation
    # =============================================================
    with tab_hull:
        st.subheader("Vessel-level cleaning time estimate")
        st.caption(
            "Given the Impact-tab cleaning-rate and a hull geometry, "
            "estimate how long it takes to clean each side of the vessel.")

        cached_impact = st.session_state.get("last_impact_result")
        if not cached_impact or cached_impact.get("scen_key") != scenario_full_key(scen):
            st.warning(
                "⚠ No impact-sim result cached for the current scenario. "
                "Go to the **Impact simulation** tab and click "
                "**Run full simulation** first — the cleaned-area KPI is "
                "used here to compute the per-side time.")
            st.stop()

        # ----- Vessel inputs --------------------------------------
        st.markdown("**Vessel particulars**")
        vin_c1, vin_c2, vin_c3, vin_c4 = st.columns(4)
        loa_m = vin_c1.number_input(
            "LOA (m)", min_value=20.0, max_value=450.0, value=200.0,
            step=5.0, key="hull_loa",
            help="Length overall of the vessel.")
        beam_m = vin_c2.number_input(
            "Beam (m)", min_value=4.0, max_value=75.0, value=32.0,
            step=1.0, key="hull_beam",
            help="Maximum breadth at the waterline.")
        draft_m = vin_c3.number_input(
            "Draft (m)", min_value=1.0, max_value=25.0, value=12.0,
            step=0.5, key="hull_draft",
            help="Vertical distance from waterline to keel.")
        track_overlap_pct = vin_c4.slider(
            "Track overlap (%)", min_value=0, max_value=50, value=15,
            step=1, key="hull_track_overlap",
            help="Fraction of the array width that overlaps the previous "
                 "track to avoid missed bands at string boundaries. "
                 "Reduces effective cleaning rate by (1 − overlap). "
                 "Typical field practice: 10–25%.")

        # ----- Cleaning-rate derivation ---------------------------
        # Area cleaned per unit wall time =
        #     array_effective_width [m]  ×  ROV speed [m/s]
        #         ×  coverage_fraction × (1 − overlap_fraction)
        # where coverage is the fraction of cells within the swept strip
        # that exceed the clean threshold (from the Impact sim) and
        # overlap accounts for deliberate re-sweeping at string edges.
        array_span_x_mm = float(cached_impact["array_span_x_mm"])
        coverage_frac = float(cached_impact["coverage_pct"]) / 100.0
        rov_speed_mps = float(cached_impact["rov_speed_kn"]) * KNOTS_TO_MPS
        overlap_frac = track_overlap_pct / 100.0
        # Effective width: the swept strip's across-track extent.
        array_width_m = array_span_x_mm / 1000.0
        effective_width_m = array_width_m * (1.0 - overlap_frac)
        # Gross footprint swept per second, m²/s
        footprint_rate_m2_s = array_width_m * rov_speed_mps
        # Effective *cleaned* rate = footprint × coverage × (1 − overlap)
        cleaning_rate_m2_s = (footprint_rate_m2_s
                              * coverage_frac
                              * (1.0 - overlap_frac))
        cleaning_rate_m2_h = cleaning_rate_m2_s * 3600.0
        # Treat anything below 1 m²/h as degenerate — avoids astronomical
        # times when coverage_frac or ROV speed is effectively zero.
        rate_valid = cleaning_rate_m2_h >= 1.0

        rate_c1, rate_c2, rate_c3, rate_c4 = st.columns(4)
        rate_c1.metric(
            "Array width",
            f"{array_width_m:.2f} m",
            help="Across-track span of the swept strip at the current yaw.")
        rate_c2.metric(
            "Effective width",
            f"{effective_width_m:.2f} m",
            help=f"= array width × (1 − {track_overlap_pct}% overlap)")
        rate_c3.metric(
            "ROV speed",
            f"{rov_speed_mps:.2f} m/s",
            help=f"= {cached_impact['rov_speed_kn']:.2f} kn")
        rate_c4.metric(
            "Cleaning rate",
            f"{cleaning_rate_m2_h:.0f} m²/h" if rate_valid else "—",
            help=(f"Footprint {footprint_rate_m2_s:.2f} m²/s × "
                  f"coverage {coverage_frac*100:.1f}% × "
                  f"(1 − {track_overlap_pct}% overlap)."))

        if not rate_valid:
            st.error(
                f"⚠ Cleaning rate is effectively zero "
                f"({cleaning_rate_m2_h:.2f} m²/h). This usually means the "
                f"Impact-tab coverage is 0% — either the scenario doesn't "
                f"clean anything above the threshold, or the Impact "
                f"simulation was run with a different scenario. "
                f"Re-run **Run full simulation** in the Impact tab before "
                f"reading the times below.")

        st.divider()

        # ----- Hull-shape selection (visual cards) -----------------
        st.markdown("**Midship cross-section**")
        shape_keys = ["full", "typical", "fine"]
        shape_cols = st.columns(3)
        for key, col in zip(shape_keys, shape_cols):
            with col:
                fig_h = plot_hull_section(key, beam_m, draft_m,
                                          title=HULL_SHAPES[key]["label"])
                st.pyplot(fig_h, clear_figure=True)
                st.caption(HULL_SHAPES[key]["blurb"])

        shape_labels = {k: HULL_SHAPES[k]["label"] for k in shape_keys}
        shape_key = st.radio(
            "Select hull shape",
            shape_keys,
            format_func=lambda k: shape_labels[k],
            horizontal=True,
            key="hull_shape_radio",
            help="The shape drives the midship perimeter and therefore "
                 "the bottom area.")

        st.divider()

        # ----- Area & time computation ----------------------------
        areas = hull_wetted_areas(loa_m, beam_m, draft_m, shape_key)

        def _fmt_duration(seconds: float) -> str:
            """Human-readable duration. Uses min / h / d depending on scale.
            Returns '—' for non-finite or negative values."""
            if not math.isfinite(seconds) or seconds < 0:
                return "—"
            minutes = seconds / 60.0
            if minutes < 60.0:
                return f"{minutes:.0f} min"
            hours = seconds / 3600.0
            if hours < 24.0:
                whole_h = int(hours)
                min_left = int(round((hours - whole_h) * 60))
                if min_left == 60:
                    whole_h += 1
                    min_left = 0
                return f"{whole_h} h {min_left:02d} min"
            days = hours / 24.0
            if days < 10.0:
                whole_d = int(days)
                h_left = int(round((days - whole_d) * 24))
                if h_left == 24:
                    whole_d += 1
                    h_left = 0
                return f"{whole_d} d {h_left:02d} h"
            return f"{days:.1f} d"

        if rate_valid:
            side_time_s = areas["side_port"] / cleaning_rate_m2_s
            bottom_time_s = areas["bottom"] / cleaning_rate_m2_s
            total_time_s_hull = (areas["side_port"] + areas["side_starboard"]
                                 + areas["bottom"]) / cleaning_rate_m2_s
        else:
            side_time_s = float("nan")
            bottom_time_s = float("nan")
            total_time_s_hull = float("nan")

        st.markdown("**Per-side wetted area and cleaning time**")
        out_c1, out_c2, out_c3, out_c4 = st.columns(4)
        out_c1.metric(
            "Port side",
            f"{areas['side_port']:.0f} m²",
            help=_fmt_duration(side_time_s))
        out_c2.metric(
            "Starboard side",
            f"{areas['side_starboard']:.0f} m²",
            help=_fmt_duration(side_time_s))
        out_c3.metric(
            "Bottom",
            f"{areas['bottom']:.0f} m²",
            help=_fmt_duration(bottom_time_s))
        out_c4.metric(
            "Total wetted area",
            f"{areas['total']:.0f} m²",
            help="Sum of both sides + bottom (excludes bow, stern, "
                 "superstructure, appendages).")

        time_c1, time_c2, time_c3, time_c4 = st.columns(4)
        time_c1.metric("Port side time", _fmt_duration(side_time_s))
        time_c2.metric("Starboard side time", _fmt_duration(side_time_s))
        time_c3.metric("Bottom time", _fmt_duration(bottom_time_s))
        time_c4.metric("**Total cleaning time**",
                       _fmt_duration(total_time_s_hull))

        st.caption(
            f"Midship section perimeter = {areas['section_perim_m']:.2f} m "
            f"(beam {beam_m:.1f} m, draft {draft_m:.1f} m, "
            f"shape **{HULL_SHAPES[shape_key]['label']}**). "
            f"Side area per side ≈ LOA × draft × k_side × C_L = "
            f"{loa_m:.0f} × {draft_m:.1f} × "
            f"{HULL_SHAPES[shape_key]['k_side']:.2f} × {areas['C_l']:.2f} "
            f"= {areas['side_port']:.0f} m². Bottom area ≈ LOA × "
            f"perimeter × C_L = {areas['bottom']:.0f} m². "
            f"Effective cleaning rate accounts for **{track_overlap_pct}% "
            f"track overlap**. Assumes continuous, uninterrupted cleaning "
            "— real-world operations also include docking, repositioning, "
            "and transit overhead.")

else:
    with st.sidebar:
        st.divider()
        st.caption("Configure both scenarios below, then click Run.")
    col_a_ctrl, col_b_ctrl = st.sidebar.columns(2)
    with col_a_ctrl.expander("Scenario A", expanded=True):
        scen_a = scenario_controls("A", Scenario(), col_a_ctrl)
    default_b = Scenario(rov_speed_kn=1.0, rpm=800, pressure_bar=250)
    with col_b_ctrl.expander("Scenario B", expanded=True):
        scen_b = scenario_controls("B", default_b, col_b_ctrl)

    top_a, top_b = st.columns(2)
    with top_a:
        st.subheader("Scenario A — layout")
        st.pyplot(plot_topdown(scen_a, "Scenario A"), clear_figure=False)
    with top_b:
        st.subheader("Scenario B — layout")
        st.pyplot(plot_topdown(scen_b, "Scenario B"), clear_figure=False)

    st.divider()
    if st.button("Run both simulations", type="primary"):
        with st.spinner("Running scenario A…"):
            strip_a, m_a, box_a = simulate_pressure(scen_a)
        with st.spinner("Running scenario B…"):
            strip_b, m_b, box_b = simulate_pressure(scen_b)

        vmax_shared = max(strip_a.max(), strip_b.max(), 1e-6)

        res_a, res_b = st.columns(2)
        with res_a:
            st.markdown("### Scenario A")
            render_result(scen_a, strip_a, m_a, box_a, st, vmax_shared=vmax_shared)
        with res_b:
            st.markdown("### Scenario B")
            render_result(scen_b, strip_b, m_b, box_b, st, vmax_shared=vmax_shared)

        st.subheader("Delta (B − A)")
        keys = [
            ("coverage_pct", "Cleaned area (%)", "+"),
            ("p50_bs", "Median exposure (bar*s)", "+"),
            ("p10_bs", "10th-pct exposure (bar*s)", "+"),
            ("mean_bs", "Mean exposure (bar*s)", "+"),
            ("missed_pct", "Untouched (%)", "-"),
        ]
        rows = []
        for key, label, better in keys:
            a = m_a[key]; b = m_b[key]; delta = b - a
            rows.append({
                "metric": label,
                "A": f"{a:.2f}",
                "B": f"{b:.2f}",
                "Δ (B−A)": f"{delta:+.2f}",
                "favourable direction": better,
            })
        st.table(rows)
    else:
        st.info("Configure scenarios A and B, then click **Run both simulations**.")


st.divider()
with st.expander("Modelling assumptions and units"):
    st.markdown(
        """
**Hull grid.** 1×1 cm cells. Each cell accumulates *integrated jet pressure
exposure* in **bar·seconds**.

**Motion visualisation.** Shows an instantaneous snapshot at time `t`.
Nozzle trails are the past `N` revolutions of each nozzle's trajectory
in the hull frame (so they look like curtate/prolate cycloids) or the
ROV frame (where they look like pure rosettes because the array
translation is factored out). Trails are drawn with a single
`LineCollection` for fast rendering, and current footprints with a
`PatchCollection`, so scrubbing the slider stays responsive even at
20 rev trail length.

**Play button.** Steps through the traversal as a flip-book (20–200
frames). The `Underlay during play` checkbox re-runs the pressure sim
at every frame and is therefore much slower — leave it off for smooth
playback.

**Cumulative underlay.** Opt-in: click *Compute underlay* to run the
pressure sim up to the current time; tick *Show cached underlay* to
display it. The cache is invalidated when scenario parameters change.

**Footprint diameter on hull.** Three modes:

- *Physical jet (default).* The impact zone is set by the spreading free
  jet: `d_fp = nozzle_exit_dia + 2 · standoff · tan(spread_half_angle)`.
  Both the **nozzle exit diameter** and the **distance to the hull
  (standoff)** drive the footprint directly. Because the deposited
  cleaning energy spreads over `π·(d_fp/2)²`, a wider footprint dilutes
  the bar·s per cell — the *Impact zone vs nozzle exit & standoff* chart
  visualises this trade-off, and the side cross-section draws the jet as
  a cone widening from the exit diameter to the hull footprint.
- *Linear with pressure (legacy).* 60 mm at 50 bar → 80 mm at 600 bar.
- *Manual override.* A fixed diameter.

**Cant correction.** Canted nozzles converge the impingement ring inward
by `standoff · tan(cant)`.

**Array yaw.** Rotates the disc array about its centroid relative to the
ROV travel direction. The ROV travel vector stays along +y in hull
coordinates.

**Steady-state core.** The first and last `array_span_y` mm of the
simulated strip are entry/exit transients. Turn on the checkbox to
exclude them from KPIs.

**Calibration tip.** Run a known operating point that produces a clean
hull, read the median bar·s on the core region, and use that as your
`clean_threshold` going forward.
        """
    )
