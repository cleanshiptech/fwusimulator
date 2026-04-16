"""
FWU Coverage Simulator
======================
Streamlit app to simulate hull-cleaning coverage of the C-Leanship ROV
flushing and washing unit (FWU).

Hull is discretised on a 1x1 cm grid. For each time step every nozzle
deposits its instantaneous jet pressure (bar) onto the cells inside its
footprint, multiplied by dt. The accumulated quantity per cell is therefore
**integrated pressure exposure** in units of bar-seconds (bar*s).

Features
--------
* Disc and nozzle geometry live in sidebar sliders.
* Array yaw angle: rotates the array relative to the ROV travel direction.
* Steady-state core: KPIs are measured only on the portion of the hull that
  the full array has swept over, so the entering/exiting transients are
  excluded from the "untouched area" figure.
* Single-scenario mode (one sidebar) and compare-scenarios mode (two full
  parameter sets A and B, shown side by side).
* Motion visualisation: scrub a time slider to see disc, nozzle and
  impact-spot positions with nozzle-trail streaks and optional cumulative
  cleaning heatmap. Choose between hull frame (ROV moving) and ROV frame
  (array stationary, hull scrolling past) for intuition.

Run with:
    streamlit run app.py
    # or, if the streamlit launcher is not on PATH:
    python -m streamlit run app.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from copy import deepcopy

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

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
    footprint_mode: str = "Linear with pressure (60->80 mm)"
    footprint_dia_mm_override: int = 70
    pressure_profile: str = "Gaussian (peak at centre)"

    # Hull strip & threshold
    sim_length_mm: int = 5000
    clean_threshold: float = 20.0

    # Post-processing
    steady_state_only: bool = True

    def footprint_dia(self) -> float:
        if self.footprint_mode == "Manual override":
            return float(self.footprint_dia_mm_override)
        return 60.0 + (self.pressure_bar - 50.0) / (600.0 - 50.0) * 20.0

    def impact_radius_mm(self) -> float:
        return max(
            0.0,
            self.nozzle_radius_mm
            - self.standoff_mm * math.tan(math.radians(self.nozzle_cant_deg)),
        )


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
    s.footprint_mode = container.radio(
        "Footprint diameter on hull",
        ["Linear with pressure (60->80 mm)", "Manual override"],
        index=0 if s.footprint_mode.startswith("Linear") else 1,
        key=k("footprint_mode"),
    )
    if s.footprint_mode == "Manual override":
        s.footprint_dia_mm_override = container.slider(
            "Footprint diameter (mm)", 20, 120, s.footprint_dia_mm_override,
            key=k("footprint_dia_mm_override"))
    else:
        container.caption(f"Footprint diameter on hull: **{s.footprint_dia():.1f} mm**")
    s.pressure_profile = container.radio(
        "Pressure distribution within footprint",
        ["Uniform", "Gaussian (peak at centre)"],
        index=0 if s.pressure_profile.startswith("Uniform") else 1,
        key=k("pressure_profile"),
    )

    container.subheader("Hull strip & KPIs")
    s.sim_length_mm = container.slider(
        "Hull strip length to simulate (mm)", 500, 10000, s.sim_length_mm, step=100,
        key=k("sim_length_mm"))
    s.clean_threshold = container.slider(
        "Cleaning threshold (bar*s)", 0.5, 200.0, float(s.clean_threshold), step=0.5,
        key=k("clean_threshold"),
        help="A cell counts as 'cleaned' when its integrated pressure "
             "exposure exceeds this value.")
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
# Top-down schematic
# -----------------------------------------------------------------------------
def plot_topdown(s: Scenario, title: str = "Top-down view") -> plt.Figure:
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


# -----------------------------------------------------------------------------
# Side cross-section
# -----------------------------------------------------------------------------
def plot_side(s: Scenario) -> plt.Figure:
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
    for sign in (-1, +1):
        nx = sign * s.nozzle_radius_mm
        end_x = nx - sign * s.standoff_mm * math.tan(math.radians(s.nozzle_cant_deg))
        ax.plot([nx, end_x], [s.standoff_mm, 0], color="#ff7f0e", lw=1.5)
        ax.add_patch(mpatches.Ellipse(
            (end_x, 0), s.footprint_dia(), 4,
            facecolor="#ff7f0e", alpha=0.4, edgecolor="none",
        ))
    ax.annotate("", xy=(-s.disc_diameter_mm / 2 - 20, 0),
                xytext=(-s.disc_diameter_mm / 2 - 20, s.standoff_mm),
                arrowprops=dict(arrowstyle="<->", color="#333"))
    ax.text(-s.disc_diameter_mm / 2 - 25, s.standoff_mm / 2,
            f"{s.standoff_mm} mm", ha="right", va="center", fontsize=8)
    ax.text(s.nozzle_radius_mm + 20, s.standoff_mm * 0.6,
            f"{s.nozzle_cant_deg}°", color="#ff7f0e", fontsize=9)
    ax.set_xlim(-s.disc_diameter_mm / 2 - 80, s.disc_diameter_mm / 2 + 80)
    ax.set_ylim(-15, s.standoff_mm + disc_h + 20)
    ax.set_aspect("equal")
    ax.set_xlabel("Across disc (mm)")
    ax.set_ylabel("Height (mm)")
    ax.set_title("Side cross-section")
    ax.grid(True, alpha=0.3)
    return fig


# -----------------------------------------------------------------------------
# Footprint stencil
# -----------------------------------------------------------------------------
def footprint_stencil(s: Scenario) -> np.ndarray:
    r_mm = s.footprint_dia() / 2.0
    r_cells = r_mm / CELL_SIZE_MM
    half = int(math.ceil(r_cells)) + 1
    yy, xx = np.ogrid[-half:half + 1, -half:half + 1]
    r2 = (xx ** 2 + yy ** 2).astype(np.float32)
    if s.pressure_profile.startswith("Uniform"):
        stencil = (r2 <= r_cells ** 2).astype(np.float32)
    else:
        sigma = r_cells / 2.0
        stencil = np.exp(-r2 / (2 * sigma ** 2)).astype(np.float32)
        stencil[r2 > (r_cells * 1.5) ** 2] = 0.0
    return stencil


# -----------------------------------------------------------------------------
# Core nozzle-position helper (used by both simulation and motion viz)
# -----------------------------------------------------------------------------
def nozzle_positions_hull(s: Scenario, t: float,
                          rotated: list[tuple[float, float, int]],
                          phases: list[float]
                          ) -> list[list[tuple[float, float]]]:
    """
    Return nozzle positions in the hull frame at time t.
    Output: list per disc -> list of (x, y) positions, one per nozzle.
    The array centroid is at (0, array_y_offset_init + rov_speed*t + row_pitch/2).
    """
    rov_speed_mm_s = s.rov_speed_kn * KNOTS_TO_MPS * 1000.0
    omega = s.rpm * 2 * math.pi / 60.0
    yaw_rad = math.radians(s.yaw_deg)
    cos_t, sin_t = math.cos(yaw_rad), math.sin(yaw_rad)
    ir = s.impact_radius_mm()

    # Same array placement as simulate_pressure: at t=0 the leading edge is at -margin.
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
# Pressure simulation (optionally stopping early for cumulative visualisation)
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

    nx = int(width_extent / CELL_SIZE_MM)
    ny = int(full_extent / CELL_SIZE_MM)
    grid = np.zeros((ny, nx), dtype=np.float32)

    dt_rot = math.radians(5.0) / max(omega_rad_s, 1e-6)
    dt_trav = 0.5 * CELL_SIZE_MM / max(rov_speed_mm_s, 1e-6)
    dt = min(dt_rot, dt_trav)
    total_time_full = s.sim_length_mm / max(rov_speed_mm_s, 1e-6)
    total_time = total_time_full if t_stop_s is None else min(t_stop_s, total_time_full)
    n_steps = int(total_time / dt) + 1
    n_steps = min(n_steps, 12000)
    dt = total_time / max(n_steps, 1)

    stencil = footprint_stencil(s)
    sh, sw = stencil.shape
    rr = sh // 2
    p_dt = float(s.pressure_bar) * dt
    weighted_stencil = stencil * p_dt

    x0 = -width_extent / 2
    array_leading_y = min(rys) - s.disc_diameter_mm / 2
    array_y_offset_init = -margin_mm - array_leading_y
    phases = [2 * math.pi * i / max(len(rotated), 1) for i in range(len(rotated))]

    for step in range(n_steps):
        t = step * dt
        array_y = array_y_offset_init + rov_speed_mm_s * t
        for di, (rx, ry, direction) in enumerate(rotated):
            phase = phases[di] + direction * omega_rad_s * t
            dcx = rx
            dcy = ry + array_y
            for kn in range(s.n_nozzles):
                theta = phase + 2 * math.pi * kn / s.n_nozzles
                ir = s.impact_radius_mm()
                lx = ir * math.cos(theta)
                ly = ir * math.sin(theta)
                gx = lx * cos_t - ly * sin_t
                gy = lx * sin_t + ly * cos_t
                nx_mm = dcx + gx
                ny_mm = dcy + gy
                ix = int(round((nx_mm - x0) / CELL_SIZE_MM))
                iy = int(round((ny_mm - 0.0) / CELL_SIZE_MM))
                xs_ = ix - rr; ys_ = iy - rr
                xe_ = xs_ + sw; ye_ = ys_ + sh
                gx0 = max(xs_, 0); gy0 = max(ys_, 0)
                gx1 = min(xe_, nx); gy1 = min(ye_, ny)
                if gx1 <= gx0 or gy1 <= gy0:
                    continue
                mx0 = gx0 - xs_; my0 = gy0 - ys_
                mx1 = mx0 + (gx1 - gx0); my1 = my0 + (gy1 - gy0)
                grid[gy0:gy1, gx0:gx1] += weighted_stencil[my0:my1, mx0:mx1]

    y_strip_start = 0
    y_strip_end = int((s.sim_length_mm + array_span_y) / CELL_SIZE_MM)
    x_strip_start = int(margin_mm / CELL_SIZE_MM)
    x_strip_end = int((margin_mm + array_span_x) / CELL_SIZE_MM)
    strip = grid[y_strip_start:y_strip_end, x_strip_start:x_strip_end]

    core_y0 = int(array_span_y / CELL_SIZE_MM)
    core_y1 = int(s.sim_length_mm / CELL_SIZE_MM)
    if core_y1 <= core_y0:
        core_y0, core_y1 = 0, strip.shape[0]
    core_x0 = 0
    core_x1 = strip.shape[1]
    core_box = (core_y0, core_y1, core_x0, core_x1)

    if s.steady_state_only:
        region = strip[core_y0:core_y1, core_x0:core_x1]
    else:
        region = strip

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
        "coverage_pct": float((region >= s.clean_threshold).mean() * 100.0)
                       if region.size else 0.0,
        "missed_pct": float((region == 0).mean() * 100.0) if region.size else 0.0,
        "region_area_mm2": float(region.size * CELL_SIZE_MM * CELL_SIZE_MM),
        "total_time_s": total_time_full,
    }
    return strip, metrics, core_box


# -----------------------------------------------------------------------------
# Motion visualisation: disc + nozzle snapshot with cycloid trails
# -----------------------------------------------------------------------------
def plot_motion(s: Scenario, t_now_s: float,
                trail_revolutions: float = 1.0,
                frame: str = "Hull frame",
                cumulative_strip: np.ndarray | None = None) -> plt.Figure:
    """
    Render a snapshot of the array at time t_now_s, with nozzle-position
    trails extending backward `trail_revolutions` disc revolutions in time.

    frame = "Hull frame": hull is stationary, array moves upward (+y).
    frame = "ROV frame":  array is stationary; hull scrolls downward.
    """
    rotated = compute_rotated_discs(s)
    phases = [2 * math.pi * i / max(len(rotated), 1) for i in range(len(rotated))]

    omega = s.rpm * 2 * math.pi / 60.0
    trail_duration_s = trail_revolutions * (2 * math.pi / max(omega, 1e-6))
    t_start = max(0.0, t_now_s - trail_duration_s)
    rov_speed_mm_s = s.rov_speed_kn * KNOTS_TO_MPS * 1000.0

    # Sample the trail densely enough to render smoothly
    trail_n = max(30, int(trail_revolutions * 90))
    ts = np.linspace(t_start, t_now_s, trail_n) if trail_duration_s > 0 else np.array([t_now_s])

    # Compute trails in HULL frame: call nozzle_positions_hull once per
    # time sample and unpack into [disc][nozzle] lists of (x, y).
    trails_by_disc = [[[] for _ in range(s.n_nozzles)] for _ in rotated]
    for t in ts:
        per_disc = nozzle_positions_hull(s, float(t), rotated, phases)
        for di in range(len(rotated)):
            for kn in range(s.n_nozzles):
                trails_by_disc[di][kn].append(per_disc[di][kn])

    # Current disc centres
    centres_now = disc_centres_hull(s, t_now_s, rotated)
    nozzles_now = nozzle_positions_hull(s, t_now_s, rotated, phases)

    # Determine plot extents. In ROV frame we subtract array_y from every y coord.
    if frame == "ROV frame":
        # Compute array_y at t_now_s
        rys_rot = [r[1] for r in rotated]
        array_leading_y = min(rys_rot) - s.disc_diameter_mm / 2
        margin_mm = 120
        array_y_offset_init = -margin_mm - array_leading_y
        array_y_now = array_y_offset_init + rov_speed_mm_s * t_now_s
        y_shift = -array_y_now
    else:
        y_shift = 0.0

    def shift_pt(xy):
        return xy[0], xy[1] + y_shift

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Cumulative heatmap underlay (hull frame only — not sensible in ROV frame)
    if cumulative_strip is not None and frame == "Hull frame":
        # The strip starts at y=0 (in hull coords) and spans strip.shape[0]*10 mm
        array_span_x = (max(r[0] for r in rotated) - min(r[0] for r in rotated)
                        + s.disc_diameter_mm)
        extent = [-array_span_x / 2, array_span_x / 2,
                  cumulative_strip.shape[0] * CELL_SIZE_MM, 0]
        vmax = max(float(cumulative_strip.max()), 1e-6)
        ax.imshow(cumulative_strip, extent=extent, aspect="auto",
                  cmap="viridis", vmin=0, vmax=vmax, alpha=0.7, zorder=0)

    # Draw disc outlines at current time
    impact_r = s.impact_radius_mm()
    fp_d = s.footprint_dia()
    disc_xs, disc_ys = [], []
    for (cx_d, cy_d, _) in centres_now:
        cx_s, cy_s = shift_pt((cx_d, cy_d))
        disc_xs.append(cx_s); disc_ys.append(cy_s)
        ax.add_patch(mpatches.Circle((cx_s, cy_s), s.disc_diameter_mm / 2,
                                     fill=False, edgecolor="#1f77b4",
                                     linewidth=1.2, zorder=3))
        ax.add_patch(mpatches.Circle((cx_s, cy_s), impact_r,
                                     fill=False, linestyle="--",
                                     edgecolor="#888", linewidth=0.6, zorder=3))

    # Draw trails + current nozzle footprints
    for di in range(len(rotated)):
        for kn in range(s.n_nozzles):
            trail = trails_by_disc[di][kn]
            if len(trail) >= 2:
                tx = [p[0] for p in trail]
                ty = [p[1] + y_shift for p in trail]
                # Fade by using alpha gradient (draw segments)
                for i in range(len(tx) - 1):
                    frac = i / max(len(tx) - 1, 1)
                    alpha = 0.15 + 0.55 * frac  # newer = more opaque
                    ax.plot(tx[i:i + 2], ty[i:i + 2],
                            color="#ff7f0e", alpha=alpha, lw=0.9, zorder=2)
            nx_now, ny_now = shift_pt(nozzles_now[di][kn])
            # Current footprint
            ax.add_patch(mpatches.Circle(
                (nx_now, ny_now), fp_d / 2,
                facecolor="#ff7f0e", alpha=0.55, edgecolor="#c65500",
                linewidth=0.6, zorder=4))
            # Nozzle point
            ax.plot(nx_now, ny_now, "o", color="#c65500", markersize=3, zorder=5)

    # Array frame rectangle (rotated)
    yaw_rad = math.radians(s.yaw_deg)
    cos_t, sin_t = math.cos(yaw_rad), math.sin(yaw_rad)
    rys_rot = [r[1] for r in rotated]
    rxs_rot = [r[0] for r in rotated]
    half_fw = max(abs(min(rxs_rot)), abs(max(rxs_rot))) + s.disc_diameter_mm / 2 + 30
    y_top_rel = min(rys_rot) - s.disc_diameter_mm / 2 - 30
    y_bot_rel = max(rys_rot) + s.disc_diameter_mm / 2 + 30
    # Shift by current array_y
    if frame == "ROV frame":
        y_top = y_top_rel + 0  # ROV frame: array sits at its rotated local y directly
        y_bot = y_bot_rel
        centroid_y = s.row_pitch_mm / 2
    else:
        rov_speed_mm_s_ = rov_speed_mm_s
        array_leading_y = min(rys_rot) - s.disc_diameter_mm / 2
        margin_mm = 120
        array_y_offset_init = -margin_mm - array_leading_y
        array_y_now = array_y_offset_init + rov_speed_mm_s_ * t_now_s
        y_top = y_top_rel + array_y_now
        y_bot = y_bot_rel + array_y_now
    frame_corners = [(-half_fw, y_top), (half_fw, y_top),
                     (half_fw, y_bot), (-half_fw, y_bot)]
    ax.add_patch(mpatches.Polygon(frame_corners, closed=True, fill=False,
                                  edgecolor="#444", linewidth=1.2, zorder=2))

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

    # Axis limits: always include all current disc centres + trails + frame
    all_x = disc_xs + [p[0] for tr in trails_by_disc for nz in tr for p in nz]
    all_y = disc_ys + [p[1] + y_shift for tr in trails_by_disc for nz in tr for p in nz]
    if not all_x:
        all_x = [-half_fw, half_fw]; all_y = [y_top, y_bot]
    x_lo = min(min(all_x), -half_fw) - 60
    x_hi = max(max(all_x), half_fw) + 180
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


# -----------------------------------------------------------------------------
# Result renderer (KPI cards + heatmaps)
# -----------------------------------------------------------------------------
def render_result(s: Scenario, strip: np.ndarray, m: dict,
                  core_box: tuple[int, int, int, int],
                  container, vmax_shared: float | None = None,
                  label: str = "") -> None:
    cy0, cy1, cx0, cx1 = core_box

    c1, c2, c3, c4 = container.columns(4)
    c1.metric(f"{label}Cleaned area", f"{m['coverage_pct']:.1f} %",
              help=f"Cells with exposure >= {s.clean_threshold:.1f} bar*s")
    c2.metric(f"{label}Median exposure", f"{m['p50_bs']:.1f} bar*s")
    c3.metric(f"{label}10th percentile", f"{m['p10_bs']:.1f} bar*s",
              help="The worst-cleaned 10% of cells receive at least this much.")
    c4.metric(f"{label}Untouched", f"{m['missed_pct']:.2f} %")

    mode = "steady-state core" if s.steady_state_only else "full strip"
    container.caption(
        f"KPIs over {mode}. Δt = {m['dt_ms']:.2f} ms, {m['n_steps']} steps. "
        f"Array span (yaw {s.yaw_deg:+.0f}°) = "
        f"{m['array_span_x_mm']:.0f} mm across × {m['array_span_y_mm']:.0f} mm along. "
        f"Impact ring r = {s.impact_radius_mm():.1f} mm."
    )

    vmax = vmax_shared if vmax_shared is not None else max(strip.max(), 1e-6)
    array_span_x = m["array_span_x_mm"]
    extent = [-array_span_x / 2, array_span_x / 2, strip.shape[0] * CELL_SIZE_MM, 0]

    fig, ax = plt.subplots(figsize=(8, 3.6))
    im = ax.imshow(strip, extent=extent, aspect="auto",
                   cmap="viridis", vmin=0, vmax=vmax)
    if s.steady_state_only:
        core_y0_mm = cy0 * CELL_SIZE_MM
        core_y1_mm = cy1 * CELL_SIZE_MM
        ax.add_patch(mpatches.Rectangle(
            (extent[0], 0), extent[1] - extent[0], core_y0_mm,
            facecolor="white", alpha=0.45, edgecolor="none"))
        ax.add_patch(mpatches.Rectangle(
            (extent[0], core_y1_mm), extent[1] - extent[0],
            strip.shape[0] * CELL_SIZE_MM - core_y1_mm,
            facecolor="white", alpha=0.45, edgecolor="none"))
        ax.axhline(core_y0_mm, color="red", lw=0.8, ls="--")
        ax.axhline(core_y1_mm, color="red", lw=0.8, ls="--")
    ax.set_xlabel("Across-track (mm)")
    ax.set_ylabel("Along-track (mm)")
    ax.set_title(f"{label}Exposure heatmap (bar*s per cell)")
    plt.colorbar(im, ax=ax, label="bar*s")
    container.pyplot(fig, clear_figure=True)

    region = strip[cy0:cy1, cx0:cx1] if s.steady_state_only else strip
    region_extent_y = (cy1 - cy0) * CELL_SIZE_MM if s.steady_state_only \
        else strip.shape[0] * CELL_SIZE_MM
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

    left, right = st.columns([1.2, 1.0])
    with left:
        st.subheader("Top-down view")
        st.pyplot(plot_topdown(scen), clear_figure=True)
    with right:
        st.subheader("Side view (one disc)")
        st.pyplot(plot_side(scen), clear_figure=True)

    st.divider()

    # Motion visualisation
    with st.expander("Motion visualisation (time snapshot + nozzle trails)",
                     expanded=False):
        # Total traversal time = sim_length / rov_speed (same as simulate_pressure)
        rov_speed_mm_s = scen.rov_speed_kn * KNOTS_TO_MPS * 1000.0
        total_time_s = scen.sim_length_mm / max(rov_speed_mm_s, 1e-6)

        mc1, mc2, mc3 = st.columns([1.5, 1.0, 1.0])
        with mc1:
            t_ms = st.slider(
                "Snapshot time (ms)",
                min_value=0,
                max_value=int(total_time_s * 1000),
                value=int(total_time_s * 1000 * 0.25),
                step=max(1, int(total_time_s * 1000 / 400)),
                help="Scrub through the traversal to see the state at any instant.")
        with mc2:
            trail_rev = st.slider(
                "Trail length (disc revolutions)",
                0.0, 5.0, 1.0, step=0.25,
                help="How far back in time to draw each nozzle's path.")
        with mc3:
            frame = st.radio(
                "Reference frame",
                ["Hull frame", "ROV frame"],
                index=0,
                help="Hull frame: array translates, hull is stationary. "
                     "ROV frame: array is stationary; trails look like rosettes.")

        show_underlay = st.checkbox(
            "Show cumulative cleaning underlay (hull frame only)",
            value=False,
            help="Re-runs the pressure simulation up to the snapshot time and "
                 "shows the bar·s exposure so far as a faint heatmap. "
                 "Adds ~1–3 s per frame change.")

        cumulative = None
        if show_underlay and frame == "Hull frame":
            with st.spinner("Running cumulative simulation…"):
                cumulative, _m, _c = simulate_pressure(scen, t_stop_s=t_ms / 1000.0)

        st.pyplot(
            plot_motion(scen, t_ms / 1000.0,
                        trail_revolutions=trail_rev,
                        frame=frame,
                        cumulative_strip=cumulative),
            clear_figure=True)

        st.caption(
            f"Total traversal time: {total_time_s*1000:.0f} ms "
            f"({total_time_s:.2f} s). "
            f"Disc period at {scen.rpm} RPM: "
            f"{60000/max(scen.rpm,1):.1f} ms/rev."
        )

    st.divider()
    if st.button("Run full simulation", type="primary"):
        with st.spinner("Sweeping the array across the hull…"):
            strip, metrics, core_box = simulate_pressure(scen)
        render_result(scen, strip, metrics, core_box, st)
    else:
        st.info("Adjust parameters in the sidebar, then click **Run full simulation**.")

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
        st.pyplot(plot_topdown(scen_a, "Scenario A"), clear_figure=True)
    with top_b:
        st.subheader("Scenario B — layout")
        st.pyplot(plot_topdown(scen_b, "Scenario B"), clear_figure=True)

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
translation is factored out).

**Cumulative underlay.** Re-runs the pressure simulation from 0 up to the
chosen snapshot time and draws the bar·s exposure so far as a faint
heatmap. Useful for seeing *which parts* of the hull have received
cleaning by time t.

**Footprint diameter on hull.** Linear with pressure (60 mm at 50 bar →
80 mm at 600 bar) or manual override.

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
