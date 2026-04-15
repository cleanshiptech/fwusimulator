"""
FWU Coverage Simulator
======================
Streamlit app to simulate hull-cleaning coverage of the C-Leanship ROV
pressure washing unit (FWU).

Hull is discretised on a 1x1 cm grid. For each time step every nozzle
deposits its instantaneous jet pressure (bar) onto the cells inside its
footprint, multiplied by dt. The accumulated quantity per cell is therefore
**integrated pressure exposure** in units of bar·seconds (bar·s).

The user can vary disc layout, rotation speed, nozzle geometry, jet
pressure and ROV traverse speed; the schematic and heatmap update live.

Run locally with:
    streamlit run app.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

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
    "Simulates integrated jet pressure exposure on the hull. "
    "Hull is split into a 1×1 cm grid; each cell accumulates "
    "bar·seconds of jet exposure as the ROV traverses."
)

KNOTS_TO_MPS = 0.514444  # 1 knot = 0.514444 m/s
CELL_SIZE_MM = 10.0      # fixed 1 cm grid as requested

# -----------------------------------------------------------------------------
# Sidebar — parameters
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Array geometry")
    array_width_mm = st.slider(
        "Array width (mm)", 1000, 2400, 1700, step=10,
        help="Total width of the array enclosure across the ROV.",
    )
    n_row1 = st.slider("Discs in front row", 2, 6, 4)
    n_row2 = st.slider("Discs in back row", 2, 6, 3)
    disc_pitch_mm = st.slider(
        "Disc pitch (mm)", 200, 700, 380, step=5,
        help="Centre-to-centre distance between adjacent discs in the SAME row. "
             "Both rows use this pitch. The back row is interlocked into the "
             "gaps of the front row, so back-row discs sit halfway between "
             "front-row discs (across-track).",
    )
    row_pitch_mm = st.slider(
        "Row pitch (mm)", 100, 700, 320, step=5,
        help="Centre-to-centre distance between the two disc rows (along ROV travel).",
    )

    st.header("Disc & nozzles")
    disc_diameter_mm = st.slider("Disc diameter (mm)", 200, 500, 360, step=5)
    n_nozzles = st.slider("Nozzles per disc", 1, 6, 3)
    nozzle_radius_mm = st.slider(
        "Nozzle radius from disc centre (mm)",
        20, 200, 80, step=1,
        help="Radial distance from the disc spin axis to each nozzle outlet.",
    )
    nozzle_cant_deg = st.slider(
        "Nozzle cant towards centre (°)", 0, 30, 10,
        help="Angle of jet relative to vertical, tilted towards the disc centre.",
    )
    standoff_mm = st.slider(
        "Nozzle standoff to hull (mm)", 5, 60, 18,
        help="Distance from nozzle outlet to hull surface.",
    )
    counter_rotate = st.checkbox("Adjacent discs counter-rotate", value=True)

    st.header("Operating point")
    rpm = st.slider("Disc rotation (RPM)", 50, 1500, 600, step=10)
    rov_speed_kn = st.slider("ROV traverse speed (knots)", 0.1, 4.0, 1.5, step=0.1)
    pressure_bar = st.slider("Jet pressure at nozzle (bar)", 50, 600, 200, step=10)
    nozzle_exit_mm = st.slider("Nozzle exit diameter (mm)", 0.5, 5.0, 1.5, step=0.1)

    st.header("Jet footprint model")
    footprint_mode = st.radio(
        "Footprint diameter on hull",
        ["Linear with pressure (60→80 mm)", "Manual override"],
        index=0,
    )
    if footprint_mode == "Manual override":
        footprint_dia_mm = st.slider("Footprint diameter (mm)", 20, 120, 70)
    else:
        footprint_dia_mm = 60.0 + (pressure_bar - 50.0) / (600.0 - 50.0) * 20.0
        st.caption(f"Footprint diameter on hull: **{footprint_dia_mm:.1f} mm**")

    pressure_profile = st.radio(
        "Pressure distribution within footprint",
        ["Uniform", "Gaussian (peak at centre)"],
        index=1,
        help="Uniform = same pressure everywhere inside the footprint disc. "
             "Gaussian = peak at the impingement centre, falling off radially.",
    )

    st.header("Hull strip & cleaning threshold")
    sim_length_mm = st.slider(
        "Hull strip length to simulate (mm)", 500, 3000, 1500, step=100,
    )
    clean_threshold = st.slider(
        "Cleaning threshold (bar·s)",
        0.5, 200.0, 20.0, step=0.5,
        help="A cell counts as 'cleaned' when its integrated pressure "
             "exposure exceeds this value.",
    )

# -----------------------------------------------------------------------------
# Derived geometry
# -----------------------------------------------------------------------------
@dataclass
class Disc:
    cx_mm: float          # x position (across-track, transverse)
    cy_mm: float          # y position (along-track, ROV travel direction)
    direction: int        # +1 CCW, -1 CW

def disc_layout() -> list[Disc]:
    """
    Both rows share the same disc pitch. Front row is centred about x=0 with
    n_row1 discs at x = (i - (n_row1-1)/2) * pitch.  Back row is interlocked:
    every back disc sits halfway between two adjacent front discs, so the
    back-row centres are at x = (j - (n_row2-1)/2) * pitch shifted by half a
    pitch only when the parity requires it. We keep the geometry symmetric
    about x=0 by centring each row independently.
    """
    discs: list[Disc] = []
    p = disc_pitch_mm
    # Front row, centred about 0
    for i in range(n_row1):
        x = (i - (n_row1 - 1) / 2.0) * p
        d = +1 if (i % 2 == 0 or not counter_rotate) else -1
        discs.append(Disc(cx_mm=x, cy_mm=0.0, direction=d))
    # Back row, also centred about 0 but offset by half a pitch when the
    # parities of n_row1 and n_row2 differ — this puts the back discs into
    # the front-row gaps. With n_row1=4 (even) and n_row2=3 (odd), no shift
    # is needed: front centres at ±0.5p, ±1.5p; back centres at 0, ±p; the
    # back discs land exactly in the centre of each front-row gap.
    needs_half_shift = (n_row1 % 2) == (n_row2 % 2)
    shift = (p / 2.0) if needs_half_shift else 0.0
    for j in range(n_row2):
        x = (j - (n_row2 - 1) / 2.0) * p + shift
        d = -1 if (j % 2 == 0 or not counter_rotate) else +1
        discs.append(Disc(cx_mm=x, cy_mm=row_pitch_mm, direction=d))
    return discs

discs = disc_layout()

# Effective impact radius on hull, given cant and standoff
impact_radius_mm = max(
    0.0,
    nozzle_radius_mm - standoff_mm * math.tan(math.radians(nozzle_cant_deg)),
)

# Actual span of the disc array (centre-to-centre + disc diameter)
_xs = [d.cx_mm for d in discs]
array_span_mm = (max(_xs) - min(_xs)) + disc_diameter_mm

# -----------------------------------------------------------------------------
# Top-down schematic
# -----------------------------------------------------------------------------
def plot_topdown() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    # Frame hugs the actual disc array, plus a small margin
    xs = [d.cx_mm for d in discs]
    ys = [d.cy_mm for d in discs]
    half_w = max(abs(min(xs)), abs(max(xs))) + disc_diameter_mm / 2 + 40
    y_top = min(ys) - disc_diameter_mm / 2 - 40
    y_bot = max(ys) + disc_diameter_mm / 2 + 40
    ax.add_patch(mpatches.Rectangle(
        (-half_w, y_top),
        2 * half_w, y_bot - y_top,
        fill=False, edgecolor="#444", linewidth=1.5,
    ))
    for d in discs:
        ax.add_patch(mpatches.Circle(
            (d.cx_mm, d.cy_mm), disc_diameter_mm / 2,
            fill=False, edgecolor="#1f77b4", linewidth=1.2,
        ))
        ax.add_patch(mpatches.Circle(
            (d.cx_mm, d.cy_mm), impact_radius_mm,
            fill=False, linestyle="--", edgecolor="#888", linewidth=0.6,
        ))
        for k in range(n_nozzles):
            theta = 2 * math.pi * k / n_nozzles
            nx = d.cx_mm + impact_radius_mm * math.cos(theta)
            ny = d.cy_mm + impact_radius_mm * math.sin(theta)
            ax.add_patch(mpatches.Circle(
                (nx, ny), footprint_dia_mm / 2,
                facecolor="#ff7f0e", alpha=0.30, edgecolor="none",
            ))
        arrow_r = disc_diameter_mm / 2 * 0.55
        ax.annotate(
            "", xy=(d.cx_mm + arrow_r * d.direction, d.cy_mm),
            xytext=(d.cx_mm, d.cy_mm + arrow_r),
            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=0.8),
        )
    ax.text(half_w + 80, row_pitch_mm / 2, "ROV travel →",
            rotation=-90, fontsize=9, color="#555")
    ax.set_xlim(-half_w - 150, half_w + 200)
    ax.set_ylim(-disc_diameter_mm / 2 - 100, row_pitch_mm + disc_diameter_mm / 2 + 100)
    ax.set_aspect("equal")
    ax.set_xlabel("Across-track (mm)")
    ax.set_ylabel("Along-track (mm)")
    ax.set_title("Top-down view of the disc array")
    ax.grid(True, alpha=0.3)
    return fig

# -----------------------------------------------------------------------------
# Side cross-section schematic
# -----------------------------------------------------------------------------
def plot_side() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.axhline(0, color="#555", lw=1.5)
    ax.text(impact_radius_mm * 1.1, -3, "hull", color="#555", fontsize=8)
    disc_h = 60
    ax.add_patch(mpatches.Rectangle(
        (-disc_diameter_mm / 2, standoff_mm),
        disc_diameter_mm, disc_h,
        facecolor="#cfe3f5", edgecolor="#1f77b4",
    ))
    ax.text(0, standoff_mm + disc_h / 2, "disc", ha="center", va="center", fontsize=9)
    for sign in (-1, +1):
        nx = sign * nozzle_radius_mm
        outlet = (nx, standoff_mm)
        end_x = nx - sign * standoff_mm * math.tan(math.radians(nozzle_cant_deg))
        ax.plot([outlet[0], end_x], [outlet[1], 0], color="#ff7f0e", lw=1.5)
        ax.add_patch(mpatches.Ellipse(
            (end_x, 0), footprint_dia_mm, 4,
            facecolor="#ff7f0e", alpha=0.4, edgecolor="none",
        ))
    ax.annotate("", xy=(-disc_diameter_mm / 2 - 20, 0),
                xytext=(-disc_diameter_mm / 2 - 20, standoff_mm),
                arrowprops=dict(arrowstyle="<->", color="#333"))
    ax.text(-disc_diameter_mm / 2 - 25, standoff_mm / 2,
            f"{standoff_mm} mm", ha="right", va="center", fontsize=8)
    ax.text(nozzle_radius_mm + 20, standoff_mm * 0.6,
            f"{nozzle_cant_deg}°", color="#ff7f0e", fontsize=9)
    ax.set_xlim(-disc_diameter_mm / 2 - 80, disc_diameter_mm / 2 + 80)
    ax.set_ylim(-15, standoff_mm + disc_h + 20)
    ax.set_aspect("equal")
    ax.set_xlabel("Across disc (mm)")
    ax.set_ylabel("Height (mm)")
    ax.set_title("Side cross-section: nozzle standoff & cant")
    ax.grid(True, alpha=0.3)
    return fig

# -----------------------------------------------------------------------------
# Footprint stencil (precomputed) — pressure weight per 1 cm cell
# -----------------------------------------------------------------------------
def footprint_stencil() -> np.ndarray:
    """
    Returns a 2D array of dimensionless pressure weights (0..1) on the
    1-cm grid, representing the spatial distribution of pressure within
    one nozzle footprint at one instant. The integral over the stencil
    equals the average weight × number of cells under the footprint.

    For 'Uniform': weight = 1.0 inside the footprint disc, 0 outside.
    For 'Gaussian': weight = exp(-r^2 / (2 sigma^2)), with sigma chosen so
    the footprint diameter equals 2*2*sigma (i.e. ~95% of the energy
    falls inside the stated footprint diameter).
    """
    r_mm = footprint_dia_mm / 2.0
    r_cells = r_mm / CELL_SIZE_MM
    half = int(math.ceil(r_cells)) + 1
    yy, xx = np.ogrid[-half:half + 1, -half:half + 1]
    r2 = (xx ** 2 + yy ** 2).astype(np.float32)
    if pressure_profile.startswith("Uniform"):
        stencil = (r2 <= r_cells ** 2).astype(np.float32)
    else:
        sigma = r_cells / 2.0  # 2-sigma at footprint edge
        stencil = np.exp(-r2 / (2 * sigma ** 2)).astype(np.float32)
        stencil[r2 > (r_cells * 1.5) ** 2] = 0.0
    return stencil

# -----------------------------------------------------------------------------
# Coverage simulation — integrated pressure (bar·s) per 1 cm cell
# -----------------------------------------------------------------------------
def simulate_pressure() -> tuple[np.ndarray, dict]:
    rov_speed_mm_s = rov_speed_kn * KNOTS_TO_MPS * 1000.0
    omega_rad_s = rpm * 2 * math.pi / 60.0

    margin_mm = 100
    full_extent = sim_length_mm + row_pitch_mm + 2 * margin_mm + disc_diameter_mm
    width_extent = array_span_mm + 2 * margin_mm

    nx = int(width_extent / CELL_SIZE_MM)
    ny = int(full_extent / CELL_SIZE_MM)
    grid = np.zeros((ny, nx), dtype=np.float32)  # bar·s

    # Time-step: <=5° rotation AND <=0.5 cell of ROV travel per step
    dt_rot = math.radians(5.0) / max(omega_rad_s, 1e-6)
    dt_trav = 0.5 * CELL_SIZE_MM / max(rov_speed_mm_s, 1e-6)
    dt = min(dt_rot, dt_trav)

    total_time = sim_length_mm / max(rov_speed_mm_s, 1e-6)
    n_steps = int(total_time / dt) + 1
    n_steps = min(n_steps, 12000)
    dt = total_time / n_steps

    stencil = footprint_stencil()                # dimensionless 0..1
    sh, sw = stencil.shape
    rr = sh // 2                                 # half-size of stencil

    # Per-step pressure deposit per cell = pressure_bar * stencil * dt  (bar·s)
    # Pre-multiply pressure × dt to avoid recomputing inside the inner loop.
    p_dt = float(pressure_bar) * dt
    weighted_stencil = stencil * p_dt

    x0 = -width_extent / 2
    y0 = -margin_mm

    array_y_offset_init = -margin_mm - disc_diameter_mm / 2
    phases = [2 * math.pi * i / max(len(discs), 1) for i in range(len(discs))]

    for step in range(n_steps):
        t = step * dt
        array_y = array_y_offset_init + rov_speed_mm_s * t
        for di, d in enumerate(discs):
            phase = phases[di] + d.direction * omega_rad_s * t
            cx = d.cx_mm
            cy = d.cy_mm + array_y
            for k in range(n_nozzles):
                theta = phase + 2 * math.pi * k / n_nozzles
                nx_mm = cx + impact_radius_mm * math.cos(theta)
                ny_mm = cy + impact_radius_mm * math.sin(theta)
                ix = int(round((nx_mm - x0) / CELL_SIZE_MM))
                iy = int(round((ny_mm - y0) / CELL_SIZE_MM))
                xs = ix - rr; ys = iy - rr
                xe = xs + sw; ye = ys + sh
                gx0 = max(xs, 0); gy0 = max(ys, 0)
                gx1 = min(xe, nx); gy1 = min(ye, ny)
                if gx1 <= gx0 or gy1 <= gy0:
                    continue
                mx0 = gx0 - xs; my0 = gy0 - ys
                mx1 = mx0 + (gx1 - gx0); my1 = my0 + (gy1 - gy0)
                grid[gy0:gy1, gx0:gx1] += weighted_stencil[my0:my1, mx0:mx1]

    # Strip = the region of hull that the entire array has passed over
    y_strip_start = int((row_pitch_mm + disc_diameter_mm / 2 + margin_mm) / CELL_SIZE_MM)
    y_strip_end = int((row_pitch_mm + disc_diameter_mm / 2 + margin_mm + sim_length_mm) / CELL_SIZE_MM)
    x_strip_start = int(margin_mm / CELL_SIZE_MM)
    x_strip_end = int((margin_mm + array_span_mm) / CELL_SIZE_MM)
    strip = grid[y_strip_start:y_strip_end, x_strip_start:x_strip_end]

    metrics = {
        "rov_speed_mm_s": rov_speed_mm_s,
        "dt_ms": dt * 1000.0,
        "n_steps": n_steps,
        "mean_bs": float(strip.mean()) if strip.size else 0.0,
        "p10_bs": float(np.percentile(strip, 10)) if strip.size else 0.0,
        "p50_bs": float(np.percentile(strip, 50)) if strip.size else 0.0,
        "p90_bs": float(np.percentile(strip, 90)) if strip.size else 0.0,
        "min_bs": float(strip.min()) if strip.size else 0.0,
        "max_bs": float(strip.max()) if strip.size else 0.0,
        "coverage_pct": float((strip >= clean_threshold).mean() * 100.0)
                       if strip.size else 0.0,
        "missed_pct": float((strip == 0).mean() * 100.0) if strip.size else 0.0,
    }
    return strip, metrics

# -----------------------------------------------------------------------------
# Layout
# -----------------------------------------------------------------------------
left, right = st.columns([1.1, 1.0])
with left:
    st.subheader("Top-down view")
    st.pyplot(plot_topdown(), clear_figure=True)
with right:
    st.subheader("Side view (one disc)")
    st.pyplot(plot_side(), clear_figure=True)

st.divider()

run_btn = st.button("Run pressure-exposure simulation", type="primary")
if run_btn:
    with st.spinner("Sweeping the array across the hull strip…"):
        strip, m = simulate_pressure()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cleaned area", f"{m['coverage_pct']:.1f} %",
              help=f"Cells with exposure ≥ {clean_threshold:.1f} bar·s")
    c2.metric("Median exposure", f"{m['p50_bs']:.1f} bar·s")
    c3.metric("10th percentile", f"{m['p10_bs']:.1f} bar·s",
              help="The worst-cleaned 10% of cells receive at least this much.")
    c4.metric("Untouched area", f"{m['missed_pct']:.2f} %")

    st.caption(
        f"Simulated {m['n_steps']} time steps at Δt = {m['dt_ms']:.2f} ms. "
        f"1×1 cm grid. ROV speed = {m['rov_speed_mm_s']:.0f} mm/s. "
        f"Impact ring radius on hull = {impact_radius_mm:.1f} mm "
        f"(nozzle r {nozzle_radius_mm} mm − standoff·tan({nozzle_cant_deg}°))."
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    extent = [
        -array_span_mm / 2, array_span_mm / 2,
        sim_length_mm, 0,
    ]
    im = ax.imshow(
        strip, extent=extent, aspect="auto",
        cmap="viridis", vmin=0, vmax=max(strip.max(), 1e-6),
    )
    ax.set_xlabel("Across-track (mm)")
    ax.set_ylabel("Along-track distance behind array (mm)")
    ax.set_title("Integrated jet pressure exposure per 1×1 cm cell (bar·s)")
    plt.colorbar(im, ax=ax, label="bar·seconds")
    st.pyplot(fig, clear_figure=True)

    # Threshold map
    fig_t, ax_t = plt.subplots(figsize=(10, 3.6))
    cleaned_mask = (strip >= clean_threshold).astype(np.float32)
    ax_t.imshow(cleaned_mask, extent=extent, aspect="auto",
                cmap="Greens", vmin=0, vmax=1)
    ax_t.set_xlabel("Across-track (mm)")
    ax_t.set_ylabel("Along-track (mm)")
    ax_t.set_title(f"Cells reaching ≥ {clean_threshold:.1f} bar·s (green = cleaned)")
    st.pyplot(fig_t, clear_figure=True)

    # Histogram
    fig2, ax2 = plt.subplots(figsize=(8, 2.6))
    ax2.hist(strip.ravel(), bins=60, color="#1f77b4", edgecolor="white")
    ax2.axvline(clean_threshold, color="red", lw=1.5, ls="--",
                label=f"Cleaning threshold = {clean_threshold:.1f} bar·s")
    ax2.set_xlabel("Integrated exposure per cell (bar·s)")
    ax2.set_ylabel("Cell count")
    ax2.legend()
    ax2.set_title("Distribution of pressure exposure across hull cells")
    st.pyplot(fig2, clear_figure=True)
else:
    st.info(
        "Adjust parameters in the sidebar, then click "
        "**Run pressure-exposure simulation** to compute exposure per cm² cell. "
        "The schematics above always reflect the current settings."
    )

st.divider()
with st.expander("Modelling assumptions and units"):
    st.markdown(
        """
**Hull grid.** Hull is discretised on a fixed 1×1 cm grid. Each cell
accumulates *integrated jet pressure exposure* with units of **bar·seconds
(bar·s)**. This is the time-integral of the local jet pressure that the
cell experiences as the array sweeps over it.

**Per-step deposit.** At each simulation step Δt:
`exposure_cell += pressure_bar · stencil_cell · Δt`
where `stencil_cell ∈ [0, 1]` is the spatial weight of the jet at that
cell (uniform inside the footprint, or Gaussian peaked at the impingement
centre).

**Footprint diameter.** Linear with pressure (60 mm at 50 bar → 80 mm at
600 bar), or manual override.

**Cant correction.** Nozzles canted toward the disc centre by `θ` shift
their impingement radius on the hull from the nozzle radius `r` to
`r − standoff · tan θ`. The visualised dashed impact ring on the
top-down view reflects this.

**Cleaning threshold.** A cell is counted as cleaned when its exposure
exceeds the user-set threshold in bar·s. A reasonable starting range is
10–50 bar·s depending on coating and biofouling severity — calibrate
against field-cleaning trials and update.

**Not yet modelled.** Pressure decay along the jet, hull curvature,
biofouling resistance, ROV pitch/roll, water-jet shear vs pressure,
nozzle wear.

**Calibration tip.** If you have a known operating point that gives a
clean hull (e.g. 250 bar, 1.5 kn, 600 RPM yields cleaned hull), run that
configuration here, read the median bar·s value, and use it as your
`clean_threshold` going forward.
        """
    )
