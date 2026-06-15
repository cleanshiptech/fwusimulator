"""Matplotlib rendering helpers for the FWU simulator (schematics, motion, hull)."""

from __future__ import annotations

import math
import pickle

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.collections import LineCollection, PatchCollection

from fwu.constants import KNOTS_TO_MPS, HULL_SHAPES
from fwu.model import (
    Scenario,
    scenario_key,
    disc_layout,
    compute_rotated_discs,
)
from fwu.sim import (
    single_disc_coverage,
    disc_centres_hull,
    nozzle_trails_vec,
)
from fwu.model import rotate_point


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
def plot_single_disc_coverage(s: Scenario) -> tuple[plt.Figure, dict]:
    d = single_disc_coverage(s)
    touched = d["touched"]
    cell = d["cell"]
    ny, nx = touched.shape
    extent = [-nx / 2 * cell, nx / 2 * cell, ny * cell, 0]

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    # Touched cells in solid colour over an untouched background.
    ax.imshow(touched.astype(float), extent=extent, aspect="equal",
              cmap="Greens", vmin=0, vmax=1.4, interpolation="nearest")
    # Overlay the theoretical ring-band edges.
    ir = d["ring_r_mm"]
    for xr in (ir, -ir):
        ax.axvline(xr, color="#1f77b4", lw=0.7, ls=":")
    ax.set_xlabel("Across-track (mm)")
    ax.set_ylabel("Along-track (mm)")
    status = "gap-free" if d["overlap"] else "GAP RISK"
    ax.set_title(f"Swept coverage — {status}")
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
                  cumulative_strip.shape[0] * s.resolved_cell_mm, 0]
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


# -----------------------------------------------------------------------------
# Hull section drawing
# -----------------------------------------------------------------------------
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
