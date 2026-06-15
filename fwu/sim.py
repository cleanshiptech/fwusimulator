"""Pressure simulation and motion/coverage compute for the FWU simulator."""

from __future__ import annotations

import math
from copy import deepcopy

import numpy as np

from fwu.constants import KNOTS_TO_MPS, CELL_SIZE_MM
from fwu.model import (
    Scenario,
    disc_layout,
    rotate_point,
    compute_rotated_discs,
    footprint_stencil,
    footprint_profile_peaknorm,
)


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
    one.auto_grid = False
    one.cell_size_mm = min(s.resolved_cell_mm, 2.0)
    rov_mm_s = max(one.rov_speed_kn * KNOTS_TO_MPS * 1000.0, 1e-6)
    rev_s = one.rpm / 60.0
    advance_per_rev = rov_mm_s / max(rev_s, 1e-6)
    # Simulate enough length to show the ring overlap pattern over a few
    # revolutions, but keep the strip from becoming an extreme tall sliver:
    # one disc-width across is enough context, ~the same along-track.
    one.sim_length_mm = int(min(max(5 * advance_per_rev, 1.4 * one.disc_diameter_mm),
                                3.0 * one.disc_diameter_mm))
    one.steady_state_only = False

    strip, m, box = simulate_pressure(one)
    cell = one.cell_size_mm
    # Threshold at a small fraction of the peak, NOT > 0: the FFT-convolution
    # deposit leaves ~1e-12 round-off in cells that should be exactly zero,
    # which `> 0` would render as speckled green "noise" across the disc
    # interior and edges. A 0.1%-of-peak cutoff keeps real coverage and drops
    # the round-off.
    touched = strip > (float(strip.max()) * 1e-3)

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

    cell = s.resolved_cell_mm
    nx = int(width_extent / cell)
    ny = int(full_extent / cell)
    grid = np.zeros((ny, nx), dtype=np.float32)
    passes = np.zeros((ny, nx), dtype=np.float32)  # nozzle passes per cell
    peak_pressure = np.zeros((ny, nx), dtype=np.float32)  # max delivered bar/cell

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
    # Sampling (Nyquist): the jet must advance no more than a quarter of a
    # CELL per step along its ring, so the swept track is sampled finer than
    # the grid and doesn't alias into a beat pattern with it (which made the
    # KPIs swing with cell size). Tighter than the old 0.5·footprint rule.
    dt_arc = 0.25 * cell / max(v_tan, 1e-6)
    dt = min(dt_rot, dt_trav, dt_arc)
    total_time_full = s.sim_length_mm / max(rov_speed_mm_s, 1e-6)
    total_time = total_time_full if t_stop_s is None else min(t_stop_s, total_time_full)
    n_steps_ideal = int(total_time / dt) + 1
    # Cap to bound runtime; record whether the cap forced a coarser dt than
    # the path-sampling constraint wants (track then undersampled / aliased).
    STEP_CAP = 400000
    n_steps = min(n_steps_ideal, STEP_CAP)
    dt = total_time / max(n_steps, 1)
    arc_per_step = v_tan * dt
    undersampled = arc_per_step > cell

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

        # Shared FFT of the hit grid, reused by every deposit below.
        fy = pny + sh - 1
        fx = pnx + sw - 1
        H = np.fft.rfft2(hit, s=(fy, fx))

        def conv_with(kernel: np.ndarray) -> np.ndarray:
            K = np.fft.rfft2(kernel, s=(fy, fx))
            c = np.fft.irfft2(H * K, s=(fy, fx))
            return c[2 * rr:2 * rr + ny, 2 * rr:2 * rr + nx].astype(np.float32)

        # Cleaning intensity at the footprint centre (the chosen measure:
        # stagnation / mean / wall shear). The per-cell value falls toward the
        # footprint edge with the same peak-normalised profile.
        intensity_peak = s.cleaning_intensity()

        if sh == 1 and sw == 1:
            # Single-cell footprint (sub-resolution jet): no spread to apply.
            hcore = hit[rr:rr + ny, rr:rr + nx]
            grid += hcore * weighted_stencil[0, 0]
            passes += hcore * binary_stencil[0, 0]
            peak_pressure += (hcore > 0.5) * intensity_peak
        else:
            # Energy (bar·s) and pass-count deposits via FFT convolution.
            grid += conv_with(weighted_stencil)
            passes += conv_with(binary_stencil)

            # Peak cleaning intensity per cell, built as a few nested bands.
            # The intensity at a cell is intensity_peak · profile(d), d =
            # distance to the nearest hit centre. Thresholding the peak-
            # normalised profile at a few levels and presence-convolving each
            # band marks cells within that band's radius of any hit; the max
            # band reached is the delivered intensity.
            peak_prof = footprint_profile_peaknorm(s)
            bands = np.zeros((ny, nx), dtype=np.float32)
            for lv in (0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0):
                sub = (peak_prof >= lv).astype(np.float32)
                if sub.sum() == 0:
                    continue
                covered = conv_with(sub) > 0.5
                np.maximum(bands, covered * (lv * intensity_peak), out=bands)
            peak_pressure += bands

    y_strip_start = 0
    y_strip_end = int((s.sim_length_mm + array_span_y) / cell)
    x_strip_start = int(margin_mm / cell)
    x_strip_end = int((margin_mm + array_span_x) / cell)
    strip = grid[y_strip_start:y_strip_end, x_strip_start:x_strip_end]
    passes_strip = passes[y_strip_start:y_strip_end, x_strip_start:x_strip_end]
    pressure_strip = peak_pressure[y_strip_start:y_strip_end,
                                   x_strip_start:x_strip_end]

    # Steady-state core excludes BOTH transients. The entry transient fills
    # the first array_span_y (array driving in). The EXIT transient — the
    # trailing discs leaving — occupies roughly the last array_span_y/2 of the
    # array's travel, so the upper bound is sim_length − array_span_y/2 (the
    # old sim_length cutoff left that decay in, dragging the KPIs down).
    core_y0 = int(array_span_y / cell)
    core_y1 = int((s.sim_length_mm - array_span_y / 2.0) / cell)
    if core_y1 <= core_y0:
        core_y0, core_y1 = 0, strip.shape[0]
    core_x0 = 0
    core_x1 = strip.shape[1]
    core_box = (core_y0, core_y1, core_x0, core_x1)

    if s.steady_state_only:
        region = strip[core_y0:core_y1, core_x0:core_x1]
        region_passes = passes_strip[core_y0:core_y1, core_x0:core_x1]
        region_pressure = pressure_strip[core_y0:core_y1, core_x0:core_x1]
    else:
        region = strip
        region_passes = passes_strip
        region_pressure = pressure_strip

    # --- Cleaning criterion: PER-CELL intensity gate × dose gate ----------
    # Each cell has its own delivered intensity (the chosen measure —
    # stagnation / mean / wall shear — is highest at the footprint centre and
    # falls toward the edges), so the intensity gate is per cell. A cell is
    # cleaned iff the delivered intensity clears the removal threshold AND it
    # received >= min passes.
    intensity_peak = s.cleaning_intensity()       # centreline (peak) value
    intensity_ok = intensity_peak >= s.removal_pressure_bar   # can ANY cell clean?
    # `passes` is a convolution, so it is fractional at footprint edges. Treat
    # < 0.5 of a pass as effectively unstruck ("untouched") so the three
    # categories (cleaned / partial / untouched) partition the area exactly.
    TOUCH = 0.5
    if region_passes.size:
        touched_mask = region_passes >= TOUCH
        cleaned_mask = (touched_mask
                        & (region_pressure >= s.removal_pressure_bar)
                        & (region_passes >= s.min_passes))
        cleaned_pct = float(cleaned_mask.mean() * 100.0)
        partial_pct = float((touched_mask & ~cleaned_mask).mean() * 100.0)
        missed_pct = float((~touched_mask).mean() * 100.0)
    else:
        cleaned_pct = partial_pct = missed_pct = 0.0

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
        "stagnation_pressure_bar": float(intensity_peak),
        "intensity_ok": bool(intensity_ok),
        "cleaning_measure": s.cleaning_measure,
        "median_passes": float(np.median(region_passes)) if region_passes.size else 0.0,
        "max_passes": float(region_passes.max()) if region_passes.size else 0.0,
        "median_pressure_bar": float(np.median(region_pressure[region_pressure > 0]))
                              if (region_pressure > 0).any() else 0.0,
        # Area partition (sums to 100%): untouched = effectively unstruck;
        # partial = struck but failed the pressure or min-passes gate.
        "missed_pct": missed_pct,
        "partial_pct": partial_pct,
        "region_area_mm2": float(region.size * cell * cell),
        "total_time_s": total_time_full,
        "arc_per_step_mm": float(arc_per_step),
        "footprint_mm": float(fp_d),
        "undersampled": bool(undersampled),
        # Spatial maps for the heatmaps (delivered pressure is the new primary).
        "pressure_strip": pressure_strip,
        "passes_strip": passes_strip,
    }
    return strip, metrics, core_box


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
