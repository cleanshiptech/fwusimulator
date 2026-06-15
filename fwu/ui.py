"""Streamlit UI builders: sidebar scenario controls and result renderer."""

from __future__ import annotations

import math
from copy import deepcopy

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from fwu.model import Scenario


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
    s.nozzle_exit_mm = container.slider(
        "Nozzle exit diameter (mm)", 0.5, 5.0, s.nozzle_exit_mm, step=0.1,
        key=k("nozzle_exit_mm"))
    s.total_flow_lpm = container.slider(
        "Total pump flow (L/min)", 50.0, 600.0, float(s.total_flow_lpm),
        step=10.0, key=k("total_flow_lpm"),
        help="Combined flow from the pumps (2 Denjet ≈ 270 L/min). This is "
             "split evenly across all nozzles; the nozzle pressure is "
             "DERIVED from it, not set directly.")
    # Nozzle pressure is now a derived quantity — show it live so the
    # flow→pressure coupling is explicit.
    container.caption(
        f"→ **{s.n_nozzles_total} nozzles** "
        f"({s.n_row1 + s.n_row2} discs × {s.n_nozzles}) share "
        f"{s.total_flow_lpm:.0f} L/min = "
        f"**{s.flow_per_nozzle_lpm:.1f} L/min/nozzle** → nozzle pressure "
        f"**{s.pressure_bar:.0f} bar** (orifice law Q = K·d²·√p). "
        "Adding nozzles or widening the exit lowers this.")

    container.subheader("Jet footprint model")
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
    else:
        container.caption(f"Footprint diameter on hull: **{s.footprint_dia():.1f} mm**")

    # Hull-grid resolution. Auto (default) derives the cell from the footprint
    # so the disk is always well-resolved (~4 cells across) and the KPIs don't
    # swing with an arbitrary grid choice. The sampling rule is Nyquist-like:
    # the grid (and the jet-path time-step) must be fine enough relative to the
    # footprint, or the swept track aliases and the cleaned-area KPI wobbles.
    s.auto_grid = container.checkbox(
        "Auto grid resolution (recommended)", value=s.auto_grid,
        key=k("auto_grid"),
        help="Derive the calculation cell from the footprint "
             "(cell ≈ footprint / 4) so the result is resolution-independent. "
             "Turn off to pick a cell size manually.")
    if s.auto_grid:
        container.caption(
            f"Grid: **{s.resolved_cell_mm:.2f} mm/cell** "
            f"(= {s.footprint_dia():.1f} mm footprint ÷ 4). The footprint "
            f"spans ~{s.footprint_dia() / s.resolved_cell_mm:.0f} cells.")
    else:
        _grid_opts = [10.0, 5.0, 2.0, 1.0]
        _grid_idx = (_grid_opts.index(s.cell_size_mm)
                     if s.cell_size_mm in _grid_opts else 1)
        s.cell_size_mm = container.selectbox(
            "Hull grid resolution (mm/cell)", _grid_opts, index=_grid_idx,
            format_func=lambda v: f"{v:g} mm", key=k("cell_size_mm"),
            help="Manual cell size. Note the Nyquist trap: a cell near "
                 "footprint/2–footprint/3 can ALIAS the swept track and make "
                 "the cleaned-area KPI wobble. Prefer auto, or keep "
                 "cell ≤ footprint/4.")
        _fp_cells = s.footprint_dia() / s.cell_size_mm
        if _fp_cells < 3.0:
            container.warning(
                f"⚠ Footprint ({s.footprint_dia():.1f} mm) spans only "
                f"{_fp_cells:.1f} cells at {s.cell_size_mm:g} mm — under-"
                "resolved, the cleaned-area KPI is unreliable here. Use auto "
                f"grid, or pick ≤ {s.footprint_dia() / 4:.1f} mm.")
    s.pressure_profile = container.radio(
        "Pressure distribution within footprint",
        ["Uniform", "Gaussian (peak at centre)"],
        index=0 if s.pressure_profile.startswith("Uniform") else 1,
        key=k("pressure_profile"),
    )

    container.subheader("Cleaning criterion")
    # Which impact measure gates cleaning. Stagnation/Mean are in bar, Wall
    # shear in kPa — see the System & Impact tab for all three vs standoff.
    _measures = ["Wall shear stress", "Stagnation pressure",
                 "Mean impact pressure"]
    _m_idx = _measures.index(s.cleaning_measure) if s.cleaning_measure in _measures else 0
    s.cleaning_measure = container.radio(
        "Cleaning driver (impact measure)", _measures, index=_m_idx,
        key=k("cleaning_measure"),
        help="Which hull-impact quantity must clear the removal threshold. "
             "Wall shear = tangential scrub (accelerated water shears fouling "
             "off); Stagnation = head-on ½ρv² (hard-fouling fracture); Mean = "
             "force ÷ footprint (bulk load). All three shown in the System & "
             "Impact tab.")
    _unit = "kPa" if s.cleaning_measure.startswith("Wall") else "bar"
    # Per-measure fouling presets (typical removal thresholds, calibratable).
    _presets = {
        "Wall shear stress": {"Soft biofilm / slime": 6.0,
                              "Light weed / early growth": 15.0,
                              "Hard calcareous / barnacle": 30.0, "Custom": None},
        "Stagnation pressure": {"Soft biofilm / slime": 15.0,
                               "Light weed / early growth": 60.0,
                               "Hard calcareous / barnacle": 150.0, "Custom": None},
        "Mean impact pressure": {"Soft biofilm / slime": 8.0,
                                "Light weed / early growth": 30.0,
                                "Hard calcareous / barnacle": 80.0, "Custom": None},
    }[s.cleaning_measure]
    _foul_choice = container.selectbox(
        f"Fouling type (sets removal {_unit})", list(_presets.keys()), index=0,
        key=k("foul_preset"),
        help="Minimum delivered intensity needed to lift this fouling. "
             "Calibrate to a run you KNOW cleans your fouling — the live "
             "value at your standoff is shown below.")
    if _presets[_foul_choice] is not None:
        s.removal_pressure_bar = _presets[_foul_choice]
    else:
        _hi = 100.0 if _unit == "kPa" else 400.0
        s.removal_pressure_bar = container.slider(
            f"Removal threshold ({_unit})", 0.5, _hi,
            float(s.removal_pressure_bar), step=0.5,
            key=k("removal_pressure_bar"))
    _val = s.cleaning_intensity()
    _gate = "✅ above" if _val >= s.removal_pressure_bar else "❌ BELOW"
    container.caption(
        f"At {s.standoff_mm} mm standoff the jet delivers "
        f"**{_val:.1f} {_unit}** ({s.cleaning_measure.lower()}) — {_gate} the "
        f"{s.removal_pressure_bar:.1f} {_unit} removal threshold. "
        f"(Exit v = {s.jet_exit_velocity:.0f} m/s, core "
        f"{s.core_length_mm:.1f} mm.)")
    s.min_passes = container.slider(
        "Minimum passes to clean", 1, 10, int(s.min_passes), key=k("min_passes"),
        help="Once the intensity gate is cleared, a cell must be struck at "
             "least this many times to count as cleaned.")

    # Jet-physics constants that convert flow → impact. Most are ASSUMED
    # (handbook values for a submerged straight-bore jet) and are the biggest
    # source of model uncertainty — a film/dye firing test would replace them.
    with container.expander("Jet physics (calibratable ⚠ assumed)"):
        st.caption(
            "These convert flow into the impact at the hull. Marked **⚠ "
            "assumed** are handbook values — measure them (single-jet firing "
            "test against pressure film / dye) to pin down the model. "
            "The Calibration status panel lists what to measure first.")
        s.nozzle_cd = st.slider(
            "Discharge coefficient Cd ⚠", 0.6, 1.0, float(s.nozzle_cd),
            step=0.01, key=k("nozzle_cd"),
            help="Ratio of real to ideal (Bernoulli) exit velocity. ~0.92 "
                 "derived from your 140 bar inlet vs 154 m/s exit — verify.")
        s.decay_K = st.slider(
            "Far-field decay constant K ⚠", 4.0, 7.0, float(s.decay_K),
            step=0.1, key=k("decay_K"),
            help="Centreline velocity beyond the core decays as v_c/v0 = "
                 "K·d/x. ~5.5 for a straight bore (assumed — measure).")
        s.core_factor = st.slider(
            "Potential-core length (× exit dia) ⚠", 3.0, 8.0,
            float(s.core_factor), step=0.5, key=k("core_factor"),
            help="Length the jet holds full velocity = factor × exit dia. "
                 "~5 for a straight bore (assumed).")
        s.jet_half_angle_deg = st.slider(
            "Jet half-angle (°) ⚠", 8.0, 20.0, float(s.jet_half_angle_deg),
            step=0.5, key=k("jet_half_angle_deg"),
            help="Spread half-angle of the free jet. ~14° straight bore "
                 "(assumed — measure).")
        s.skin_friction_cf = st.slider(
            "Wall skin-friction Cf ⚠", 0.001, 0.008, float(s.skin_friction_cf),
            step=0.0005, format="%.4f", key=k("skin_friction_cf"),
            help="Sets wall shear τ = Cf·½ρv². ~0.003 for a turbulent wall "
                 "jet (assumed). Only matters when the cleaning driver is "
                 "Wall shear.")
        s.water_density = st.selectbox(
            "Water density (kg/m³)", [1000.0, 1026.0],
            index=0 if s.water_density < 1013 else 1,
            key=k("water_density"),
            help="1000 fresh / 1026 seawater. ✓ known.")

    container.subheader("Hull strip & KPIs")
    s.sim_length_mm = container.slider(
        "Hull strip length to simulate (mm)", 500, 10000, s.sim_length_mm, step=100,
        key=k("sim_length_mm"))
    s.clean_threshold = container.slider(
        "Exposure colour scale (bar·s, diagnostic)",
        0.05, 50.0, float(s.clean_threshold), step=0.05,
        key=k("clean_threshold"),
        help="Diagnostic only: sets the green level on the bar·s exposure "
             "map inside the Advanced expander. It does NOT affect the "
             "Cleaned-area KPI, which uses the intensity gate + passes "
             "criterion above.")
    s.steady_state_only = container.checkbox(
        "Report KPIs on steady-state core only",
        value=s.steady_state_only,
        key=k("steady_state_only"),
        help="Excludes the array's entry and exit transients from the KPIs and "
             "the bottom heatmap. Transient regions stay visible in the main "
             "heatmap but are dimmed.")
    return s


# -----------------------------------------------------------------------------
# Result renderer (KPI cards + heatmaps) — unchanged
# -----------------------------------------------------------------------------
def render_result(s: Scenario, strip: np.ndarray, m: dict,
                  core_box: tuple[int, int, int, int],
                  container, vmax_shared: float | None = None,
                  label: str = "") -> None:
    cy0, cy1, cx0, cx1 = core_box

    # The cleaning-driver intensity gets its OWN box (its own units) so it is
    # never read as comparable to the bar·s exposure maps below.
    _val = m.get("stagnation_pressure_bar", 0)   # holds the chosen measure's value
    _unit = "kPa" if s.cleaning_measure.startswith("Wall") else "bar"
    _name = s.cleaning_measure.lower()
    if m.get("intensity_ok"):
        container.success(
            f"**{label}{s.cleaning_measure}: {_val:.1f} {_unit}** at the hull "
            f"— clears the {s.removal_pressure_bar:.1f} {_unit} removal gate "
            f"for this fouling. (Exit v = {s.jet_exit_velocity:.0f} m/s, "
            f"decayed over {s.standoff_mm} mm standoff with a "
            f"{s.nozzle_exit_mm:.1f} mm exit.) This is how *hard* the jet "
            "hits — distinct from the bar·s exposure maps below.")
    else:
        container.error(
            f"**{label}{s.cleaning_measure}: {_val:.1f} {_unit}** at the hull "
            f"— BELOW the {s.removal_pressure_bar:.1f} {_unit} removal gate, "
            "so cleaned area is 0% no matter how many passes. Shorten the "
            "standoff (biggest lever), raise flow, or narrow the exit to "
            "deliver more intensity to the hull.")

    # The area splits three ways and sums to ~100%: cleaned (gate met),
    # partial (struck but under-gated), untouched (never struck).
    c1, c2, c3, c4 = container.columns(4)
    c1.metric(f"{label}Cleaned area", f"{m.get('cleaned_pct', 0):.1f} %",
              help=f"Cells that clear the impact-pressure gate AND get "
                   f"≥ {s.min_passes} passes. The headline cleaning KPI.")
    c2.metric(f"{label}Partial", f"{m.get('partial_pct', 0):.1f} %",
              help="Struck at least once but NOT cleaned — either the "
                   "delivered pressure was below the removal gate (footprint "
                   f"edges) or it got fewer than {s.min_passes} passes.")
    c3.metric(f"{label}Untouched", f"{m['missed_pct']:.1f} %",
              help="Cells never struck by any nozzle (zero passes).")
    c4.metric(f"{label}Median passes", f"{m.get('median_passes', 0):.1f}",
              help="Typical number of nozzle passes per cell over the region.")
    _tot = m.get('cleaned_pct', 0) + m.get('partial_pct', 0) + m['missed_pct']
    container.caption(
        f"Cleaned + Partial + Untouched = {_tot:.0f}% of the steady-state "
        "core (the three are mutually exclusive and cover the area).")

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

    array_span_x = m["array_span_x_mm"]
    cell = s.resolved_cell_mm

    # ---- PRIMARY: two side-by-side maps — # passes and accumulated exposure
    # Two distinct, structure-preserving fields:
    #   passes              = pure geometric coverage (disc overlap, gaps),
    #   accumulated exposure= bar·s = pressure × dwell summed over passes.
    # NOTE on units: this is NOT comparable to the "delivered pressure" KPI.
    # Delivered pressure is in bar (instantaneous force); accumulated exposure
    # is in bar·s (force × time). The same 18 bar jet only dwells ~0.5 ms per
    # pass, so it accumulates well under 1 bar·s — different dimension, not a
    # smaller version of the same number.
    extent = [-array_span_x / 2, array_span_x / 2, strip.shape[0] * cell, 0]
    passes_full = m.get("passes_strip")

    def _draw_core_overlay(ax):
        if not s.steady_state_only:
            return
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

    hm_l, hm_r = container.columns(2)

    # Left: # passes per cell.
    with hm_l:
        if passes_full is not None:
            pvmax = max(float(np.percentile(
                passes_full[passes_full > 0], 99))
                if (passes_full > 0).any() else 1.0, 1.0)
            fign, axn = plt.subplots(figsize=(5.2, 3.6))
            imn = axn.imshow(passes_full, extent=extent, aspect="auto",
                             cmap="magma", vmin=0, vmax=pvmax)
            _draw_core_overlay(axn)
            axn.set_xlabel("Across-track (mm)")
            axn.set_ylabel("Along-track (mm)")
            axn.set_title(f"{label}Passes per cell")
            plt.colorbar(imn, ax=axn, label="# passes")
            st.pyplot(fign, clear_figure=True)
            st.caption(
                "Pure coverage: how many times each cell was struck. The "
                "bright lattice is the **disc-to-disc overlap** (cells reached "
                "by more than one disc); dark = gaps / thin coverage.")

    # Right: total cleaning dose (bar·s).
    with hm_r:
        vmax = vmax_shared if vmax_shared is not None else max(
            float(np.percentile(strip[strip > 0], 99))
            if (strip > 0).any() else 1.0, 1e-6)
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        im = ax.imshow(strip, extent=extent, aspect="auto",
                       cmap="viridis", vmin=0, vmax=vmax)
        _draw_core_overlay(ax)
        ax.set_xlabel("Across-track (mm)")
        ax.set_ylabel("Along-track (mm)")
        ax.set_title(f"{label}Accumulated exposure (bar·s)")
        plt.colorbar(im, ax=ax, label="bar·s = pressure × dwell time")
        st.pyplot(fig, clear_figure=True)
        st.caption(
            "Pressure **× dwell time**, summed over passes — units of bar·s, "
            "**not** the same as the delivered-pressure gate (bar). The 18 bar "
            "jet only dwells ~0.5 ms per pass, so accumulated exposure is < 1 "
            "bar·s even though each hit is 18 bar. Use this for the *pattern* "
            "(overlap/gaps weighted by intensity), not as a pressure value.")

    # ---- Advanced: delivered-pressure map + binary cleaned map -----------
    with container.expander("Advanced: delivered-pressure & cleaned maps"):
        pmap_full = m.get("pressure_strip")
        if pmap_full is not None:
            pmap = (pmap_full[cy0:cy1, cx0:cx1] if s.steady_state_only
                    else pmap_full)
            pmap_y = ((cy1 - cy0) if s.steady_state_only
                      else pmap_full.shape[0]) * cell
            ext = [-array_span_x / 2, array_span_x / 2, pmap_y, 0]
            figp, axp = plt.subplots(figsize=(8, 3.0))
            pvmax = max(float(pmap.max()), s.removal_pressure_bar * 1.2, 1e-6)
            imp = axp.imshow(pmap, extent=ext, aspect="auto", cmap="inferno",
                             vmin=0, vmax=pvmax)
            if pmap.max() >= s.removal_pressure_bar > pmap.min():
                axp.contour(pmap, levels=[s.removal_pressure_bar],
                            extent=[ext[0], ext[1], ext[3], ext[2]],
                            colors="#39ff14", linewidths=1.0)
            axp.set_xlabel("Across-track (mm)")
            axp.set_ylabel("Along-track (mm)")
            axp.set_title(
                f"{label}Delivered pressure (bar) — green = "
                f"{s.removal_pressure_bar:.0f} bar gate")
            plt.colorbar(imp, ax=axp, label="bar delivered")
            st.pyplot(figp, clear_figure=True)
            st.caption(
                "Peak pressure delivered per cell. It saturates near the "
                "rated delivered pressure almost everywhere a cell is "
                "touched, so it reads flat — use it to check the gate is met "
                "across the field, not to see ring structure (that's the "
                "total-dose map above).")

        # Binary cleaned map: pressure gate AND passes gate.
        pmap2 = m.get("pressure_strip")
        pass2 = m.get("passes_strip")
        if pmap2 is not None and pass2 is not None:
            pr = pmap2[cy0:cy1, cx0:cx1] if s.steady_state_only else pmap2
            pa = pass2[cy0:cy1, cx0:cx1] if s.steady_state_only else pass2
            cleaned = ((pr >= s.removal_pressure_bar)
                       & (pa >= s.min_passes)).astype(np.float32)
            ry = ((cy1 - cy0) if s.steady_state_only
                  else pmap2.shape[0]) * cell
            fig_t, ax_t = plt.subplots(figsize=(8, 2.8))
            ax_t.imshow(cleaned,
                        extent=[-array_span_x / 2, array_span_x / 2, ry, 0],
                        aspect="auto", cmap="Greens", vmin=0, vmax=1)
            ax_t.set_xlabel("Across-track (mm)")
            ax_t.set_ylabel("Along-track (mm)")
            ax_t.set_title(
                f"{label}Cleaned cells (green = gate met AND "
                f"≥ {s.min_passes} passes)")
            st.pyplot(fig_t, clear_figure=True)
