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

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

from fwu.constants import KNOTS_TO_MPS, HULL_SHAPES
from fwu.model import Scenario, scenario_full_key, hull_wetted_areas
from fwu.sim import simulate_pressure, full_traversal_limits
from fwu.plots import (
    plot_topdown,
    plot_side,
    plot_spray_profile,
    plot_single_disc_coverage,
    plot_footprint_sensitivity,
    plot_motion_fast,
    plot_hull_section,
)
from fwu.ui import scenario_controls, render_result

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
    "Simulates hull cleaning from the rotating jet array. Cleaning needs "
    "enough **impact at the hull** (the chosen measure — wall shear, "
    "stagnation or mean pressure — set in the sidebar) **plus** enough "
    "**passes**. **New here?** Open the *ℹ️ How to read this tab* panel at the "
    "top of each tab, and the *🎯 Calibration status* in the System tab."
)

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

    tab_overview, tab_impact, tab_motion, tab_hull, tab_system = st.tabs(
        ["Overview", "Impact simulation", "Motion simulation",
         "Hull simulation", "System & impact"])

    # =============================================================
    # TAB 0 — Overview (how it all fits together)
    # =============================================================
    with tab_overview:
        st.subheader("How this system cleans — the whole picture")
        st.markdown(
            "A patch of hull is cleaned when it gets **enough impact** (the "
            "jet hits hard enough) **AND enough coverage** (it is struck "
            "often enough). Two independent gates — miss either and it is not "
            "clean. Everything you can change feeds one of these two."
        )
        st.markdown("### Cleaning  =  Impact  ×  Coverage")

        ov_imp, ov_cov = st.columns(2)
        with ov_imp:
            st.markdown("#### 💥 Impact — *how hard the jet hits*")
            _meas = scen.cleaning_measure
            _val = scen.cleaning_intensity()
            _unit = "kPa" if _meas.startswith("Wall") else "bar"
            st.metric(f"Delivered {_meas.lower()}", f"{_val:.1f} {_unit}",
                      help="The chosen impact measure at your standoff — must "
                           "clear the fouling's removal threshold.")
            st.markdown(
                "**Set directly by**\n"
                f"- **Standoff** ({scen.standoff_mm} mm) — *the dominant "
                "lever*; ~90% of impact is gone by 25 mm.\n"
                f"- **Jet velocity** ({scen.jet_exit_velocity:.0f} m/s) at the "
                "exit, which decays to the hull.\n\n"
                "**…which is set by**\n"
                f"- Pump **flow** ({scen.total_flow_lpm:.0f} L/min, ≤ "
                f"{scen.pump_flow_cap_lpm:.0f} cap) ÷ **nozzles** "
                f"({scen.n_nozzles_total}) ÷ **bore** "
                f"({scen.nozzle_exit_mm:.2f} mm) → exit velocity = Q/A. "
                "Pressure is the *consequence*, not an input.\n"
                "- **Jet decay** to the hull (core length, K, half-angle).\n\n"
                "**Trade-off:** at fixed pump flow, more nozzles or a wider "
                "bore *lowers* the velocity per jet (flow is split / spread). "
                "The flow ceiling (pumps) and the hose pressure ceiling bound "
                "the top end — see the System tab."
            )
        with ov_cov:
            st.markdown("#### 🔁 Coverage — *how often each spot is hit*")
            st.metric("Footprint on hull", f"{scen.footprint_dia():.1f} mm",
                      help="Jet width at impact — wider covers more area but "
                           "spreads the energy thinner (lower impact).")
            st.markdown(
                "**Set directly by**\n"
                f"- **Disc rotation** ({scen.rpm} rpm) — passes per spot per "
                "second.\n"
                f"- **ROV forward speed** ({scen.rov_speed_kn:.1f} kn) — "
                "faster = fewer passes + less dwell.\n"
                f"- **Footprint width** ({scen.footprint_dia():.1f} mm) — "
                "wider rings overlap more readily.\n\n"
                "**…which is set by**\n"
                f"- **Nozzles/disc** & **array geometry** (disc pitch, row "
                "pitch, {0} discs) — whether rings overlap or leave gaps.\n"
                "- Exit dia + standoff × spread → footprint width.\n\n"
                "**Trade-off:** wider footprint helps coverage but *hurts* "
                "impact — the two pull against each other.".format(
                    scen.n_row1 + scen.n_row2)
            )

        st.info(
            "**Footprint width sits in the middle:** a wider jet covers more "
            "area (good for coverage) but spreads the same energy over more "
            "hull (bad for impact). Standoff drives both — closer = harder "
            "impact *and* a smaller, tighter footprint.")

        st.markdown("### ⛔ Constraints that bound the whole thing")
        st.markdown(
            "- **Hose pressure ceiling** — the supply hose WP caps pump "
            "pressure, so pressure is *not* a free lever (loss reduction is "
            "the only path to more).\n"
            "- **Umbilical drag** — at 2–3 kn the 300 m umbilical drag "
            "approaches its breaking load; the real limit on ROV speed is "
            "*drag*, not cleaning (System tab).\n"
            "- **Cleaning mechanism** — whether it is jet shear, stagnation "
            "pressure, or cavitation is **not yet confirmed**; pick the gating "
            "measure in the sidebar and calibrate (System tab → Calibration "
            "status).")
        st.caption(
            "Numbers above are your current scenario. The **Impact** and "
            "**Motion/Hull** tabs simulate coverage; the **System & impact** "
            "tab quantifies the impact chain and the operating constraints.")

    # =============================================================
    # TAB 1 — Impact simulation
    # =============================================================
    with tab_impact:
        with st.expander("ℹ️ How to read this tab"):
            st.markdown(
                "The **coverage simulation**: where the rotating jet array "
                "sweeps the hull, and how much of it gets cleaned.\n\n"
                "1. Set the array geometry, operating point and cleaning "
                "criterion in the **sidebar**, then click **Run full "
                "simulation**.\n"
                "2. **Cleaned / Partial / Untouched** split the swept area "
                "(they sum to 100%): *cleaned* = enough impact AND enough "
                "passes; *partial* = struck but under-gated; *untouched* = "
                "never struck.\n"
                "3. The **impact-pressure box** is how *hard* the jet hits "
                "(its own units); the **two heatmaps** show passes/cell and "
                "accumulated exposure (the lattice = disc overlap).\n"
                "4. **Single-disc coverage** below shows whether one disc's "
                "rings overlap or leave gaps.\n\n"
                "Grid resolution is auto-set from the footprint (Nyquist) so "
                "the result doesn't depend on the grid — see the sidebar.")

        # Row 1 — the short schematics, balanced heights.
        col_top, col_side, col_spray = st.columns(3)
        with col_top:
            st.subheader("Top-down view")
            st.pyplot(plot_topdown(scen), clear_figure=False)
        with col_side:
            st.subheader("Side view (one disc)")
            st.pyplot(plot_side(scen), clear_figure=False)
        with col_spray:
            st.subheader("Spray profile")
            st.pyplot(plot_spray_profile(scen), clear_figure=False)
            st.caption(
                "One nozzle jet at true mm scale (the side view is dominated "
                "by the disc width).")

        st.divider()

        # Row 2 — single-disc coverage map beside its gap metrics/verdict.
        st.subheader("Single-disc coverage — ring gap check")
        cov_map, cov_info = st.columns([1.0, 1.2])
        with cov_map:
            fig_cov, cov = plot_single_disc_coverage(scen)
            st.pyplot(fig_cov, clear_figure=True)
            st.caption(
                "Green = swept by ONE disc as it advances. A disc cleans only "
                "its **ring**, not a filled circle; adjacent discs/rows fill "
                "the rest. Continuous ring = gap-free; separated = gaps.")
        with cov_info:
            gk1, gk2 = st.columns(2)
            gk1.metric("Forward advance / rev",
                       f"{cov['advance_per_rev_mm']:.1f} mm")
            gk2.metric("Effective pitch", f"{cov['eff_pitch_mm']:.1f} mm",
                       help=f"= advance/rev ÷ {cov['n_nozzles']} nozzles. The "
                            "along-track spacing between successive nozzle "
                            "passes over a fixed point.")
            gk3, gk4 = st.columns(2)
            gk3.metric("Footprint", f"{cov['footprint_mm']:.1f} mm")
            _m = cov["overlap_margin_mm"]
            gk4.metric("Overlap margin", f"{_m:+.1f} mm",
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

        st.divider()
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
        with st.expander("ℹ️ How to read this tab"):
            st.markdown(
                "An **animation** of the jets sweeping the hull — a visual "
                "sanity check on the geometry, not a new calculation.\n\n"
                "- **Prepare** pre-renders the frames, then **Play** scrubs "
                "through the traversal; the slider jumps to any time.\n"
                "- *Hull frame* shows the cycloid trails as the array "
                "advances; *ROV frame* factors out the translation so the "
                "nozzle paths look like pure rosettes.\n"
                "- The optional **cumulative underlay** overlays the bar·s "
                "exposure built up so far.")

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
        with st.expander("ℹ️ How to read this tab"):
            st.markdown(
                "Scales the **per-strip cleaning rate** from the Impact tab "
                "up to a **whole-vessel cleaning time**.\n\n"
                "1. **Run the Impact simulation first** — this tab reads its "
                "cleaned-area rate.\n"
                "2. Enter the vessel particulars (LOA, beam, draft) and pick "
                "a hull-shape preset; it estimates the wetted area per side + "
                "bottom.\n"
                "3. Output: per-side and total cleaning time. It ignores bow/"
                "stern/appendages and docking/transit overhead — apply a "
                "1.3–1.6× multiplier for real quotes.")

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
        else:

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

    # =============================================================
    # TAB 4 — System & impact
    # =============================================================
    with tab_system:
        with st.expander("ℹ️ How to read this tab"):
            st.markdown(
                "This tab is the **system model**: it computes what the jet "
                "does to the hull from the pump flow, and shows the operating "
                "constraints — independent of *where* the array sweeps (that "
                "is the Impact tab).\n\n"
                "- **Operating point** — exit velocity, dynamic pressure, and "
                "the thrust the ROV must hold against.\n"
                "- **Impact vs standoff** — the three candidate cleaning "
                "measures as the jet decays. Standoff is the dominant lever "
                "(~90% of impact gone by 25 mm).\n"
                "- **Operational constraints** — the pressure budget down the "
                "umbilical and the umbilical drag, which is the real binding "
                "limit near 2–3 kn.\n\n"
                "Many jet constants (K, Cd, half-angle, Cf) are **assumed** — "
                "see the Calibration status below and the sidebar *Jet "
                "physics* expander.")

        st.subheader("Jet impact at the hull")
        st.caption(
            "What the jet actually does to the hull, from first principles: "
            "pump flow → per-nozzle exit velocity (v = Q/A) → submerged-jet "
            "decay over the standoff → impact. Three measures are shown "
            "because the cleaning mechanism is not yet confirmed; pick which "
            "one gates cleaning in the sidebar.")

        # --- Calibration status -------------------------------------------
        with st.expander("🎯 Calibration status — measured vs assumed", expanded=False):
            st.markdown(
                "**✓ Measured / specified** (trust these):\n"
                f"- Total flow {scen.total_flow_lpm:.0f} L/min, "
                f"{scen.n_nozzles_total} × {scen.nozzle_exit_mm:.2f} mm bore "
                f"nozzles, {scen.rpm} rpm, water density "
                f"{scen.water_density:.0f} kg/m³.\n\n"
                "**⚠ Assumed — handbook values, the main uncertainty:**\n"
                f"- Discharge coeff Cd = {scen.nozzle_cd:.2f} (derived — verify)\n"
                f"- Far-field decay K = {scen.decay_K:.1f}\n"
                f"- Potential core = {scen.core_factor:.1f} × exit dia "
                f"(= {scen.core_length_mm:.1f} mm)\n"
                f"- Jet half-angle = {scen.jet_half_angle_deg:.0f}°\n"
                f"- Wall skin-friction Cf = {scen.skin_friction_cf:.4f}\n\n"
                "**📋 Measure first (highest value):** a single-jet firing "
                "test against pressure film / dye-in-water at known standoffs "
                "directly gives the **real spread angle, core length and "
                "footprint** — replacing K, half-angle and Cd, which is where "
                "most of the model uncertainty sits. Edit any of these in the "
                "sidebar *Jet physics (calibratable)* expander.")

        # --- Operating point ----------------------------------------------
        o1, o2, o3, o4 = st.columns(4)
        o1.metric("Exit velocity", f"{scen.jet_exit_velocity:.0f} m/s",
                  help="v = Q/A per nozzle (continuity).")
        o2.metric("Exit dynamic pressure", f"{scen.pressure_bar:.0f} bar",
                  help="½ρv² at the nozzle exit — the jet's starting impact.")
        _q_m3s = scen.flow_per_nozzle_lpm / 1000.0 / 60.0
        _thrust = scen.water_density * _q_m3s * scen.jet_exit_velocity
        o3.metric("Thrust / jet", f"{_thrust:.0f} N",
                  help="ρ·Q·v reaction force of one jet.")
        o4.metric("Total reaction", f"{_thrust * scen.n_nozzles_total:.0f} N",
                  help=f"All {scen.n_nozzles_total} jets — the thrust the "
                       "ROV must hold against "
                       f"(≈ {_thrust * scen.n_nozzles_total / 9.81:.0f} kgf).")

        # --- Three-measure decay chart ------------------------------------
        xs = np.linspace(2, 60, 200)
        stag = [scen.stagnation_pressure_bar(x) for x in xs]
        mean = [scen.mean_impact_pressure_bar(x) for x in xs]
        shear = [scen.wall_shear_kpa(x) for x in xs]
        core = scen.core_length_mm

        figd, axd = plt.subplots(figsize=(8, 4))
        axd.axvspan(0, core, color="#cfe3f5", alpha=0.5, label="potential core")
        axd.plot(xs, stag, color="#1f77b4", label="Stagnation ½ρv² (bar)")
        axd.plot(xs, mean, color="#2ca02c", label="Mean force/area (bar)")
        axd.set_xlabel("Standoff to hull (mm)")
        axd.set_ylabel("Pressure (bar)")
        axd.axvline(scen.standoff_mm, color="#888", ls=":", lw=1.0)
        axd.set_title("Impact vs standoff — all three measures")
        # shear on a twin axis (kPa)
        axs2 = axd.twinx()
        axs2.plot(xs, shear, color="#d62728", label="Wall shear τ (kPa)")
        axs2.set_ylabel("Wall shear (kPa)", color="#d62728")
        axs2.tick_params(axis="y", labelcolor="#d62728")
        # combined legend
        l1, lb1 = axd.get_legend_handles_labels()
        l2, lb2 = axs2.get_legend_handles_labels()
        axd.legend(l1 + l2, lb1 + lb2, fontsize=8, loc="upper right")
        axd.grid(True, alpha=0.3)
        st.pyplot(figd, clear_figure=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Stagnation @ standoff",
                  f"{scen.stagnation_pressure_bar():.1f} bar",
                  help="Centreline ½ρv_c² — head-on impact; hard-fouling "
                       "fracture.")
        m2.metric("Mean @ standoff",
                  f"{scen.mean_impact_pressure_bar():.1f} bar",
                  help="Jet force ÷ footprint area — bulk normal load; falls "
                       "fast as the footprint grows.")
        m3.metric("Wall shear @ standoff",
                  f"{scen.wall_shear_kpa():.1f} kPa",
                  help="τ = Cf·½ρv_c² — tangential scrub; the shear-removal "
                       "mechanism. **Gating driver by default.**")
        st.caption(
            f"At {scen.standoff_mm} mm standoff (core ≈ {core:.1f} mm). "
            "Stagnation & shear share the v² shape; mean falls faster because "
            "the footprint grows with standoff. **~90% of impact is gone by "
            "~25 mm** — standoff is the dominant lever.")

        st.divider()

        # --- What cleaning requires (backward from the hull target) -------
        st.subheader("What cleaning this fouling requires")
        _unit = "kPa" if scen.cleaning_measure.startswith("Wall") else "bar"
        _req_subsea = scen.required_subsea_pressure_bar()
        _req_topside = scen.required_topside_pressure_bar()
        _max_subsea = scen.max_subsea_pressure_bar
        _reachable = _req_topside <= scen.hose_pressure_ceiling_bar + 1e-6 \
            and _req_subsea <= _max_subsea + 1e-6
        st.caption(
            "Your real target is the **impact at the hull**. Working back from "
            f"the **{scen.removal_pressure_bar:.1f} {_unit}** removal "
            f"threshold ({scen.cleaning_measure.lower()}) at "
            f"{scen.standoff_mm} mm standoff:")
        r1, r2, r3 = st.columns(3)
        r1.metric("Subsea pressure needed", f"{_req_subsea:.0f} bar",
                  help="Regulate the subsea (cleaning) pressure to about this "
                       "to just clear the fouling at your standoff.")
        r2.metric("Topside to dial", f"{_req_topside:.0f} bar",
                  delta=("over hose ceiling"
                         if _req_topside > scen.hose_pressure_ceiling_bar
                         else f"of {scen.hose_pressure_ceiling_bar:.0f} ceiling"),
                  delta_color="inverse"
                  if _req_topside > scen.hose_pressure_ceiling_bar else "off",
                  help="= subsea ÷ transmission ratio. What the topside gauge "
                       "should read.")
        r3.metric("Max reachable subsea", f"{_max_subsea:.0f} bar",
                  help=f"Highest subsea pressure your {scen.n_nozzles_total} "
                       "nozzles can reach (limited by pumps or the hose).")
        if _reachable:
            st.success(
                f"✅ **Reachable.** Regulate the subsea pressure up to "
                f"~{_req_subsea:.0f} bar (topside ~{_req_topside:.0f} bar) and "
                "the fouling clears at this standoff. You have headroom to "
                f"{_max_subsea:.0f} bar subsea.")
        else:
            # Diagnose: is the hose the limit, and would blanking help?
            _hose_caps = (scen.hose_pressure_ceiling_bar
                          * scen.pressure_transmission_ratio) <= _max_subsea + 1e-6
            st.warning(
                f"⚠ **Not reachable** at this config: cleaning needs "
                f"~{_req_subsea:.0f} bar subsea but you can only reach "
                f"~{_max_subsea:.0f} bar. "
                + ("The **hose ceiling** is the limit — blanking discs will "
                   "NOT help (max reachable is the same with fewer nozzles); "
                   "you need a higher-rated hose, lower umbilical loss, or a "
                   "shorter standoff."
                   if _hose_caps else
                   "You're **pump-flow limited** — **blanking discs would "
                   "help** (fewer nozzles let the same pump flow reach a "
                   "higher pressure), until the hose ceiling binds."))
        st.caption(
            "This is the loop closed: the cleaning gate (hull impact) → the "
            "subsea pressure to regulate to → the topside to dial → whether "
            "your hardware can reach it. Shorter standoff is the cheapest "
            "lever (impact rises fast as you approach the jet core).")

        # --- Two-tier: hydrodynamic removal vs adhesion bond --------------
        with st.expander(
                "🦠 Fouling thresholds — hydrodynamic removal vs adhesion bond"):
            _shear = scen.wall_shear_kpa()
            _stag_kpa = scen.stagnation_pressure_bar() * 100.0
            st.markdown(
                "Two **different mechanisms** set how hard fouling is to "
                "remove — and your jet sits between them, which is why a "
                "barnacle's **body comes off but its base stays**:\n\n"
                "**1. Hydrodynamic removal** (the body / soft fouling) — the "
                "flow's shear + stagnation lift and lever the *protruding* "
                "organism off. Low thresholds:\n"
                "- Soft biofouling removed at **0.01–0.28 kPa** wall shear "
                "(microalgae, sporelings).\n"
                "- Practical in-water cleaning runs **≤ ~1.3 kPa** wall shear "
                "/ ~1.7 bar stagnation without coating damage.\n"
                f"- *Your jet delivers ~{_shear:.1f} kPa shear* → clears the "
                "body comfortably.\n\n"
                "**2. Adhesion bond** (the cemented base plate) — flat, no "
                "leverage, so removal needs the full bond strength:\n"
                "- Barnacle base on **silicone foul-release: 17–55 kPa** "
                "(Kim et al. 2008).\n"
                "- Barnacle base on a **hard substrate (steel/epoxy): "
                "0.5–2 MPa = 500–2000 kPa** (ASTM/FIT shear-adhesion test).\n"
                f"- *Your jet's peak stagnation is ~{_stag_kpa:.0f} kPa at the "
                "centreline, but the **mean** over the whole base is far less* "
                "→ on a hard hull the base survives.\n\n"
                "**So:** the cleaning gate above uses the *hydrodynamic* "
                "thresholds (what removes the body). Getting the **base** off "
                "a hard hull needs ~100× more and is bond-limited — a "
                "mechanical/standoff problem, not a pressure one.")

        st.divider()

        # --- Operational constraints sub-section --------------------------
        st.subheader("Operational constraints")
        st.caption(
            "The binding limits are not at the hull — they are the pressure "
            "budget down the umbilical and the umbilical drag. Shown here so "
            "the friction-vs-drag trade-off is visible.")

        # Pressure follows the flow: the commanded flow needs this nozzle
        # (subsea) pressure, which implies this topside via the MEASURED
        # transmission ratio (~0.57 across two systems). Pressure is a
        # consequence here, not an input.
        _subsea = scen.subsea_pressure_bar      # nozzle demand from the flow
        _ratio = scen.pressure_transmission_ratio
        _topside = scen.topside_pressure_bar    # = subsea ÷ ratio
        _loss = _topside - _subsea
        b1, b2, b3 = st.columns(3)
        b1.metric("Subsea (nozzle) needed", f"{_subsea:.0f} bar",
                  help="½ρ(v/Cd)² — the pressure at the manifold required to "
                       "push the commanded flow through the nozzles.")
        b2.metric("Transmission ✓", f"{_ratio * 100:.0f} %",
                  delta=f"+{_loss:.0f} bar line loss", delta_color="inverse",
                  help="Subsea ÷ topside — MEASURED ~57% across two systems "
                       "(SSO3 + one other, Mar 2023–Mar 2025).")
        b3.metric("Topside required",
                  f"{_topside:.0f} bar",
                  delta="over hose ceiling" if scen.pressure_ceiling_exceeded
                  else f"of {scen.hose_pressure_ceiling_bar:.0f} ceiling",
                  delta_color="inverse" if scen.pressure_ceiling_exceeded
                  else "off",
                  help="Pump-side pressure the flow demands = subsea ÷ "
                       "transmission. Bounded by the hose/relief ceiling.")
        st.caption(
            f"Pressure is the **consequence of the flow**: {scen.total_flow_lpm:.0f} "
            f"L/min needs {_subsea:.0f} bar at the nozzles, which (÷ the "
            f"measured {_ratio:.2f} transmission) needs {_topside:.0f} bar "
            "topside. The two-system data sets the ~0.57 ratio — replacing "
            "the old assumed friction model; the ~42 bar 'unexplained' gap "
            "was largely this proportional loss. Real scatter means it ranges "
            "around this line.")

        # Delivered flow → velocity, with the limiting ceiling.
        fl1, fl2, fl3 = st.columns(3)
        _cmd, _del = scen.total_flow_lpm, scen.delivered_flow_lpm
        fl1.metric("Delivered flow", f"{_del:.0f} L/min",
                   delta=(f"−{_cmd - _del:.0f} throttled"
                          if scen.flow_throttled else None),
                   delta_color="inverse",
                   help=f"Commanded {_cmd:.0f}, clamped by the pump cap "
                        f"({scen.pump_flow_cap_lpm:.0f}) and the hose ceiling "
                        f"({scen.hose_allowed_flow_lpm:.0f}).")
        fl2.metric("Exit velocity", f"{scen.jet_exit_velocity:.0f} m/s",
                   help="v = Q/A per nozzle on the DELIVERED flow.")
        fl3.metric("Limiting ceiling",
                   {"none": "—", "pump flow": "Pump flow",
                    "hose pressure": "Hose pressure"}[scen.limiting_ceiling],
                   help="Which constraint binds the delivered flow.")
        # Design-point note: full pump flow ↔ the resulting nozzle pressure.
        _A_n = scen.nozzle_exit_area_m2 * scen.n_nozzles_total
        _v_full = (scen.pump_flow_cap_lpm / 1000.0 / 60.0) / max(_A_n, 1e-12)
        _p_full = 0.5 * scen.water_density \
            * (_v_full / max(scen.nozzle_cd, 1e-6)) ** 2 / 1e5
        st.caption(
            f"Design point: **full pump flow ({scen.pump_flow_cap_lpm:.0f} "
            f"L/min) → ~{_p_full:.0f} bar at the nozzle** through "
            f"{scen.n_nozzles_total} × {scen.nozzle_exit_mm:.2f} mm — the bore "
            "is sized to the pump capacity. The hose ceiling allows up to "
            f"**{scen.hose_allowed_flow_lpm:.0f} L/min** at this nozzle count.")
        if scen.limiting_ceiling == "hose pressure":
            st.warning(
                f"⚠ **Hose-pressure limited.** Delivered flow is throttled to "
                f"{_del:.0f} L/min — the topside the flow wants exceeds the "
                f"{scen.hose_pressure_ceiling_bar:.0f} bar hose/relief ceiling, "
                "so the relief bypasses. **Blanking discs/nozzles will NOT "
                "raise per-nozzle impact** in this regime (you're not "
                "flow-limited) — it narrows the swath at the same intensity. "
                "Raise impact via a wider bore, lower umbilical loss, or a "
                "higher-rated hose.")
        elif scen.limiting_ceiling == "pump flow":
            st.info(
                f"ℹ️ **Pump-flow limited** at {_del:.0f} L/min. Here blanking "
                "nozzles *would* raise per-nozzle impact (same flow, fewer "
                "jets) — until the hose ceiling binds.")

        # Umbilical drag vs ROV speed, with MBL and a fairing toggle.
        st.markdown("**Umbilical drag vs ROV speed**")
        _fairing = st.checkbox(
            "Fairing fitted (Cd 1.2 → 0.6)", value=False, key="sys_fairing",
            help="A bare cylinder has Cd≈1.2; a faired/streamlined umbilical "
                 "roughly halves drag without any pressure trade-off.")
        _cd_umb = 0.6 if _fairing else 1.2
        _D, _L, _rho_sw, _MBL = 0.123, 300.0, 1026.0, 60_000.0
        sp_kn = np.linspace(0.2, 3.5, 100)
        sp_ms = sp_kn * KNOTS_TO_MPS
        drag = 0.5 * _rho_sw * _cd_umb * (_D * _L) * sp_ms ** 2  # N
        figu, axu = plt.subplots(figsize=(8, 3.2))
        axu.plot(sp_kn, drag / 1000.0, color="#1f77b4",
                 label=f"drag (Cd={_cd_umb})")
        axu.axhline(_MBL / 1000.0, color="#d62728", ls="--",
                    label=f"umbilical MBL {_MBL/1000:.0f} kN")
        _cur_drag = 0.5 * _rho_sw * _cd_umb * (_D * _L) * \
            (scen.rov_speed_kn * KNOTS_TO_MPS) ** 2
        axu.axvline(scen.rov_speed_kn, color="#888", ls=":")
        axu.scatter([scen.rov_speed_kn], [_cur_drag / 1000.0],
                    color="#888", zorder=5)
        axu.set_xlabel("ROV speed (knots)")
        axu.set_ylabel("Umbilical drag (kN)")
        axu.legend(fontsize=8)
        axu.grid(True, alpha=0.3)
        st.pyplot(figu, clear_figure=True)
        st.caption(
            f"At {scen.rov_speed_kn:.1f} kn the 300 m × Ø123 mm umbilical drag "
            f"is **{_cur_drag/1000:.1f} kN** (Cd={_cd_umb}) vs the "
            f"{_MBL/1000:.0f} kN MBL. Drag ∝ speed² and dominates all other "
            "forces — it, not pump pressure, is the binding operational limit "
            "near 2–3 kn. A fairing relieves it with no pressure cost.")

else:
    with st.sidebar:
        st.divider()
        st.caption(
            "Both scenarios start from the SAME defaults — change one thing "
            "in B to isolate its effect, then click Run.")
    col_a_ctrl, col_b_ctrl = st.sidebar.columns(2)
    with col_a_ctrl.expander("Scenario A", expanded=True):
        scen_a = scenario_controls("A", Scenario(), col_a_ctrl)
    # B starts identical to A so a single changed parameter is a clean
    # one-variable comparison (e.g. 3 vs 4 nozzles at the same pump flow —
    # otherwise a differing flow masks the derived-pressure effect).
    with col_b_ctrl.expander("Scenario B", expanded=True):
        scen_b = scenario_controls("B", Scenario(), col_b_ctrl)

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
            ("cleaned_pct", "Cleaned area (%)", "+"),
            ("stagnation_pressure_bar", "Impact pressure (bar)", "+"),
            ("median_passes", "Median passes", "+"),
            ("missed_pct", "Untouched (%)", "-"),
            # Accumulated exposure (bar·s) — diagnostic, shown last.
            ("p50_bs", "Median exposure (bar·s)", "+"),
            ("mean_bs", "Mean exposure (bar·s)", "+"),
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
