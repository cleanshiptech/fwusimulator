# PWU Coverage Simulator

Streamlit tool that simulates how the ROV [[Domains/product/robot|pressure washing unit]]
(PWU) deposits jet pressure on the hull as it traverses, on a 1×1 cm grid.

## What it computes

The hull is split into a grid (default 5 mm cells). As the ROV moves at the
chosen traverse speed and each disc spins at the chosen RPM, the sim tracks,
per cell, how many nozzle **passes** it received and how much **integrated
jet-pressure exposure** (bar·seconds) it accumulated.

The headline **Cleaned area** comes from the intensity-gate + dose criterion
below. The accumulated bar·s is a **dose diagnostic only** — it lives in a
collapsed *Advanced: exposure dose* expander and never decides cleaning (a
high bar·s below the intensity gate still cleans nothing). It is kept for
inspecting track thinness and overlap quality, not as the verdict.

## Evaluating cleaning effectiveness (intensity gate + dose)

Raw bar·s alone is a poor measure of cleaning power because it trades
pressure against time linearly — it says "10 bar for 1 s" equals "200 bar
for 0.05 s", which is physically wrong: below a fouling-specific intensity
threshold you remove nothing no matter how long you dwell. The headline
**Cleaned area** KPI therefore uses a two-part criterion:

1. **Intensity gate.** The jet's **stagnation pressure delivered at the
   hull** must exceed a removal threshold for the fouling type. A submerged
   free jet holds its nozzle pressure within a potential core (length ≈
   6 × exit diameter), then its centreline pressure decays as
   `(core_len / standoff)²`. So delivered pressure — *not* nozzle pressure —
   is what cleans, and **standoff** and **nozzle exit diameter** are the
   dominant levers (e.g. 100 bar at the nozzle through a 1.3 mm exit at
   18 mm standoff delivers only ~19 bar at the hull; shorten the standoff to
   8 mm and it delivers ~95 bar). Below the gate, cleaned area is 0 %.
2. **Passes gate.** Among cells that clear the impact-pressure gate, a cell
   counts as cleaned once it receives at least the **minimum number of
   passes**.

**Two quantities, two units — do not confuse them:**
- **Impact pressure** (bar) — how *hard* the jet hits at any instant. This
  is the gate. Shown in its own info box.
- **Accumulated exposure** (bar·s) — pressure **× dwell time**, summed over
  passes. Because the jet sweeps at ~12 m/s it dwells only ~0.5 ms on a spot
  per pass, so an 18 bar jet accumulates well under 1 bar·s. These are
  different dimensions (bar vs bar·s); a small bar·s next to an 18 bar impact
  pressure is expected, not a contradiction.

**How to compare configurations.** Calibrate the removal threshold to a run
you *know* cleans your fouling (the sidebar shows the impact pressure of the
current jet live — set the threshold to that). Then change a parameter: if
impact pressure stays above the gate and cleaned-area rises, that change
cleans *more*. Fouling presets (soft biofilm / light weed / hard calcareous)
seed typical removal thresholds.

**Primary heatmaps (side by side):**
- **Passes per cell** — pure geometric coverage. The bright lattice is the
  disc-to-disc overlap; dark is gaps / thin coverage.
- **Accumulated exposure (bar·s)** — the same lattice weighted by how hard
  each pass hit.

The **delivered-pressure map** (which saturates to a near-flat slab, since
peak pressure is ~the same everywhere a cell is touched) and the **binary
cleaned map** live in the Advanced expander.

## Parameters

Array geometry
- Array width (mm), front/back row disc count, row pitch, optional half-pitch stagger

Disc & nozzles
- Disc diameter, nozzles per disc, nozzle radius from disc centre,
  cant angle toward centre (degrees), nozzle standoff to hull,
  optional adjacent-disc counter-rotation

Operating point
- Disc RPM, ROV traverse speed (knots), jet pressure (bar),
  nozzle exit diameter (mm)

Footprint model
- Footprint diameter on hull, three modes:
  - **Physical jet (default)** — the impact zone is set by the spreading
    free jet: `d_fp = nozzle_exit_dia + 2·standoff·tan(spread_half_angle)`.
    Both the **nozzle exit diameter** and the **distance to the hull
    (standoff)** drive the footprint, with a tunable jet spread half-angle.
  - Linear with pressure (60→80 mm) — legacy curve, pressure only.
  - Manual override — fixed diameter.
- Pressure distribution within footprint: uniform or Gaussian
- Hull grid resolution: 10 / 5 / 2 / 1 mm per cell (default 5 mm). Finer
  grids resolve the small (mm-scale) physical footprint; the deposit step
  is FFT-convolution-based, so cost is independent of footprint size.

Hull strip
- Strip length to simulate (mm), cleaning threshold (bar·s)

## Outputs

- Live top-down schematic of the disc array and nozzle impact rings
- Side cross-section drawing the spreading jet cone from the nozzle exit
  diameter down to the hull footprint
- **Impact zone vs nozzle exit & standoff** chart (physical-jet mode):
  footprint diameter vs standoff for several exit diameters, plus the
  relative cleaning-energy density (∝ 1/footprint area)
- Heatmap of bar·s exposure per cell across the hull strip
- Binary "cleaned" map for cells exceeding the threshold
- Histogram of exposure values
- KPIs: cleaned area %, untouched area %, median and p10/p90 exposure

## Tabs

As of `app_fast_v4.py` the app is organised into three tabs:

1. **Impact simulation** — schematics plus the original "Run full
   simulation" button. Results are cached in `st.session_state` so the
   Hull tab can consume them.
2. **Motion simulation** — the Prepare/Play/Stop pre-rendered animation
   (hull or ROV frame, segment selector, stable-camera playback).
3. **Hull simulation** — vessel-level cleaning time estimate. Inputs:
   LOA, beam, draft, plus a visual midship-section selector (full
   / typical / fine). Outputs: per-side wetted area, per-side time, and
   total cleaning time. Needs a cached Impact result to run.

The Hull tab derives its cleaning rate as
`array_effective_width × ROV speed × coverage_fraction`, using the
Impact-tab coverage KPI (fraction of cells above the clean threshold).

## Hull geometry model

Midship cross-section is parametrised by:
- **Cm** — midship coefficient (not currently used for area, reserved)
- **r_rel** — bilge radius as a fraction of beam
- **deadrise** — keel rise angle (degrees, V-bottom only)
- **k_side** — per-side length multiplier (hull curvature in plan)
- **C_L** — length correction factor (0.96 – 1.02 by shape family)

Side area per side ≈ `LOA × draft × k_side × C_L`.
Bottom area ≈ `LOA × section_perimeter × C_L` where the section
perimeter is `beam + π·r` for flat-bottom hulls and
`2·(beam/2 − r)/cos(α) + π·r` for V-bottom hulls.

Three presets:
- **Full (block)**: Cm 0.98, bilge 10%, flat bottom — container, bulk, VLCC
- **Typical**: Cm 0.88, bilge 30%, flat bottom — general cargo, product tanker, ferry
- **Fine (V-keel)**: Cm 0.72, bilge 15%, 20° deadrise — naval, fast ferry, yacht

## Run

```bash
pip install -r requirements.txt
streamlit run app_fast.py   # or app_fast_v4.py
```

## Notes

The model is geometric and time-integrated. It does not yet capture jet
pressure decay along the jet axis, hull curvature, biofouling resistance,
or ROV pitch/roll. The cant correction shifts the impingement ring inward
by `standoff · tan(cant)`. The Gaussian footprint option uses 2σ at the
stated footprint diameter and is **energy-normalised** to deposit the same
total per pass as the uniform disc (it concentrates the same energy toward
the centre, raising the peak, rather than discarding ~half of it).

**Physical-jet footprint.** In the default footprint mode the impact-zone
diameter is `nozzle_exit_dia + 2·standoff·tan(spread_half_angle)`, so a
larger standoff or a wider nozzle exit grows the footprint, and the
deposited cleaning energy is diluted over `π·(d_fp/2)²`. A high-pressure
jet with a small exit at short standoff produces a tight, mm-scale
footprint that cleans a narrow kerf rather than an area — set the **hull
grid resolution** to 1–2 mm to resolve it (the coarse 10 mm grid collapses
a sub-cm footprint onto a single cell and reports ~0 % coverage).

**Performance.** Every nozzle hit deposits the same weighted footprint
stencil, so the per-cell exposure is computed as the 2-D convolution of a
binned hit-count grid with the stencil (`numpy.fft`). Runtime is therefore
independent of footprint size and timestep count, keeping 1–2 mm grids
responsive even with a large footprint.

**Time integration & dwell.** Exposure is `Σ pressure · Δt` per cell, i.e.
jet pressure integrated over the time the footprint dwells on that cell.
The dominant motion is the nozzle tip sweeping its impact ring at
`ω · r_impact` — at high RPM this is ~10–15 m/s, far faster than the ROV
traverse. The time step is bounded so the footprint advances at most half
its own width per step along that ring (`Δt ≤ 0.5 · footprint / (ω·r)`);
otherwise the jet would *skip* cells between samples, breaking the swept
track into a dotted line and undercounting per-cell exposure. The KPI
caption reports the per-step arc length, and a warning fires if the step
budget still can't resolve the path. Note: total deposited energy
(`mean bar·s`) is conserved regardless of step size — only the *spatial*
distribution (median/peak per cell, coverage) depends on resolving the
path. A fast, small, high-pressure jet legitimately deposits low bar·s on
any single spot: it cleans a thin kerf along the ring, not a filled area.

The Hull-tab model ignores bow, stern, superstructure, and appendages;
total wetted area is therefore a lower bound. Real-world cleaning time
also includes docking/positioning overhead, overlap, and transit between
strings — apply an appropriate multiplier (typically 1.3 – 1.6×) when
quoting customer-facing estimates.
