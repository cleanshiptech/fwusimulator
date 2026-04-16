# PWU Coverage Simulator

Streamlit tool that simulates how the ROV [[Domains/product/robot|pressure washing unit]]
(PWU) deposits jet pressure on the hull as it traverses, on a 1×1 cm grid.

## What it computes

The hull is split into 1×1 cm cells. As the ROV moves at the chosen
traverse speed and each disc spins at the chosen RPM, every nozzle deposits
its instantaneous jet pressure (bar) onto the cells inside its footprint,
weighted by the simulation time-step. The accumulated quantity per cell is
**integrated pressure exposure** in **bar·seconds (bar·s)**.

This is a useful proxy for cleaning energy: a cell needs enough jet
pressure for long enough to lift coating-grade biofilm. Calibrate the
"clean threshold" (bar·s) against a known field-cleaning trial.

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
- Footprint diameter on hull: linear with pressure (60→80 mm) or manual
- Pressure distribution within footprint: uniform or Gaussian

Hull strip
- Strip length to simulate (mm), cleaning threshold (bar·s)

## Outputs

- Live top-down schematic of the disc array and nozzle impact rings
- Side cross-section showing nozzle standoff and cant
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
stated footprint diameter.

The Hull-tab model ignores bow, stern, superstructure, and appendages;
total wetted area is therefore a lower bound. Real-world cleaning time
also includes docking/positioning overhead, overlap, and transit between
strings — apply an appropriate multiplier (typically 1.3 – 1.6×) when
quoting customer-facing estimates.
