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

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

The model is geometric and time-integrated. It does not yet capture jet
pressure decay along the jet axis, hull curvature, biofouling resistance,
or ROV pitch/roll. The cant correction shifts the impingement ring inward
by `standoff · tan(cant)`. The Gaussian footprint option uses 2σ at the
stated footprint diameter.
