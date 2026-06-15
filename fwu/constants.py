"""Shared physical constants and hull-shape presets for the FWU simulator."""

from __future__ import annotations

KNOTS_TO_MPS = 0.514444

# Orifice flow constant: Q = ORIFICE_K · d² · √p, with Q in L/min, the
# nozzle exit diameter d in mm and the pressure p in bar. Fitted to the
# Denjet nozzle datasheet (546.xxx.A3 series) to <1% across bores 0.6–1.46 mm
# and pressures 40–300 bar. Used to DERIVE nozzle pressure from the fixed
# pump flow split across all nozzles (more/larger nozzles → lower pressure).
ORIFICE_K = 0.642

# Hull-grid resolution is now per-scenario (Scenario.cell_size_mm). This
# constant is kept only as a legacy default; the sim/render paths read the
# scenario value so the grid can be refined to 1–2 mm for small footprints.
CELL_SIZE_MM = 10.0


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
