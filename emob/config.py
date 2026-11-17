"""Fixed constants of the study (decisions D1–D8, approved 2026-11-05).

Every value here is documented in knowledge_base/02_design_rationale.md.
"""
import os
from pathlib import Path

# Paths can be overridden on the server: EMOB_RAW_2022, EMOB_PROCESSED
PROJECT = Path(r"D:\Projects\EmoB")
RAW_2022 = Path(os.environ.get("EMOB_RAW_2022", PROJECT / "data" / "raw" / "capture_2022-01-26"))
PROCESSED = Path(os.environ.get("EMOB_PROCESSED", PROJECT / "data" / "processed" / "v1"))

# --- sampling -------------------------------------------------------------
FS = 50                 # Hz, grid of the 2022 CSVs (ts_SQ step = 20)
TS_STEP = 20            # expected ts_SQ increment
STEP = 5                # samples per output step -> 0.1 s
CENTRE = 2              # centre sample inside a step (5*i + 2)

# --- input channels (main configuration M) --------------------------------
# column names as they appear in the CSV (pandas de-duplicates repeated names)
SENSORS = {
    "hand": ("ts_SB - Main droite - ACC", ["x_sb.6", "y_sb.6", "z_sb.6"]),
    "belt": ("ts_SB - Ceinture - ACC", ["x_sb", "y_sb", "z_sb"]),
}
SENSOR_ORDER = ["hand", "belt"]
# pocket phone (config P / alternative main input), stored separately in the .npz
EXTRA_SENSORS = {
    "phone_acc": ("ts_SF - poche droite - ACC", ["x_sf", "y_sf", "z_sf"]),
    "phone_gyr": ("ts_SF - poche droite - GYR", ["x_sf.2", "y_sf.2", "z_sf.2"]),
}
EXTRA_ORDER = ["phone_acc", "phone_gyr"]
# A sensor block whose axes take fewer distinct values than this over the whole recording carries
# no usable signal (2022 user1/2/4: SB ACC rounded to 0.01 with 3-7 distinct values; normal
# blocks have > 1 500). Found 2026-11-05.
MIN_UNIQUE = 50

# --- label columns ----------------------------------------------------------
FAMILY_COLS = ["etiq standard", "etiq mvt_corps", "etiq mvt_bras", "etiq mvt_jambe"]
CAT_COL = "etiq categorie"
ACT_COL = "etiq activite"
BODY_COL = "etiq mvt_corps"
LEG_COL = "etiq mvt_jambe"
PROTOCOL_MARKERS = {"Debut", "Fin"}

# --- head L2: 20 fine classes (proposal §3.2) -------------------------------
L2_CLASSES = [
    "Lever_Bras", "Baisser_Bras", "Lever_2_Bras", "Baisser_2_Bras", "Autre_Mouvement_Bras",
    "Marcher", "Lever_Jambe", "Baisser_Jambe", "Autre_Mouvement_Jambe",
    "Debout_Assis", "Assis_Debout", "Assis_Couche", "Couche_Assis",
    "Debout_Penche", "Penche_Debout", "Debout_Agenou", "Agenou_Debout",
    "Debout_Accroupi", "Accroupi_Debout", "Autre_Transition",
]

# --- head L1: 7 coarse classes (proposal §3.3) ------------------------------
L1_CLASSES = ["ARM", "WALK", "LIE_TRANSITION", "MODERATE", "SIT_TRANSITION", "LEG", "OTHER_TRANSITION"]
L2_TO_L1 = {
    "Lever_Bras": "ARM", "Baisser_Bras": "ARM", "Lever_2_Bras": "ARM", "Baisser_2_Bras": "ARM",
    "Autre_Mouvement_Bras": "ARM",
    "Marcher": "WALK",
    "Assis_Couche": "LIE_TRANSITION", "Couche_Assis": "LIE_TRANSITION",
    "Debout_Penche": "MODERATE", "Penche_Debout": "MODERATE", "Debout_Agenou": "MODERATE",
    "Agenou_Debout": "MODERATE", "Debout_Accroupi": "MODERATE", "Accroupi_Debout": "MODERATE",
    "Debout_Assis": "SIT_TRANSITION", "Assis_Debout": "SIT_TRANSITION",
    "Lever_Jambe": "LEG", "Baisser_Jambe": "LEG", "Autre_Mouvement_Jambe": "LEG",
    "Autre_Transition": "OTHER_TRANSITION",
}

# --- head S: 6 posture states (proposal §3.4) -------------------------------
S_CLASSES = ["Debout", "Assis", "Couche", "Penche", "Agenou", "Accroupi"]
S_ENGLISH = ["STANDING", "SITTING", "LYING", "BENT", "KNEELING", "SQUATTING"]
POSTURE_TRANSITIONS = [c for c in L2_CLASSES if c in {
    "Debout_Assis", "Assis_Debout", "Assis_Couche", "Couche_Assis", "Debout_Penche", "Penche_Debout",
    "Debout_Agenou", "Agenou_Debout", "Debout_Accroupi", "Accroupi_Debout"}]
OTHER_TRANSITION = "Autre_Transition"

IGNORE = -1  # label index for "no supervision at this step"


def transition_states(name: str) -> tuple[str, str]:
    """'Debout_Assis' -> ('Debout', 'Assis')."""
    a, b = name.split("_")
    return a, b
