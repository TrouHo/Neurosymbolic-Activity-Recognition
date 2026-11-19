"""01 Inventory: recordings, annotated time and usable devices per participant.

Usable block (raw): the reader of the build marks a row valid when its three axes are finite and
not all zero; a block whose axes take fewer than C.MIN_UNIQUE (50) distinct values over the whole
recording carries no signal and is unusable everywhere (P1, P2, P4 eMOB sensors: smoothed zeros).
Usable block (v2): the block is kept after sensor time alignment (params/sensor_offsets.json); its
availability is the share of the labelled region where it has a valid aligned sample.
Labelled region: from the first to the last row with a label on any track (body, arm, leg, protocol).
"""
from __future__ import annotations

import numpy as np

from emob import config as C
from inspection_data.common import PNAME, ROW_S, Context, Registry

DEVICE_OF = {"phone_acc": "phone", "phone_gyr": "phone_gyr", "hand": "hand", "belt": "waist", "leg": "leg"}


def run(ctx: Context, reg: Registry):
    reg.section = "01 Inventory"
    rec = []
    for u in ctx.users:
        P, raw, st = PNAME[u], ctx.raw[u], ctx.stats(u)
        n = raw["n"]
        rec.append(n * ROW_S)
        reg.add(f"inv.{P}.recording_s", n * ROW_S, "s", "raw", f"recording duration of {P}: CSV rows / 50")
        reg.add(f"inv.{P}.labelled_region_s", st["labelled_region_s"], "s", "v2",
                f"{P}: first to last row with a label on any track (after label alignment)")
        reg.add(f"inv.{P}.annotated_s", st["annotated_s"], "s", "v2",
                f"{P}: rows with a label on at least one track (after label alignment and masking)")
        lab = np.flatnonzero(np.any(np.stack([raw["fam"][c] != "" for c in C.FAMILY_COLS]), axis=0))
        region = slice(lab[0], lab[-1] + 1)
        for b in C.BLOCK_ORDER:
            ok = raw["blocks"][b][1]
            dev = DEVICE_OF[b]
            reg.add(f"inv.{P}.{dev}.raw_usable", bool(ok.any()), "yes/no", "raw",
                    f"{P} {dev}: block has a signal (finite, not all-zero rows, >= {C.MIN_UNIQUE} distinct values per axis)")
            reg.add(f"inv.{P}.{dev}.raw_valid_pct", 100 * ok[region].mean(), "% rows", "raw",
                    f"{P} {dev}: share of the raw labelled region with a valid sample")
            s = st["sensors"][b]
            reg.add(f"inv.{P}.{dev}.v2_used", s["shift_s"] is not None, "yes/no", "v2",
                    f"{P} {dev}: block kept after sensor time alignment (status: {s['status']})")
            reg.add(f"inv.{P}.{dev}.v2_available_pct", st["availability_in_labelled_region_pct"][b], "% rows", "v2",
                    f"{P} {dev}: share of the labelled region with a valid aligned sample")
            d2 = ctx.v2[u]
            centre = np.arange(len(d2["annotated"])) * C.STEP + C.CENTRE
            av = d2["avail50"][centre, C.BLOCK_ORDER.index(b)]
            reg.add(f"inv.{P}.{dev}.v2_available_annotated_pct", 100 * av[d2["annotated"]].mean(), "% steps", "v2",
                    f"{P} {dev}: share of the annotated steps with a valid aligned sample")
    for b in C.BLOCK_ORDER:
        dev = DEVICE_OF[b]
        reg.add(f"inv.n_participants.{dev}.raw", sum(bool(ctx.raw[u]["blocks"][b][1].any()) for u in ctx.users),
                "participants", "raw", f"participants whose {dev} block has a signal")
        reg.add(f"inv.n_participants.{dev}.v2", sum(ctx.stats(u)["sensors"][b]["shift_s"] is not None for u in ctx.users),
                "participants", "v2", f"participants whose {dev} block is kept after alignment")
    hb = [PNAME[u] for u in ctx.users if all(ctx.stats(u)["sensors"][b]["shift_s"] is not None for b in ("hand", "belt"))]
    reg.add("inv.hand_and_waist.v2", ", ".join(hb), "participants", "v2",
            "participants with both hand and waist sensors kept (patient configuration)")
    phone_only = [PNAME[u] for u in ctx.users if not any(ctx.stats(u)["sensors"][b]["shift_s"] is not None for b in ("hand", "belt", "leg"))]
    reg.add("inv.phone_only.v2", ", ".join(phone_only), "participants", "v2", "participants with the phone as only device")
    reg.add("inv.n_participants", len(ctx.users), "participants", "raw", "participants in the 2022 session")
    reg.add("inv.recording_min_s", min(rec), "s", "raw", "shortest recording")
    reg.add("inv.recording_max_s", max(rec), "s", "raw", "longest recording")
    reg.add("inv.recording_total_s", sum(rec), "s", "raw", "total recorded time, 8 participants")
    reg.add("inv.recording_total_min", sum(rec) / 60, "min", "raw", "total recorded time, 8 participants")
