"""Run the whole inspection and write reports/inspection/inspection_numbers.md and .json.

Usage (from code/):  .venv\\Scripts\\python.exe -m inspection_data.run_all
"""
from __future__ import annotations

import importlib
import time

from inspection_data.common import Context, Registry, write

STEPS = ["01_inventory", "02_labels", "03_preparation", "04_rules", "05_targets", "06_models"]


def main():
    t0 = time.time()
    ctx = Context()
    print(f"loaded raw + rebuilt v2 in {time.time() - t0:.0f} s; v2 on disk matches the rebuild")
    reg = Registry()
    for s in STEPS:
        importlib.import_module(f"inspection_data.{s}").run(ctx, reg)
        print(f"{s}: {len(reg.entries)} numbers so far")
    md, js = write(reg, ctx)
    print(f"written {md} and {js} ({len(reg.entries)} numbers, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
