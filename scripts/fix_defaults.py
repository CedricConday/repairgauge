"""Compare candidate defaults on the development phantoms (seeds 100 to 102).

The defaults in ``repairgauge.core.Config`` were chosen from this table and then
frozen. The reported backtests use other seeds (0 to 4) and, for real data,
subjects that were not looked at here.

    python scripts/fix_defaults.py --out results/defaults_dev --jobs 4
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from repairgauge.backtest import run_planted
from repairgauge.core import Config
from repairgauge.synth import make_phantom

SEEDS = (100, 101, 102)
NOISE = (0.02, 0.04, 0.08)
SHIFT = (0.0, 0.25, 0.5, 1.0)
DEFICIT = (0.15, 0.30)
CANDIDATES = (
    [("sigma", replace(Config(), reference_sigma_mm=s)) for s in (0.0, 10.0, 20.0, 40.0)]
    + [("tol", replace(Config(), align_tol_mm=t)) for t in (0.0, 0.15, 0.5)]
)


def _one(job) -> pd.DataFrame:
    seed, noise, shift, deficit = job
    phantom = make_phantom(seed, noise=noise, shift_vox=shift, deficit=deficit)
    parts = []
    for _, cfg in CANDIDATES:
        rows = run_planted(phantom, cfg)
        parts.append(rows.assign(seed=seed, noise=noise, shift_vox=shift,
                                 sigma=cfg.reference_sigma_mm, tol=cfg.align_tol_mm))
    return pd.concat(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("results/defaults_dev"))
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    grid = list(itertools.product(SEEDS, NOISE, SHIFT, DEFICIT))
    with Pool(args.jobs) as pool:
        rows = pd.concat(pool.map(_one, grid), ignore_index=True)
    rows.to_csv(args.out / "per_lesion.csv", index=False, float_format="%.5g")

    out = []
    for (sigma, tol), g in rows.groupby(["sigma", "tol"]):
        for z in (2.0, 2.5, 3.0):
            for name, sub in (("all", g), ("shift<=0.5", g[g.shift_vox <= 0.5])):
                zero, pos = sub[sub.recovery_true == 0], sub[sub.recovery_true > 0]
                quarter = sub[sub.recovery_true == 0.25]
                # judged depends on the floor but not on z, so the error columns repeat across z
                judged = sub[sub.judged.fillna(False).astype(bool)]
                out.append({
                    "sigma_mm": sigma, "align_tol_mm": tol, "z": z, "shifts": name,
                    "false_repaired_at_zero": (zero.z >= z).mean(),
                    "false_worsened_at_zero": (zero.z <= -z).mean(),
                    "sens_recovery_25": (quarter.z >= z).mean(),
                    "sens_any_recovery": (pos.z >= z).mean(),
                    "judged_share": len(judged) / len(sub),
                    "recovery_mae": (judged.recovery - judged.recovery_true).abs().mean(),
                })
    table = pd.DataFrame(out)
    table.to_csv(args.out / "candidates.csv", index=False, float_format="%.4g")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
