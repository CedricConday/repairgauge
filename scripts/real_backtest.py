"""Plant known recovery into same-day test-retest pairs of healthy adults and score it.

Reads the layout written by ``scripts/fetch_mirror.py``. For every subject and map
type, session 1 is the baseline and session 2 the follow-up. Nothing changed in
these people between the two scans, so every call on an unplanted region is false.

    python scripts/real_backtest.py --data work/mirror --out results/real_mirror \
        --subjects sub-60476 sub-60501 --maps MTRmap ihMTR --jobs 4
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from glob import glob
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from repairgauge.backtest import run_real, write_report
from repairgauge.core import Config


def _one(job) -> pd.DataFrame:
    data, sub, kind, seeds, deficits, cfg, shift = job
    maps = [glob(f"{data}/derivatives/ihmt/{sub}/{ses}/anat/*_space-T1w_{kind}.nii.gz")[0]
            for ses in ("ses-01", "ses-02")]
    wm = nib.load(glob(f"{data}/derivatives/pymp2rage/{sub}/anat/*_space-T1w_desc-wm_mask.nii.gz")[0])
    base, follow = nib.load(maps[0]), nib.load(maps[1])
    if base.shape != follow.shape or not np.allclose(base.affine, follow.affine, atol=1e-3):
        raise RuntimeError(f"{sub} {kind}: the two sessions are not on one grid")
    rows = run_real(base.get_fdata(dtype=np.float32), follow.get_fdata(dtype=np.float32),
                    np.asanyarray(wm.dataobj) > 0, base.header.get_zooms()[:3], seeds, deficits,
                    cfg, follow_shift_vox=shift)
    return rows.assign(subject=sub, map=kind)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=Path("work/mirror"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--subjects", nargs="+", required=True)
    ap.add_argument("--maps", nargs="+", default=["MTRmap", "ihMTR"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--deficit", type=float, nargs="+", default=[0.0, 0.15, 0.30])
    ap.add_argument("--lesion-shift", type=float, default=0.0)
    ap.add_argument("--reference-sigma", type=float)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    cfg = Config()
    if args.reference_sigma is not None:
        cfg = replace(cfg, reference_sigma_mm=args.reference_sigma)
    jobs = [(args.data, s, m, args.seeds, args.deficit, cfg, args.lesion_shift)
            for s in args.subjects for m in args.maps]
    with Pool(args.jobs) as pool:
        rows = pd.concat(pool.map(_one, jobs), ignore_index=True)
    meta = {"mode": "real pairs, OpenNeuro ds008825", "subjects": args.subjects,
            "seeds": args.seeds, "deficits": args.deficit, "lesion_shift_vox": args.lesion_shift}
    for kind in args.maps:
        result = write_report(rows[rows["map"] == kind], args.out / kind, cfg, {**meta, "map": kind})
        print(kind, {k: result[k] for k in ("planted", "nothing_planted") if k in result})


if __name__ == "__main__":
    main()
