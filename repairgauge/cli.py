"""Command line: ``repairgauge run``, ``repairgauge backtest``, ``repairgauge phantom``."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import nibabel as nib
import numpy as np

from . import __version__
from .backtest import run_real, run_synthetic, write_report
from .core import Config, label_image, label_lesions, make_nawm, measure
from .synth import make_phantom


def _load(path: Path) -> nib.Nifti1Image:
    return nib.as_closest_canonical(nib.load(str(path)))


def _same_grid(ref: nib.Nifti1Image, other: nib.Nifti1Image, name: str) -> None:
    if ref.shape[:3] != other.shape[:3] or not np.allclose(ref.affine, other.affine, atol=1e-3):
        raise SystemExit(
            f"{name} is not on the baseline grid. RepairGauge does no registration: "
            "resample every input to the baseline map first."
        )


def _save(data: np.ndarray, like: nib.Nifti1Image, path: Path) -> None:
    nib.save(nib.Nifti1Image(data, like.affine), str(path))


def _config(args) -> Config:
    cfg = Config()
    for key in asdict(cfg):
        value = getattr(args, key, None)
        if value is not None:
            setattr(cfg, key, value)
    return cfg


def _add_config(ap: argparse.ArgumentParser) -> None:
    d = Config()
    g = ap.add_argument_group("thresholds (see docs/VALIDATION.md for how the defaults were fixed)")
    g.add_argument("--z", dest="z_threshold", type=float, help=f"|z| for a change call [{d.z_threshold}]")
    g.add_argument("--reference-sigma", dest="reference_sigma_mm", type=float,
                   help=f"mm; width of the local NAWM reference, 0 = global [{d.reference_sigma_mm}]")
    g.add_argument("--align-tol", dest="align_tol_mm", type=float,
                   help=f"mm; residual misalignment the noise floor absorbs [{d.align_tol_mm}]")
    g.add_argument("--margin", dest="nawm_margin_mm", type=float,
                   help=f"mm; lesions are grown by this before removal from NAWM [{d.nawm_margin_mm}]")
    g.add_argument("--n-null", dest="n_null", type=int, help=f"pseudo-lesions per lesion [{d.n_null}]")
    g.add_argument("--min-voxels", dest="min_voxels", type=int,
                   help=f"smaller lesions are flagged too_small [{d.min_voxels}]")
    g.add_argument("--wm-erode", dest="wm_erode_mm", type=float,
                   help=f"mm; rim removed from a --wm mask [{d.wm_erode_mm}]")
    g.add_argument("--min-deficit-ref", dest="min_deficit_ref", type=float,
                   help=f"ref_sd; shallower lesions are flagged [{d.min_deficit_ref}]")
    g.add_argument("--min-deficit-floors", dest="min_deficit_floors", type=float,
                   help=f"noise floors; shallower lesions are flagged [{d.min_deficit_floors}]")
    g.add_argument("--seed", type=int, help=f"seed of the pseudo-lesion sampling [{d.seed}]")


def _run(args) -> None:
    base, follow = _load(args.baseline), _load(args.followup)
    lesions = _load(args.lesions)
    _same_grid(base, follow, "follow-up")
    _same_grid(base, lesions, "lesion image")
    wm = _load(args.nawm or args.wm)
    _same_grid(base, wm, "white matter mask")
    cfg = _config(args)
    zooms = base.header.get_zooms()[:3]
    labels = label_lesions(np.asanyarray(lesions.dataobj))
    # a NAWM mask is taken as given; a white matter mask also loses its rim
    nawm = make_nawm(np.asanyarray(wm.dataobj), labels, zooms, cfg.nawm_margin_mm,
                     0.0 if args.nawm else cfg.wm_erode_mm)
    table, summary, labels = measure(base.get_fdata(dtype=np.float32),
                                     follow.get_fdata(dtype=np.float32), labels, nawm, zooms, cfg)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "lesions.csv", index=False, float_format="%.5g")
    summary = {"version": __version__, **summary}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _save(labels.astype(np.int32), base, args.out / "lesion_ids.nii.gz")
    _save(label_image(labels, table), base, args.out / "repair_labels.nii.gz")
    print(f"{summary['n_lesions']} lesions, {summary['n_judged']} judged: "
          f"{summary['n_repaired']} repaired, {summary['n_unchanged']} unchanged, "
          f"{summary['n_worsened']} worsened; volume-weighted recovery "
          f"{summary['volume_weighted_recovery']:.2f}; wrote {args.out}")


def _backtest(args) -> None:
    cfg = _config(args)
    if args.baseline:
        if not (args.followup and args.wm):
            raise SystemExit("planting into a real pair needs --baseline, --followup and --wm")
        base, follow, wm = _load(args.baseline), _load(args.followup), _load(args.wm)
        _same_grid(base, follow, "follow-up")
        _same_grid(base, wm, "white matter mask")
        rows = run_real(base.get_fdata(dtype=np.float32), follow.get_fdata(dtype=np.float32),
                        np.asanyarray(wm.dataobj) > 0, base.header.get_zooms()[:3], args.seeds,
                        args.deficit, cfg, follow_shift_vox=args.lesion_shift)
        meta = {"mode": "real pair", "baseline": args.baseline.name, "followup": args.followup.name,
                "seeds": args.seeds, "deficits": args.deficit, "lesion_shift_vox": args.lesion_shift}
    else:
        rows = run_synthetic(args.seeds, args.noise, args.shift, args.deficit, cfg, args.jobs)
        meta = {"mode": "phantom", "seeds": args.seeds, "noise": args.noise,
                "shift_vox": args.shift, "deficits": args.deficit}
    result = write_report(rows, args.out, cfg, meta)
    print(json.dumps({k: result[k] for k in ("planted", "nothing_planted") if k in result}, indent=2))
    print(f"wrote {args.out}")


def _phantom(args) -> None:
    p = make_phantom(args.seed, noise=args.noise, shift_vox=args.shift, deficit=args.deficit)
    args.out.mkdir(parents=True, exist_ok=True)
    like = nib.Nifti1Image(np.zeros((1, 1, 1), np.uint8), np.eye(4))
    _save(p.baseline, like, args.out / "baseline.nii.gz")
    _save(p.followup, like, args.out / "followup.nii.gz")
    _save(p.lesions, like, args.out / "lesions.nii.gz")
    _save(p.wm.astype(np.uint8), like, args.out / "wm.nii.gz")
    p.truth.to_csv(args.out / "truth.csv", index=False)
    print(f"phantom with {len(p.truth)} lesions in {args.out}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="repairgauge", description=__doc__)
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="measure every lesion of one person between two visits")
    run.add_argument("--baseline", type=Path, required=True, help="myelin-sensitive map, visit 1")
    run.add_argument("--followup", type=Path, required=True, help="same measure, visit 2, same grid")
    run.add_argument("--lesions", type=Path, required=True, help="baseline lesion labels or binary mask")
    grp = run.add_mutually_exclusive_group(required=True)
    grp.add_argument("--nawm", type=Path, help="normal-appearing white matter mask")
    grp.add_argument("--wm", type=Path, help="white matter mask; lesions plus a margin are removed")
    run.add_argument("--out", type=Path, required=True)
    _add_config(run)
    run.set_defaults(func=_run)

    bt = sub.add_parser("backtest", help="plant known recovery and score the tool against it")
    bt.add_argument("--out", type=Path, required=True)
    bt.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    bt.add_argument("--deficit", type=float, nargs="+", default=[0.15, 0.30],
                    help="planted deficit as a fraction of NAWM; 0 plants nothing")
    bt.add_argument("--noise", type=float, nargs="+", default=[0.02, 0.04, 0.08],
                    help="phantom only: voxel noise as a fraction of the white matter value")
    bt.add_argument("--shift", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0],
                    help="phantom only: shift of the follow-up in voxels")
    bt.add_argument("--jobs", type=int, default=1)
    bt.add_argument("--baseline", type=Path, help="real pair: plant into this scan ...")
    bt.add_argument("--followup", type=Path, help="... and this one, same person, same grid")
    bt.add_argument("--wm", type=Path, help="real pair: white matter mask")
    bt.add_argument("--lesion-shift", type=float, default=0.0,
                    help="real pair: move the planted lesions by this many voxels at follow-up")
    _add_config(bt)
    bt.set_defaults(func=_backtest)

    ph = sub.add_parser("phantom", help="write one phantom pair with its truth table")
    ph.add_argument("--out", type=Path, required=True)
    ph.add_argument("--seed", type=int, default=0)
    ph.add_argument("--noise", type=float, default=0.04)
    ph.add_argument("--shift", type=float, default=0.3)
    ph.add_argument("--deficit", type=float, nargs="+", default=[0.25])
    ph.set_defaults(func=_phantom)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
