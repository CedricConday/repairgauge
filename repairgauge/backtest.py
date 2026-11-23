"""Run the tool on planted lesions and score it against the truth."""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .core import REPAIRED, WORSENED, Config, make_nawm, measure
from .synth import RECOVERIES, SIZES_MM3, Planted, make_phantom, plant_pair


def run_planted(planted: Planted, cfg: Config) -> pd.DataFrame:
    """Measure a planted pair and join the result to the truth, one row per lesion."""
    nawm = make_nawm(planted.wm, planted.lesions, planted.zooms, cfg.nawm_margin_mm, cfg.wm_erode_mm)
    table, _, _ = measure(planted.baseline, planted.followup, planted.lesions, nawm,
                          planted.zooms, cfg)
    return planted.truth.merge(table.drop(columns=["n_voxels"]), on="lesion_id", how="left")


def _rate(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else float("nan")


def score(rows: pd.DataFrame, by=("size_mm3", "recovery_true")) -> pd.DataFrame:
    """Per cell: how many lesions, how many judged, recovery error, and label rates.

    ``called_repaired`` and ``called_worsened`` count every lesion that got a z-score,
    flagged or not. Where the truth is recovery > 0, ``called_repaired`` is the
    sensitivity; at recovery 0 it is the false-positive rate; at recovery <= 0,
    1 - ``called_repaired`` is the specificity. The ``judged_*`` columns give the same
    rates among unflagged lesions only.
    """
    out = []
    for key, g in rows.groupby(list(by), sort=True):
        key = key if isinstance(key, tuple) else (key,)
        j = g[g["judged"].fillna(False).astype(bool)]
        err = j["recovery"] - j["recovery_true"]
        out.append({
            **dict(zip(by, key, strict=True)),
            "n": len(g),
            "judged": _rate(g["judged"].fillna(False).astype(bool)),
            "recovery_mean": float(j["recovery"].mean()) if len(j) else float("nan"),
            "recovery_mae": float(err.abs().mean()) if len(j) else float("nan"),
            "called_repaired": _rate(g["label"] == REPAIRED),
            "called_worsened": _rate(g["label"] == WORSENED),
            "judged_repaired": _rate(j["label"] == REPAIRED),
            "judged_worsened": _rate(j["label"] == WORSENED),
        })
    return pd.DataFrame(out)


def headline(rows: pd.DataFrame) -> dict:
    """The few numbers the README quotes. All lesions unless a key says ``judged``."""
    judged = rows[rows["judged"].fillna(False).astype(bool)]
    zero, pos, neg = (rows[rows["recovery_true"] == 0], rows[rows["recovery_true"] > 0],
                      rows[rows["recovery_true"] < 0])
    full = rows[rows["recovery_true"] == 1.0]
    err = judged["recovery"] - judged["recovery_true"]
    jz = judged[judged["recovery_true"] == 0]
    return {
        "n_lesions": int(len(rows)),
        "judged_share": _rate(rows["judged"].fillna(False).astype(bool)),
        "recovery_mae_judged": float(err.abs().mean()) if len(judged) else float("nan"),
        "recovery_bias_judged": float(err.mean()) if len(judged) else float("nan"),
        "false_repaired_at_zero": _rate(zero["label"] == REPAIRED),
        "false_worsened_at_zero": _rate(zero["label"] == WORSENED),
        "false_repaired_at_zero_judged": _rate(jz["label"] == REPAIRED),
        "false_worsened_at_zero_judged": _rate(jz["label"] == WORSENED),
        "sensitivity_repaired_any_recovery": _rate(pos["label"] == REPAIRED),
        "sensitivity_repaired_full_recovery": _rate(full["label"] == REPAIRED),
        "specificity_repaired": 1.0 - _rate(pd.concat([zero, neg])["label"] == REPAIRED),
        "sensitivity_worsened": _rate(neg["label"] == WORSENED),
    }


def _one_phantom(job) -> pd.DataFrame:
    seed, noise, shift, deficit, cfg = job
    rows = run_planted(make_phantom(seed, noise=noise, shift_vox=shift, deficit=deficit), cfg)
    return rows.assign(seed=seed, noise=noise, shift_vox=shift)


def run_synthetic(seeds, noises, shifts, deficits, cfg: Config | None = None,
                  jobs: int = 1) -> pd.DataFrame:
    """One phantom per seed, noise level, shift and deficit; every lesion scored."""
    cfg = cfg or Config()
    grid = [(s, n, sh, d, cfg) for s, n, sh, d in itertools.product(seeds, noises, shifts, deficits)]
    if jobs > 1:
        with Pool(jobs) as pool:
            parts = pool.map(_one_phantom, grid)
    else:
        parts = [_one_phantom(j) for j in grid]
    return pd.concat(parts, ignore_index=True)


def run_real(baseline: np.ndarray, followup: np.ndarray, wm: np.ndarray, zooms, seeds,
             deficits, cfg: Config | None = None, follow_shift_vox: float = 0.0,
             sizes_mm3=SIZES_MM3, recoveries=RECOVERIES) -> pd.DataFrame:
    """Plant into one real pair, once per seed and deficit. ``deficit`` 0 plants nothing."""
    cfg = cfg or Config()
    parts = []
    for seed, deficit in itertools.product(seeds, deficits):
        planted = plant_pair(baseline, followup, wm, zooms, seed, deficit=deficit,
                             sizes_mm3=sizes_mm3, recoveries=recoveries,
                             follow_shift_vox=follow_shift_vox)
        parts.append(run_planted(planted, cfg).assign(seed=seed))
    return pd.concat(parts, ignore_index=True)


def write_report(rows: pd.DataFrame, out: Path, cfg: Config, meta: dict) -> dict:
    """Write per_lesion.csv, the score tables and headline.json into ``out``."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out / "per_lesion.csv", index=False, float_format="%.5g")
    planted = rows[rows["deficit_true"] > 0]
    result = {"version": __version__, **meta, "config": asdict(cfg)}
    if len(planted):
        score(planted).to_csv(out / "by_size_recovery.csv", index=False, float_format="%.4g")
        score(planted, by=("deficit_true", "recovery_true")).to_csv(
            out / "by_deficit_recovery.csv", index=False, float_format="%.4g")
        for extra in ("noise", "shift_vox"):
            if extra in rows:
                score(planted, by=(extra, "size_mm3")).to_csv(
                    out / f"by_{extra}_size.csv", index=False, float_format="%.4g")
                score(planted, by=(extra, "recovery_true")).to_csv(
                    out / f"by_{extra}_recovery.csv", index=False, float_format="%.4g")
        result["planted"] = headline(planted)
    blank = rows[rows["deficit_true"] == 0]
    if len(blank):
        called = blank["label"].isin([REPAIRED, WORSENED])
        result["nothing_planted"] = {
            "n_lesions": int(len(blank)),
            "called_repaired": _rate(blank["label"] == REPAIRED),
            "called_worsened": _rate(blank["label"] == WORSENED),
            "called_any_change": _rate(called),
            "flagged_share": 1.0 - _rate(blank["judged"].fillna(False).astype(bool)),
        }
        by_size = blank.assign(called=called).groupby("size_mm3")["called"].agg(["size", "mean"])
        by_size.rename(columns={"size": "n", "mean": "called_any_change"}).to_csv(
            out / "nothing_planted_by_size.csv", float_format="%.4g")
    (out / "headline.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
