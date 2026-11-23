"""Per-lesion change of a myelin-sensitive value between two aligned scans.

The measurement, in order:

1. **Reference.** At each time point the map is divided by a local normal-appearing
   white matter (NAWM) centre: a Gaussian-weighted mean of NAWM values around every
   voxel. A global scale change and a smooth multiplicative bias between visits cancel
   in that ratio. The ratio is then expressed in units of the baseline NAWM spread
   (robust coefficient of variation), so ``n = (map / centre - 1) / cv``.
2. **Deficit, change, recovery.** ``deficit`` is how far the lesion mean sits below
   the reference at baseline, ``change`` is follow-up minus baseline, ``recovery`` is
   ``change / deficit``.
3. **Noise floor.** The lesion's own shape is dropped at many random places in NAWM
   and the same change is computed there. The spread of those values is what "no
   change" looks like for a lesion of this size and shape on this pair of scans.
   A second term adds what a small residual misalignment would do to this lesion.
   The same pseudo-lesions also show how far a patch of healthy white matter of this
   shape sits from the reference at baseline (``ref_sd``). The deficit is only known
   to that precision, because the lesion's value before it formed is not known.
4. **Label.** ``z = (change - null mean) / floor``; repaired, unchanged or worsened at
   ``z_threshold``. A change is called only if it also lies outside the matching
   quantiles of the null itself, because real scan differences have heavier tails
   than a normal curve. Lesions that are too small or too shallow are flagged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import norm

REPAIRED, UNCHANGED, WORSENED, UNDETERMINED = "repaired", "unchanged", "worsened", "undetermined"
LABEL_CODES = {REPAIRED: 1, UNCHANGED: 2, WORSENED: 3, UNDETERMINED: 4}
COLUMNS = ["lesion_id", "n_voxels", "volume_mm3", "x", "y", "z_vox", "baseline", "followup",
           "deficit", "deficit_pct", "change", "recovery", "recovery_se", "null_mean", "noise_sd",
           "align_sd", "floor_sd", "ref_sd", "null_lo", "null_hi", "n_null", "z", "label", "flag",
           "judged"]


@dataclass
class Config:
    """Every threshold in one place. See docs/VALIDATION.md for how the defaults were fixed."""

    reference_sigma_mm: float = 20.0  # width of the local NAWM reference; 0 = one global value
    nawm_margin_mm: float = 2.0  # lesions are grown by this much before removal from NAWM
    wm_erode_mm: float = 1.0  # a white matter mask loses this rim (partial volume) to make NAWM
    n_null: int = 1000  # pseudo-lesions per lesion
    min_null: int = 100  # fewer valid placements than this: no z-score
    null_min_coverage: float = 0.9  # share of a pseudo-lesion that must fall in NAWM
    null_max_voxels: int = 4000  # larger lesions are subsampled for the null only
    align_tol_mm: float = 0.3  # residual misalignment the floor should absorb; 0 = off
    z_threshold: float = 2.5  # |z| at or above this is called a change
    min_voxels: int = 10  # smaller lesions are flagged too_small
    min_deficit_ref: float = 3.0  # deficit must be at least this many ref_sd ...
    min_deficit_floors: float = 4.0  # ... and this many noise floors (floor / deficit <= 0.25)
    seed: int = 0


@dataclass
class Reference:
    centre_baseline: float  # global NAWM median, baseline
    centre_followup: float
    cv_baseline: float  # robust NAWM spread / centre, baseline: the unit of every value
    cv_followup: float  # reported only; a large gap to baseline means the noise changed
    n_nawm: int


def label_lesions(lesions: np.ndarray) -> np.ndarray:
    """Integer label image. A binary mask is split into 26-connected components."""
    lesions = np.asarray(lesions)
    ids = np.unique(lesions[lesions != 0])
    if len(ids) > 1:
        return np.rint(lesions).astype(np.int32)
    labels, _ = ndi.label(lesions != 0, structure=np.ones((3, 3, 3), bool))
    return labels.astype(np.int32)


def make_nawm(wm: np.ndarray, labels: np.ndarray, zooms, margin_mm: float,
              erode_mm: float = 0.0) -> np.ndarray:
    """White matter with every lesion, grown by ``margin_mm``, removed.

    ``erode_mm`` also removes the rim of the white matter mask, where voxels are part
    grey matter or fluid and would pull the reference down.
    """
    wm = np.asarray(wm) > 0
    lesion = np.asarray(labels) > 0
    if erode_mm > 0:
        wm = ndi.distance_transform_edt(wm | lesion, sampling=zooms) > erode_mm
    if margin_mm > 0 and lesion.any():
        lesion = ndi.distance_transform_edt(~lesion, sampling=zooms) <= margin_mm
    return wm & ~lesion


def _robust(values: np.ndarray) -> tuple[float, float]:
    centre = float(np.median(values))
    return centre, float(1.4826 * np.median(np.abs(values - centre)))


def local_ratio(img: np.ndarray, nawm: np.ndarray, zooms, sigma_mm: float) -> tuple[np.ndarray, float]:
    """``img`` divided by its local NAWM centre, and the global NAWM centre."""
    centre, spread = _robust(img[nawm])
    if not centre > 0:
        raise ValueError(
            f"NAWM median is {centre:.4g}; the map must be positive in white matter "
            "(values are compared as ratios to NAWM)"
        )
    if sigma_mm <= 0:
        return (img / centre).astype(np.float32), centre
    sig = [sigma_mm / z for z in zooms]
    weight = nawm.astype(np.float32)
    # clipping keeps a stray vessel or CSF voxel inside the mask from pulling the mean
    vals = np.clip(img, centre - 4 * spread, centre + 4 * spread).astype(np.float32) * weight
    num = ndi.gaussian_filter(vals, sig, mode="constant", truncate=3.0)
    den = ndi.gaussian_filter(weight, sig, mode="constant", truncate=3.0)
    local = np.where(den > 0.01, num / np.maximum(den, 1e-6), centre)
    return (img / local).astype(np.float32), centre


def _null(delta, n_base, nawm, nawm_idx, offsets, rng, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Mean change and mean baseline value over the lesion's shape at random NAWM positions.

    The shape is translated and mirrored, never resized. A placement counts when at
    least ``null_min_coverage`` of it lies in NAWM; only those voxels are averaged.
    """
    if len(offsets) > cfg.null_max_voxels:
        offsets = offsets[rng.choice(len(offsets), cfg.null_max_voxels, replace=False)]
    shape = np.array(delta.shape)
    batch = int(max(16, min(cfg.n_null, 1_000_000 // len(offsets))))
    changes: list[np.ndarray] = []
    bases: list[np.ndarray] = []
    found = tries = 0
    while found < cfg.n_null and tries < 20 * cfg.n_null:
        centres = nawm_idx[rng.integers(0, len(nawm_idx), batch)]
        flips = rng.choice(np.array([-1, 1]), size=(batch, 1, 3))
        coords = centres[:, None, :] + offsets[None] * flips
        inside = ((coords >= 0) & (coords < shape)).all(axis=2)
        c = np.clip(coords, 0, shape - 1)
        idx = (c[..., 0], c[..., 1], c[..., 2])
        hit = nawm[idx] & inside
        n_hit = hit.sum(axis=1)
        ok = n_hit >= cfg.null_min_coverage * len(offsets)
        changes.append(((delta[idx] * hit).sum(axis=1) / np.maximum(n_hit, 1))[ok])
        bases.append(((n_base[idx] * hit).sum(axis=1) / np.maximum(n_hit, 1))[ok])
        found += int(ok.sum())
        tries += batch
    return np.concatenate(changes)[: cfg.n_null], np.concatenate(bases)[: cfg.n_null]


def _align_sd(n_base: np.ndarray, mask: np.ndarray, zooms, tol_mm: float) -> float:
    """RMS change of the lesion mean when the baseline is shifted ``tol_mm`` along each axis.

    A lesion is darker than its surroundings, so any shift between the visits raises
    the mean inside a fixed mask. Small lesions with deep deficits are hit hardest.
    """
    if tol_mm <= 0:
        return 0.0
    base = n_base[mask].mean()
    diffs = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            shift = [0.0, 0.0, 0.0]
            shift[axis] = sign * tol_mm / zooms[axis]
            moved = ndi.shift(n_base, shift, order=1, mode="nearest")
            diffs.append(moved[mask].mean() - base)
    return float(np.sqrt(np.mean(np.square(diffs))))


def _crop(mask: np.ndarray, pad: int = 4) -> tuple[slice, ...]:
    """Bounding box of ``mask`` with a margin. Everything outside it is never read."""
    box = ndi.find_objects(mask.astype(np.uint8))[0]
    return tuple(slice(max(s.start - pad, 0), min(s.stop + pad, n)) for s, n in zip(box, mask.shape, strict=True))


def measure(baseline, followup, lesions, nawm, zooms=(1.0, 1.0, 1.0),
            cfg: Config | None = None) -> tuple[pd.DataFrame, dict, np.ndarray]:
    """Measure every lesion. Returns the per-lesion table, the summary and the label image.

    ``baseline`` and ``followup`` are aligned arrays of one myelin-sensitive measure,
    ``lesions`` a label image or binary mask drawn at baseline, ``nawm`` a boolean
    NAWM mask (see :func:`make_nawm`). The returned image carries the lesion ids.
    """
    cfg = cfg or Config()
    zooms = tuple(float(z) for z in zooms)
    if not (np.shape(baseline) == np.shape(followup) == np.shape(lesions) == np.shape(nawm)):
        raise ValueError("baseline, follow-up, lesions and NAWM must share one grid")
    full_labels = label_lesions(lesions)
    nawm = np.asarray(nawm, bool) & (full_labels == 0)
    if nawm.sum() < 1000:
        raise ValueError(f"NAWM mask has {int(nawm.sum())} voxels; need at least 1000")
    box = _crop(nawm | (full_labels > 0))
    origin = np.array([s.start for s in box])
    baseline = np.asarray(baseline, dtype=np.float32)[box]
    followup = np.asarray(followup, dtype=np.float32)[box]
    labels = full_labels[box]
    finite = np.isfinite(baseline) & np.isfinite(followup)
    baseline = np.where(finite, baseline, 0.0)
    followup = np.where(finite, followup, 0.0)
    nawm = nawm[box] & finite

    r1, c1 = local_ratio(baseline, nawm, zooms, cfg.reference_sigma_mm)
    r2, c2 = local_ratio(followup, nawm, zooms, cfg.reference_sigma_mm)
    cv1 = _robust(r1[nawm])[1]
    cv2 = _robust(r2[nawm])[1]
    if not cv1 > 0:
        raise ValueError("NAWM has zero spread at baseline; cannot set a unit")
    # One unit for both visits. Dividing each visit by its own spread would turn a
    # change in image noise between visits into an apparent change in every lesion.
    n1 = (r1 - 1.0) / cv1
    n2 = (r2 - 1.0) / cv1
    delta = n2 - n1
    ref = Reference(c1, c2, cv1, cv2, int(nawm.sum()))

    rng = np.random.default_rng(cfg.seed)
    nawm_idx = np.argwhere(nawm)
    voxel_mm3 = float(np.prod(zooms))
    tail = float(norm.sf(cfg.z_threshold))
    pad = 3
    rows = []
    for lesion_id, sl in enumerate(ndi.find_objects(labels), start=1):
        if sl is None:
            continue
        big = tuple(slice(max(s.start - pad, 0), min(s.stop + pad, n)) for s, n in zip(sl, labels.shape, strict=True))
        mask = (labels[big] == lesion_id) & finite[big]
        n_vox = int(mask.sum())
        if n_vox == 0:
            continue
        vox = np.argwhere(mask)
        centroid = vox.mean(axis=0)
        offsets = vox - np.rint(centroid).astype(int)
        base = float(n1[big][mask].mean())
        follow = float(n2[big][mask].mean())
        deficit, change = -base, follow - base
        null, null_base = _null(delta, n1, nawm, nawm_idx, offsets, rng, cfg)
        align_sd = _align_sd(n1[big], mask, zooms, cfg.align_tol_mm)
        row = {
            "lesion_id": lesion_id, "n_voxels": n_vox, "volume_mm3": n_vox * voxel_mm3,
            "x": centroid[0] + big[0].start + origin[0],
            "y": centroid[1] + big[1].start + origin[1],
            "z_vox": centroid[2] + big[2].start + origin[2],
            "baseline": base, "followup": follow, "deficit": deficit,
            "deficit_pct": 100.0 * deficit * cv1, "change": change,
            "align_sd": align_sd, "n_null": len(null),
        }
        if len(null) < cfg.min_null:
            rows.append({**row, "label": UNDETERMINED, "flag": "no_null", "judged": False})
            continue
        null_mean, noise_sd = float(null.mean()), float(null.std(ddof=1))
        ref_sd = float(null_base.std(ddof=1))
        lo, hi = np.quantile(null, [tail, 1.0 - tail])
        floor = float(np.hypot(noise_sd, align_sd))
        z = (change - null_mean) / floor
        if z >= cfg.z_threshold and change > hi:
            label = REPAIRED
        elif z <= -cfg.z_threshold and change < lo:
            label = WORSENED
        else:
            label = UNCHANGED
        if n_vox < cfg.min_voxels:
            flag = "too_small"
        elif deficit < max(cfg.min_deficit_ref * ref_sd, cfg.min_deficit_floors * floor):
            flag = "shallow_deficit"
        else:
            flag = ""
        judged = flag == ""
        recovery = change / deficit if judged else float("nan")
        # two sources: noise in the change, and not knowing the healthy value (the divisor)
        se = float(np.hypot(floor, recovery * ref_sd) / deficit) if judged else float("nan")
        rows.append({**row, "recovery": recovery, "recovery_se": se, "null_mean": null_mean,
                     "noise_sd": noise_sd, "floor_sd": floor, "ref_sd": ref_sd,
                     "null_lo": float(lo), "null_hi": float(hi), "z": z, "label": label,
                     "flag": flag, "judged": judged})
    table = pd.DataFrame(rows, columns=COLUMNS)
    table["judged"] = table["judged"].astype(bool)
    return table, summarise(table, ref, cfg), full_labels


def summarise(table: pd.DataFrame, ref: Reference, cfg: Config) -> dict:
    """Per-person numbers. Counts and shares use judged lesions only."""
    judged = table[table["judged"]]
    vol = judged["volume_mm3"].to_numpy(dtype=float)
    total = float(vol.sum())

    def share(label: str) -> float:
        return float(vol[(judged["label"] == label).to_numpy()].sum() / total) if total > 0 else float("nan")

    return {
        "n_lesions": int(len(table)),
        "n_judged": int(len(judged)),
        "n_flagged": {k: int(v) for k, v in table.loc[~table["judged"], "flag"].value_counts().items()},
        "n_repaired": int((judged["label"] == REPAIRED).sum()),
        "n_unchanged": int((judged["label"] == UNCHANGED).sum()),
        "n_worsened": int((judged["label"] == WORSENED).sum()),
        "lesion_volume_mm3": float(table["volume_mm3"].sum()),
        "judged_volume_mm3": total,
        "volume_weighted_recovery": (
            float(np.average(judged["recovery"], weights=vol)) if total > 0 else float("nan")
        ),
        "share_judged_volume_repaired": share(REPAIRED),
        "share_judged_volume_worsened": share(WORSENED),
        "reference": asdict(ref),
        "config": asdict(cfg),
    }


def label_image(labels: np.ndarray, table: pd.DataFrame) -> np.ndarray:
    """1 repaired, 2 unchanged, 3 worsened, 4 flagged or undetermined, 0 elsewhere."""
    lut = np.zeros(int(labels.max()) + 1, dtype=np.uint8)
    for row in table.itertuples():
        lut[row.lesion_id] = LABEL_CODES[row.label] if row.judged else LABEL_CODES[UNDETERMINED]
    return lut[labels]
