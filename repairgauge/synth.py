"""Known answers: a phantom with planted lesions, and planting into real scan pairs.

No open dataset gives a myelin value per lesion at two visits with the true amount
of repair. So the answer is planted. A lesion is a soft-edged ellipsoid where the
map is lowered by a known fraction ``deficit`` at baseline; at follow-up a known
fraction ``recovery`` of that deficit is given back (negative = worsening). The
follow-up is then degraded the way a second visit is: its own noise, a global scale
change, a smooth bias field and a sub-voxel shift.

:func:`make_phantom` builds everything from nothing. :func:`plant_pair` puts the same
lesions into two real scans of one person, so the noise is real.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial.transform import Rotation

SIZES_MM3 = (15, 50, 150, 500, 1500)
RECOVERIES = (0.0, 0.25, 0.5, 0.75, 1.0, -0.5)


@dataclass
class Planted:
    baseline: np.ndarray
    followup: np.ndarray
    lesions: np.ndarray  # int32 labels, the "segmentation" at baseline
    wm: np.ndarray  # bool
    truth: pd.DataFrame  # lesion_id, size_mm3, n_voxels, deficit_true, recovery_true
    zooms: tuple


def _smooth_field(shape, sigma, rng) -> np.ndarray:
    """Gaussian random field with zero mean and unit standard deviation."""
    field = ndi.gaussian_filter(rng.standard_normal(shape).astype(np.float32), sigma, mode="reflect")
    return (field - field.mean()) / field.std()


def _ellipsoid(volume_vox: float, rng, edge_sigma: float = 0.6) -> np.ndarray:
    """Soft-edged ellipsoid profile in [0, 1] on a small grid; ``profile >= 0.5`` is the mask."""
    r = (3.0 * volume_vox / (4.0 * np.pi)) ** (1.0 / 3.0)
    a, b = rng.uniform(0.75, 1.33, size=2)
    radii = r * np.array([a, b, 1.0 / (a * b)])
    half = int(np.ceil(radii.max() + 3 * edge_sigma)) + 1
    grid = np.stack(np.meshgrid(*[np.arange(-half, half + 1, dtype=np.float32)] * 3, indexing="ij"), -1)
    rot = Rotation.random(random_state=rng).as_matrix().astype(np.float32)
    hard = ((((grid @ rot) / radii) ** 2).sum(-1) <= 1.0).astype(np.float32)
    return ndi.gaussian_filter(hard, edge_sigma)


def place_lesions(wm: np.ndarray, sizes_vox, rng, gap_vox: float = 5.0, max_tries: int = 400):
    """Non-overlapping ellipsoids inside ``wm``. Returns profiles, their slices and a label image.

    Lesions are placed largest first and kept ``gap_vox`` apart so that normal-appearing
    white matter remains between them. A lesion that finds no room is skipped.
    """
    depth = ndi.distance_transform_edt(wm)
    occupied = np.zeros(wm.shape, bool)
    labels = np.zeros(wm.shape, np.int32)
    placed = []
    order = np.argsort(sizes_vox)[::-1]
    for i in order:
        profile = _ellipsoid(sizes_vox[i], rng)
        half = profile.shape[0] // 2
        reach = np.argwhere(profile >= 0.5)
        radius = float(np.abs(reach - half).max()) + 1.0
        candidates = np.argwhere(depth > radius + 1.0)
        if len(candidates) == 0:
            continue
        for _ in range(max_tries):
            c = candidates[rng.integers(len(candidates))]
            if (c - half < 0).any() or (c + half + 1 > np.array(wm.shape)).any():
                continue
            sl = tuple(slice(int(v - half), int(v + half + 1)) for v in c)
            mask = profile >= 0.5
            if (occupied[sl] & mask).any() or not wm[sl][mask].all():
                continue
            placed.append((int(i), sl, profile))
            labels[sl][mask] = len(placed)
            near = ndi.distance_transform_edt(~mask) <= gap_vox
            occupied[sl] |= near
            break
    return placed, labels


def _plant(base: np.ndarray, follow: np.ndarray, wm: np.ndarray, zooms, rng, sizes_mm3,
           recoveries, deficit, follow_shift_vox: float = 0.0) -> tuple[np.ndarray, pd.DataFrame]:
    """Lower ``base`` and ``follow`` in place inside planted lesions; return labels and truth."""
    voxel = float(np.prod(zooms))
    deficits = np.atleast_1d(np.asarray(deficit, dtype=float))
    cells = [(s, r, d) for s in sizes_mm3 for r in recoveries for d in deficits]
    sizes_vox = np.array([c[0] / voxel for c in cells])
    placed, labels = place_lesions(wm, sizes_vox, rng)
    direction = rng.standard_normal(3)
    direction *= follow_shift_vox / np.linalg.norm(direction)
    rows = []
    for lesion_id, (i, sl, profile) in enumerate(placed, start=1):
        size, recovery, d = cells[i]
        base[sl] *= 1.0 - d * profile
        moved = ndi.shift(profile, direction, order=1) if follow_shift_vox > 0 else profile
        follow[sl] *= 1.0 - d * (1.0 - recovery) * moved
        rows.append({"lesion_id": lesion_id, "size_mm3": size,
                     "n_voxels": int((labels[sl] == lesion_id).sum()),
                     "deficit_true": d, "recovery_true": recovery})
    return labels, pd.DataFrame(rows)


def make_phantom(seed: int = 0, shape=(112, 112, 80), *, noise: float = 0.04,
                 shift_vox: float = 0.3, scale: float = 1.05, bias: float = 0.03,
                 deficit=0.25, sizes_mm3=SIZES_MM3, recoveries=RECOVERIES,
                 followup_noise_factor: float = 1.0, wm_value: float = 40.0) -> Planted:
    """A two-visit phantom at 1 mm with one lesion per size and recovery level.

    ``noise`` is the voxel noise standard deviation as a fraction of the white matter
    value, drawn independently for each visit. ``scale`` multiplies the whole follow-up,
    ``bias`` is the peak amplitude of a smooth multiplicative field on the follow-up and
    ``shift_vox`` the length of a rigid shift of the follow-up in a random direction.
    """
    rng = np.random.default_rng(seed)
    grid = np.stack(np.meshgrid(*[np.linspace(-1, 1, n, dtype=np.float32) for n in shape],
                                indexing="ij"))
    rr = np.sqrt((grid**2).sum(axis=0))
    brain, wm_true = rr <= 0.92, rr <= 0.76
    # white matter is not flat: a slow regional trend and a finer texture, both anatomy,
    # so both are the same at the two visits
    texture = 1.0 + 0.03 * _smooth_field(shape, 6.0, rng) + 0.02 * _smooth_field(shape, 1.5, rng)
    tissue = np.where(wm_true, 1.0, np.where(brain, 0.8, 0.0)).astype(np.float32)
    tissue = ndi.gaussian_filter(tissue, 0.6) * texture * wm_value

    wm = ndi.binary_erosion(wm_true, iterations=1)
    base, follow = tissue.copy(), tissue.copy()
    labels, truth = _plant(base, follow, wm, (1.0, 1.0, 1.0), rng, sizes_mm3, recoveries, deficit)

    direction = rng.standard_normal(3)
    direction *= shift_vox / np.linalg.norm(direction)
    if shift_vox > 0:
        follow = ndi.shift(follow, direction, order=3, mode="nearest")
    field = _smooth_field(shape, 25.0, rng)
    follow = follow * scale * (1.0 + bias * field / np.abs(field).max())
    sd = noise * wm_value
    base = base + sd * rng.standard_normal(shape).astype(np.float32)
    follow = follow + sd * followup_noise_factor * rng.standard_normal(shape).astype(np.float32)
    return Planted(base.astype(np.float32), follow.astype(np.float32), labels, wm, truth,
                   (1.0, 1.0, 1.0))


def plant_pair(baseline: np.ndarray, followup: np.ndarray, wm: np.ndarray, zooms, seed: int = 0, *,
               deficit=0.25, sizes_mm3=SIZES_MM3, recoveries=RECOVERIES,
               follow_shift_vox: float = 0.0, erode_vox: int = 1) -> Planted:
    """Plant lesions with known deficit and recovery into two real scans of one person.

    The scans keep their own noise, drift and residual misalignment. ``deficit=0``
    plants nothing and only marks the lesion shapes: every label other than
    "unchanged" is then a false positive. ``follow_shift_vox`` moves the planted
    lesion at follow-up, to mimic a lesion that the registration left misaligned.
    """
    rng = np.random.default_rng(seed)
    base = np.array(baseline, dtype=np.float32)
    follow = np.array(followup, dtype=np.float32)
    wm = np.asarray(wm) > 0
    core = ndi.binary_erosion(wm, iterations=erode_vox) if erode_vox > 0 else wm
    labels, truth = _plant(base, follow, core, zooms, rng, sizes_mm3, recoveries, deficit,
                           follow_shift_vox)
    return Planted(base, follow, labels, wm, truth, tuple(float(z) for z in zooms))
