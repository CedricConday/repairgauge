import json

import nibabel as nib
import numpy as np
import pytest

from repairgauge.cli import main
from repairgauge.core import Config, label_image, label_lesions, make_nawm, measure


def _scene(seed=0, shape=(48, 48, 40), noise=0.5):
    """Flat white matter at 40 with two cubic lesions at 30: one stays, one returns to 40."""
    rng = np.random.default_rng(seed)
    wm = np.zeros(shape, bool)
    wm[4:-4, 4:-4, 4:-4] = True
    lesions = np.zeros(shape, np.int32)
    lesions[10:16, 10:16, 10:16] = 1
    lesions[30:36, 30:36, 20:26] = 2
    base = np.full(shape, 40.0, np.float32)
    base[lesions > 0] = 30.0
    follow = base.copy()
    follow[lesions == 2] = 40.0
    base += noise * rng.standard_normal(shape).astype(np.float32)
    follow += noise * rng.standard_normal(shape).astype(np.float32)
    return base, follow, lesions, wm


def _measure(base, follow, lesions, wm, **kw):
    cfg = Config(**kw)
    nawm = make_nawm(wm, lesions, (1, 1, 1), cfg.nawm_margin_mm, cfg.wm_erode_mm)
    return measure(base, follow, lesions, nawm, (1, 1, 1), cfg)


def test_known_answer_two_lesions():
    table, summary, labels = _measure(*_scene())
    stays, returns = table.iloc[0], table.iloc[1]
    assert stays.label == "unchanged" and returns.label == "repaired"
    assert abs(stays.recovery) < 0.05 and abs(returns.recovery - 1.0) < 0.05
    assert abs(stays.deficit_pct - 25.0) < 1.0
    assert summary["n_repaired"] == 1 and summary["n_unchanged"] == 1
    assert abs(summary["share_judged_volume_repaired"] - 0.5) < 1e-6
    assert abs(summary["volume_weighted_recovery"] - 0.5) < 0.05
    assert set(np.unique(label_image(labels, table))) == {0, 1, 2}


def test_scale_and_bias_cancel():
    base, follow, lesions, wm = _scene()
    ramp = np.linspace(0.95, 1.05, base.shape[0], dtype=np.float32)[:, None, None]
    table, _, _ = _measure(base, follow * 1.3 * ramp, lesions, wm)
    assert abs(table.recovery[0]) < 0.08 and abs(table.recovery[1] - 1.0) < 0.08
    assert list(table.label) == ["unchanged", "repaired"]


def test_noisier_followup_is_not_a_change():
    base, follow, lesions, wm = _scene()
    rng = np.random.default_rng(1)
    follow = follow + 1.5 * rng.standard_normal(follow.shape).astype(np.float32)
    table, _, _ = _measure(base, follow, lesions, wm)
    assert abs(table.recovery[0]) < 0.1
    assert table.label[0] == "unchanged"


def test_worsening_is_called():
    base, follow, lesions, wm = _scene()
    follow[lesions == 1] -= 5.0
    table, _, _ = _measure(base, follow, lesions, wm)
    assert table.label[0] == "worsened"
    assert abs(table.recovery[0] + 0.5) < 0.05


def test_small_lesion_needs_a_bigger_change():
    base, follow, lesions, wm = _scene()
    lesions[20:22, 20:22, 30:32] = 3
    base[lesions == 3] -= 10.0
    follow[lesions == 3] -= 10.0
    table, _, _ = _measure(base, follow, lesions, wm)
    assert table.noise_sd[2] > 2 * table.noise_sd[0]
    assert table.flag[2] == "too_small" and not table.judged[2]
    assert np.isnan(table.recovery[2])


def test_shallow_deficit_is_flagged():
    base, follow, lesions, wm = _scene()
    base[lesions == 1] += 9.9
    follow[lesions == 1] += 9.9
    table, summary, _ = _measure(base, follow, lesions, wm)
    assert table.flag[0] == "shallow_deficit" and np.isnan(table.recovery[0])
    assert summary["n_judged"] == 1 and summary["n_flagged"] == {"shallow_deficit": 1}


def test_seeded_and_binary_mask_is_split():
    base, follow, lesions, wm = _scene()
    a, _, _ = _measure(base, follow, lesions > 0, wm)
    b, _, _ = _measure(base, follow, lesions > 0, wm)
    assert len(a) == 2 and a.equals(b)
    assert label_lesions(lesions > 0).max() == 2


def test_rejects_nonpositive_map_and_grid_mismatch():
    base, follow, lesions, wm = _scene()
    with pytest.raises(ValueError, match="positive"):
        _measure(base - 100.0, follow, lesions, wm)
    with pytest.raises(ValueError, match="one grid"):
        measure(base, follow[:-1], lesions, wm)


def test_cli_run(tmp_path):
    base, follow, lesions, wm = _scene()
    base[0, 0, 0] = np.nan  # maps are often NaN outside the brain
    for name, arr in (("b", base), ("f", follow), ("l", lesions), ("w", wm.astype(np.uint8))):
        nib.save(nib.Nifti1Image(arr, np.diag([1.0, 1.0, 1.0, 1.0])), tmp_path / f"{name}.nii.gz")
    out = tmp_path / "out"
    main(["run", "--baseline", str(tmp_path / "b.nii.gz"), "--followup", str(tmp_path / "f.nii.gz"),
          "--lesions", str(tmp_path / "l.nii.gz"), "--wm", str(tmp_path / "w.nii.gz"),
          "--out", str(out)])
    summary = json.loads((out / "summary.json").read_text())
    assert summary["n_lesions"] == 2 and summary["n_repaired"] == 1
    assert (out / "lesions.csv").exists()
    codes = nib.load(out / "repair_labels.nii.gz").get_fdata()
    assert set(np.unique(codes)) == {0, 1, 2}
    assert nib.load(out / "lesion_ids.nii.gz").get_fdata().max() == 2
