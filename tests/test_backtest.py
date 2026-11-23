import json

import numpy as np
import pandas as pd

from repairgauge.backtest import headline, run_planted, run_real, score
from repairgauge.cli import main
from repairgauge.core import Config
from repairgauge.synth import make_phantom, plant_pair

SMALL = dict(shape=(72, 72, 56), sizes_mm3=(50, 300), recoveries=(0.0, 0.5, 1.0, -0.5))


def test_phantom_known_answer():
    """Fast known-answer test: planted recovery comes back, labels match the truth."""
    rows = run_planted(make_phantom(7, noise=0.02, shift_vox=0.1, **SMALL), Config())
    assert len(rows) == 8 and rows.judged.all()
    err = (rows.recovery - rows.recovery_true).abs()
    assert err.median() < 0.1 and err.max() < 0.35
    expected = rows.recovery_true.map(lambda r: "repaired" if r > 0 else "worsened" if r < 0 else "unchanged")
    assert (rows.label == expected).all()
    head = headline(rows)
    assert head["false_repaired_at_zero"] == 0.0 and head["sensitivity_repaired_any_recovery"] == 1.0
    assert set(score(rows).columns) >= {"size_mm3", "recovery_true", "recovery_mae", "called_repaired"}


def test_phantom_is_reproducible():
    a, b = make_phantom(3, **SMALL), make_phantom(3, **SMALL)
    assert np.array_equal(a.followup, b.followup) and a.truth.equals(b.truth)
    assert not np.array_equal(a.followup, make_phantom(4, **SMALL).followup)
    pd.testing.assert_frame_equal(run_planted(a, Config()), run_planted(b, Config()))


def test_plant_pair_keeps_the_scans_and_plants_the_truth():
    p = make_phantom(5, deficit=0.0, noise=0.02, shift_vox=0.0, **SMALL)
    blank = plant_pair(p.baseline, p.followup, p.wm, p.zooms, seed=1, deficit=0.0)
    assert np.array_equal(blank.baseline, p.baseline) and (blank.truth.deficit_true == 0).all()
    planted = plant_pair(p.baseline, p.followup, p.wm, p.zooms, seed=1, deficit=0.3)
    inside = planted.lesions > 0
    assert planted.baseline[inside].mean() < 0.85 * p.baseline[inside].mean()
    rows = run_real(p.baseline, p.followup, p.wm, p.zooms, [1], [0.3], Config())
    judged = rows[rows.judged]
    assert len(judged) >= 0.7 * len(rows)
    assert (judged.recovery - judged.recovery_true).abs().median() < 0.15


def test_cli_backtest_and_phantom(tmp_path):
    main(["backtest", "--out", str(tmp_path / "bt"), "--seeds", "0", "--noise", "0.03",
          "--shift", "0.2", "--deficit", "0.3"])
    head = json.loads((tmp_path / "bt" / "headline.json").read_text())
    assert head["planted"]["n_lesions"] >= 25
    assert head["planted"]["sensitivity_repaired_full_recovery"] > 0.7
    assert (tmp_path / "bt" / "by_size_recovery.csv").exists()
    main(["phantom", "--out", str(tmp_path / "ph"), "--seed", "0"])
    main(["run", "--baseline", str(tmp_path / "ph" / "baseline.nii.gz"),
          "--followup", str(tmp_path / "ph" / "followup.nii.gz"),
          "--lesions", str(tmp_path / "ph" / "lesions.nii.gz"),
          "--wm", str(tmp_path / "ph" / "wm.nii.gz"), "--out", str(tmp_path / "run")])
    table = pd.read_csv(tmp_path / "run" / "lesions.csv")
    truth = pd.read_csv(tmp_path / "ph" / "truth.csv")
    assert len(table) == len(truth)
