# Validation record

All numbers are RepairGauge 0.1.0, run on 2026-11-25 on 4 CPU cores (arm64). The phantom
run takes 2 min 40 s, the real pairs 10 min.

## How the defaults were fixed

The defaults were chosen on a development set and then frozen. The reported runs below use
other seeds and other subjects.

* **Phantom, seeds 100–102** (`scripts/fix_defaults.py --out results/defaults_dev --jobs 4`):
  NAWM reference width 0 / 10 / 20 / 40 mm, alignment tolerance 0 / 0.15 / 0.3 / 0.5 mm,
  z 2.0 / 2.5 / 3.0, over noise 2 / 4 / 8 %, shift 0–1 voxel and deficit 15 / 30 %. The
  table is in `results/defaults_dev/candidates.csv`.
  At tolerance 0.3 mm and z 2.5, false "repaired" at zero recovery was 0 for shifts up to
  0.5 voxel at every width, so the width was set by how many lesions it could judge.
* **Real pairs, sub-22449 and sub-24037, seeds 100–102**, width 0 / 10 / 20 mm. On MTR
  10 mm judged 68 % of planted lesions, against 46 % at 20 mm and 34 % with one global
  value, with no false calls at zero recovery in any of the three. Logs:
  `work/logs/dev_real_sigma*.log` (not in git).
* Frozen: width 10 mm, tolerance 0.3 mm, z 2.5, minimum 10 voxels, deficit at least
  3 NAWM spreads and 4 noise floors (commit 82a9a3d).

## Phantom (seeds 0–4)

```bash
repairgauge backtest --out results/phantom --jobs 4
```

112 × 112 × 80 at 1 mm, white matter texture. Per phantom: 30 lesions (15, 50, 150,
500, 1,500 mm³ × recovery 0, 0.25, 0.5, 0.75, 1, −0.5). The follow-up gets its own noise,
a 5 % scale change, a 3 % bias field and a rigid shift. Grid: 5 seeds × noise 2 / 4 / 8 %
× shift 0 / 0.25 / 0.5 / 1 voxel × deficit 15 / 30 %, which is 3,600 lesions.

Headline (`results/phantom/headline.json`, all shifts pooled): judged 95 %; recovery error
on judged lesions 0.084 (bias +0.043); false "repaired" at zero recovery 8.8 %, all of it
at a 1 voxel shift (53 of 150; 0 of 450 at ≤ 0.5); sensitivity to any recovery 94 %, to
full recovery 99 %; specificity of "repaired" 96 %; sensitivity to worsening 85 %.

By shift (`results/phantom/by_shift_vox_recovery.csv`), the rate called repaired:

| shift (voxel) | recovery 0 | 0.25 | 0.5 | 1.0 | worsening called |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.00 | 0.78 | 0.93 | 0.99 | 0.93 |
| 0.25 | 0.00 | 0.79 | 0.95 | 0.99 | 0.91 |
| 0.5 | 0.00 | 0.84 | 0.97 | 0.99 | 0.86 |
| 1.0 | **0.35** | 0.92 | 0.97 | 0.99 | 0.70 |

A one-voxel shift pushes recovery at zero to a mean of +0.15. A misaligned follow-up
smears the lesion edge into NAWM, and that reads as recovery. The alignment term covers
0.3 mm, not 1 mm.

By size (`by_size_recovery.csv`): 15 mm³ lesions are the weak cell. Only 50–90 % are
judged, 25 % recovery is called 25 % of the time, worsening 40 %, and 10 % are falsely
called repaired at zero. From 50 mm³ up, any recovery of 25 % or more is called in 95–100 %
of cases. Large lesions (150–1,500 mm³) are falsely called repaired at zero 9–13 % of the
time, pooled over shifts. By noise (`by_noise_recovery.csv`): at 8 % voxel noise, 25 %
recovery is called 76 % of the time against 90 % at 2 %.

## Real noise: MIRROR, OpenNeuro ds008825 (four held-out subjects, seeds 0–4)

```bash
python scripts/fetch_mirror.py --out work/mirror --n 8      # about 30 MB per subject
python scripts/real_backtest.py --out results/real_mirror \
    --subjects sub-24762 sub-60476 sub-60501 sub-60505 --seeds 0 1 2 3 4 --jobs 4
```

These are the MTR and ihMTR maps of both same-day sessions in each subject's T1w space,
with that space's white matter mask from MP2RAGE. Six subjects were fetched, the first in
listing order. Two went to development; these four are the rest. Lesions are planted at
15 % and 30 % deficit with the same sizes and recovery levels. A run with deficit 0 plants
nothing and scores the same regions, so any change called there is false.

| | MTR | ihMTR |
|---|---:|---:|
| planted lesions | 1,084 | 1,084 |
| judged | 68.3 % | 9.2 % |
| flagged shallow_deficit / too_small | 328 / 16 | 968 / 16 |
| recovery error, judged (mean absolute; bias) | 0.115 (+0.033) | 0.254 (−0.050) |
| false "repaired" at zero recovery: all / judged | 0.6 % / 0 | 0 / 0 |
| false "worsened" at zero recovery: all / judged | 0 / 0 | 0.6 % / 0 |
| full recovery called repaired | 94.9 % | 68.4 % |
| any recovery called repaired | 73.5 % | 43.5 % |
| specificity of "repaired" | 99.7 % | 100 % |
| worsening called worsened | 77.5 % | 37.5 % |
| nothing planted: repaired / worsened / any | 2.8 % / 0.7 % / 3.5 % | 0.2 % / 1.7 % / 1.8 % |

MTR by deficit (`results/real_mirror/MTRmap/by_deficit_recovery.csv`): at 30 %, 89–95 % are
judged, and recovery of 0.5 / 0.75 / 1 is called repaired 90 / 99 / 100 % of the time;
25 % recovery only 41 %. At 15 %, a third to a half are judged, and 50 % recovery is called
54 % of the time.

MTR by size (`by_size_recovery.csv`): 15 mm³ lesions are judged 33–40 % of the time and
25 % recovery is never called. At 1,500 mm³, recovery is over-read (mean 1.14 at 0.75 and
1.29 at 1.0), and only 4 of 40 zero-recovery lesions fitted into the white matter, so
false calls at that size are barely tested.

Nothing planted, by size (`nothing_planted_by_size.csv`), the rate of any change called:

| size (mm³) | 15 | 50 | 150 | 500 | 1,500 |
|---|---:|---:|---:|---:|---:|
| MTR | 0.017 | 0.000 | 0.042 | 0.009 | **0.153** |
| ihMTR | 0.008 | 0.008 | 0.000 | 0.009 | **0.097** |

Almost every unplanted region is flagged (MTR 99.8 %, ihMTR 97.6 %), because healthy white
matter has no deficit. The calls are reported anyway: they show that a large region on a
real pair moves more than the pseudo-lesion null predicts.

## Unverified

* No patient scans, no histology. The marmoset EAE set (doi:10.7554/eLife.73786) is the
  next known answer.
* MWF, R1 and T1w/T2w maps were not tested, and no field strength other than MIRROR's.
* The real pairs were scanned on the same day. Drift between visits months apart is
  simulated only on the phantom (scale and bias).
