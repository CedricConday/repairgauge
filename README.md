# RepairGauge

RepairGauge measures, lesion by lesion, whether a myelin-sensitive MRI value recovered
between two scans of the same person. It also tests itself on known answers before you
trust it.

> **Nothing here is medical advice or evidence about multiple sclerosis.** It is research
> software. It has not been checked against histology or patient outcome.

## Why

Open tools already compute myelin maps (qMRLab, QUIT, DECAES, hMRI), and open tools segment
lesions (LST-AI). I found no open tool that follows a myelin value inside each lesion over
time and comes with a known-answer test. That is what a myelin-repair trial needs to read
out. RepairGauge is that missing step. It does not make the maps and does not segment the
lesions; it takes both as inputs.

## What it needs

* a baseline and a follow-up map of the same measure (MTR, MWF, ihMTR, T1w/T2w, R1, ...);
* a baseline lesion label image, or a binary mask, which is split into connected parts;
* a normal-appearing white matter (NAWM) mask, or a white matter mask. Lesions plus a 2 mm
  margin and a 1 mm rim are then removed from it.

All inputs must already be on one grid. **RepairGauge does no registration.** If the grids
differ, it stops and says so.

## What it measures

For every lesion:

1. **Reference.** Each visit is divided by a local NAWM centre (Gaussian weight, 10 mm).
   A global scale change or a smooth bias between visits cancels in that ratio. Values are
   expressed in units of the baseline NAWM spread.
2. **Deficit, change, recovery.** The deficit is how far the lesion sits below the reference
   at baseline. The change is follow-up minus baseline. Recovery is change divided by deficit:
   1 means back to the reference, 0 means no change, a negative value means it got worse.
3. **Noise floor.** The lesion's own shape is dropped at 1,000 random places in NAWM, and
   the same change is measured there. This shows what "no change" looks like for a lesion
   of this size and shape on this pair of scans. A small lesion therefore needs a bigger
   change than a large one. A term for 0.3 mm of residual misalignment is added.
4. **Label.** z = change against that floor. The label is repaired at z ≥ 2.5, worsened at
   z ≤ −2.5, otherwise unchanged. A lesion is flagged `too_small` under 10 voxels, and
   `shallow_deficit` when its deficit is under 3 NAWM spreads or under 4 noise floors.
   Flagged lesions still get a label, but they are not "judged", and recovery is not quoted
   for them.

Per person: counts per label, volume-weighted recovery, and the share of judged lesion
volume that was repaired. Outputs are `lesions.csv`, `summary.json`, `lesion_ids.nii.gz`
and `repair_labels.nii.gz` (1 repaired, 2 unchanged, 3 worsened, 4 undetermined).
Every threshold is in `repairgauge.core.Config` and on the command line.

## Install and use

```bash
pip install repairgauge
repairgauge run --baseline mtr_v1.nii.gz --followup mtr_v2.nii.gz \
    --lesions lesions_v1.nii.gz --wm wm_v1.nii.gz --out out/
```

Worked example on the built-in phantom (seed 0: 30 lesions, five sizes, six recovery
levels, deficit 25 %, follow-up with its own noise, a 5 % scale change, a bias field and
a 0.3 voxel shift):

```bash
repairgauge phantom --out ph/
repairgauge run --baseline ph/baseline.nii.gz --followup ph/followup.nii.gz \
    --lesions ph/lesions.nii.gz --wm ph/wm.nii.gz --out ph/out
# 30 lesions, 30 judged: 19 repaired, 6 unchanged, 5 worsened; volume-weighted recovery 0.33
```

The 1,500 mm³ lesions in that run, true recovery against measured: −0.50 → −0.47
(worsened), 0 → 0.00 (unchanged), 0.25 → 0.27, 0.50 → 0.50, 0.75 → 0.75, 1.00 → 0.88
(all repaired). `ph/truth.csv` holds the answers for every lesion.

## Known-answer results (2026-11-25, v0.1.0 defaults)

The defaults were fixed on development data first: phantom seeds 100–102, plus real pairs
from two subjects, sub-22449 and sub-24037. The numbers below come from seeds 0–4 and four
other subjects. The method is in `docs/VALIDATION.md`.

**Phantom**, 3,600 planted lesions (5 seeds × 3 noise levels × 4 shifts × 2 deficits × 30
lesions):

| | shift ≤ 0.5 voxel | shift 1 voxel |
|---|---:|---:|
| lesions judged | 95 % | 95 % |
| recovery error, judged lesions (mean absolute) | 0.07 | 0.14 |
| false "repaired" at zero recovery | **0 of 450** | 53 of 150 (35 %) |
| any recovery called repaired | 93 % | 97 % |
| 25 % recovery called repaired | 80 % | 92 % |
| worsening called worsened | 90 % | 70 % |

**Real noise.** MIRROR (OpenNeuro ds008825) is 22 healthy adults scanned twice on the same
day, so nothing changed between the two scans. The same lesions were planted into both
scans of sub-24762, sub-60476, sub-60501 and sub-60505: 15 % or 30 % deficit, five seeds,
1,084 lesions per map. Regions where nothing was planted were scored as well (542).

| | MTR | ihMTR |
|---|---:|---:|
| planted lesions judged | 68 % | 9 % |
| recovery error, judged (mean absolute) | 0.11 | 0.25 |
| false "repaired" at zero recovery, judged | 0 | 0 |
| full recovery called repaired | 95 % | 68 % |
| any recovery called repaired | 73 % | 44 % |
| worsening called worsened | 78 % | 38 % |
| nothing planted: any change called | 3.5 % (19 of 542) | 1.8 % (10 of 542) |

On MTR, 30 % deficits are judged 92 % of the time and 15 % deficits 44 %. A 30 % deficit
that fully recovers is called repaired every time. ihMTR is too noisy at this resolution
to judge most lesions: 89 % are flagged `shallow_deficit`. The tool says so lesion by
lesion instead of guessing.

## What this does not show

* **A planted change is not histology.** A lowered value that comes back is a test of the
  arithmetic and the noise floor, not of myelin biology.
* **No patient data yet.** Real MS lesions have edges, rims and partial volume that the
  planted ellipsoids do not.
* **Aligned inputs are assumed.** A one-voxel residual shift between visits turns 35 % of
  unchanged phantom lesions into false "repaired" calls. Register well, or treat the
  result as unreliable.
* **Large regions on real scans.** Where nothing was planted, regions of 1,500 mm³ were
  called changed 15 % of the time on MTR and 10 % on ihMTR (11 and 7 of 72); smaller
  regions stayed between 0 and 4 %. Real scan differences are spatially correlated in a
  way the null does not fully capture at that size. The same size over-reads full
  recovery on MTR (mean 1.29). Large lesions also often did not fit into the white matter
  of these brains, so some cells have few lesions (as few as 4).
* Five sizes, two deficits, four subjects, two maps. MWF, R1 and T1w/T2w were not tested.

## Next

The next known answer is the marmoset EAE dataset: serial 7 T MRI with histology for
remyelination per lesion (eLife, doi:10.7554/eLife.73786, CC0). It has not been started.

## Credits

* MIRROR test-retest dataset, OpenNeuro ds008825 (CC0); code at github.com/PennLINC/mirror.
* NumPy, SciPy, pandas, nibabel.

## Licence

MIT, © 2026 Cedric Conday.
