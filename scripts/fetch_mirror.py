"""Fetch the few MIRROR (OpenNeuro ds008825, CC0) files the real-noise backtest needs.

The dataset is 45 GB. This takes, per subject, the MTR and ihMTR maps of both
same-day sessions in the subject's T1w space and the white matter mask in that
space: about 30 MB per subject. Files come from the public OpenNeuro S3 bucket.

    python scripts/fetch_mirror.py --out work/mirror --n 8
"""

from __future__ import annotations

import argparse
import re
import urllib.request
from pathlib import Path

BUCKET = "https://s3.amazonaws.com/openneuro.org"
DS = "ds008825"
WANT = ("_space-T1w_MTRmap.nii.gz", "_space-T1w_ihMTR.nii.gz")


def _list(prefix: str, delimiter: str = "") -> tuple[list[str], list[tuple[str, int]]]:
    url = f"{BUCKET}?list-type=2&prefix={prefix}&delimiter={delimiter}&max-keys=1000"
    xml = urllib.request.urlopen(url, timeout=60).read().decode()
    if "<IsTruncated>true" in xml:
        raise RuntimeError(f"listing of {prefix} is truncated; narrow the prefix")
    dirs = re.findall(r"<CommonPrefixes><Prefix>(.*?)</Prefix>", xml)
    files = [(k, int(s)) for k, s in re.findall(r"<Key>(.*?)</Key>.*?<Size>(\d+)</Size>", xml)]
    return dirs, files


def _get(key: str, out: Path, size: int) -> None:
    dest = out / key.split("/", 1)[1]
    if dest.exists() and dest.stat().st_size == size:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(f"{BUCKET}/{key}", dest)
    print(f"{size / 1e6:7.1f} MB  {dest}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("work/mirror"))
    ap.add_argument("--n", type=int, default=8, help="number of subjects, in listing order")
    args = ap.parse_args()
    dirs, _ = _list(f"{DS}/derivatives/ihmt/sub-", "/")
    subjects = sorted(d.rstrip("/").rsplit("/", 1)[1] for d in dirs)[: args.n]
    total = 0
    for sub in subjects:
        keys = []
        for ses in ("ses-01", "ses-02"):
            _, files = _list(f"{DS}/derivatives/ihmt/{sub}/{ses}/anat/")
            keys += [(k, s) for k, s in files if k.endswith(WANT)]
        _, files = _list(f"{DS}/derivatives/pymp2rage/{sub}/anat/")
        keys += [(k, s) for k, s in files if k.endswith("_space-T1w_desc-wm_mask.nii.gz")]
        if len(keys) != 5:
            print(f"{sub}: expected 5 files, found {len(keys)}; skipped", flush=True)
            continue
        for k, s in keys:
            _get(k, args.out, s)
            total += s
    print(f"done: {total / 1e6:.0f} MB in {args.out}")


if __name__ == "__main__":
    main()
