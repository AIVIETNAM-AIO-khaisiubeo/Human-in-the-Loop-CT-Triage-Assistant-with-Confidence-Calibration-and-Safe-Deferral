"""LiTS17 data: download from Kaggle, pair CTs with masks, leakage-safe splits.

Data source
-----------
The Liver Tumor Segmentation Challenge (LiTS17, ISBI 2017 / MICCAI 2017) training set,
131 contrast-enhanced abdominal CT volumes with reference masks, as mirrored on Kaggle in
two parts:

* ``andrewmvd/liver-tumor-segmentation``         - ``segmentations/`` + ``volume_pt1`` ... ``volume_pt5``
* ``andrewmvd/liver-tumor-segmentation-part-2``  - the remaining ``volume_pt*`` folders

Files are ``volume-<n>.nii`` (CT in Hounsfield units) and ``segmentation-<n>.nii``
(0 background, 1 liver, 2 tumour). Both parts are needed to pair every CT with its mask.

Split
-----
Nothing is trained in this project (the segmenter is the pretrained NVIDIA VISTA3D), so the
cases are split patient-level at random into **calib** (fits the calibration of the model's
case score) and **test** (evaluation). Note that VISTA3D's own training data include MSD
Task 03, which is built from these LiTS training volumes - see README section "Model".
"""
from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


def case_number(path: str) -> int:
    """Case number = last number in the file name (``volume-12.nii`` -> 12)."""
    return int(re.findall(r"\d+", os.path.basename(path).split(".")[0])[-1])


def discover_pairs(dirs: list[str | Path]) -> dict[int, tuple[str, str]]:
    files: list[str] = []
    for d in dirs:
        files += glob.glob(f"{d}/**/*.nii*", recursive=True)
    files = sorted(files)
    cts = {case_number(p): p for p in files if os.path.basename(p).startswith("volume")}
    masks = {case_number(p): p for p in files if os.path.basename(p).startswith("segmentation")}
    ids = sorted(set(cts) & set(masks))
    return {i: (cts[i], masks[i]) for i in ids}


def download_lits(cfg: dict, verbose: bool = True) -> list[str]:
    """Download (or find in the local kagglehub cache) both Kaggle parts and return their folders.

    Needs Kaggle credentials: a ``KAGGLE_API_TOKEN`` (or ``KAGGLE_USERNAME`` + ``KAGGLE_KEY``)
    environment variable / Colab secret, or ``~/.kaggle/kaggle.json``. Part 1 is ~5 GB and
    part 2 ~11 GB; a second call returns immediately from the cache.
    """
    try:
        import kagglehub
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("kagglehub is not installed - pip install kagglehub") from e
    paths = []
    for ds in cfg["data"]["kaggle_datasets"]:
        if verbose:
            print(f"[download] {ds} (cached after the first download) ...", flush=True)
        try:
            paths.append(str(kagglehub.dataset_download(ds)))
        except Exception as e:  # pragma: no cover - depends on network/credentials
            raise RuntimeError(
                f"Kaggle download of {ds} failed ({type(e).__name__}: {e}). Set KAGGLE_API_TOKEN "
                "(Kaggle -> Settings -> API -> Generate New Token), or KAGGLE_USERNAME + KAGGLE_KEY, "
                "or point data.lits_dirs at folders that already contain the LiTS files.") from e
        if verbose:
            print(f"[download]   -> {paths[-1]}")
    return paths


LOCAL_PATHS_FILE = Path("data") / "lits_paths.json"


def resolve_lits_dirs(cfg: dict, allow_download: bool = True, verbose: bool = True) -> list[str]:
    """Folders holding the LiTS files: config -> remembered download -> kagglehub download."""
    dirs = [str(d) for d in (cfg["data"].get("lits_dirs") or []) if Path(d).exists()]
    if dirs:
        return dirs
    if LOCAL_PATHS_FILE.exists():
        dirs = [d for d in json.loads(LOCAL_PATHS_FILE.read_text()) if Path(d).exists()]
        if dirs:
            return dirs
    if not allow_download:
        raise FileNotFoundError("LiTS folders not found - run `python -m radagent download` first "
                                "or set data.lits_dirs in the config")
    dirs = download_lits(cfg, verbose=verbose)
    LOCAL_PATHS_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_PATHS_FILE.write_text(json.dumps(dirs, indent=1))
    return dirs


def locate(path: str, cfg: dict) -> str:
    """Find a LiTS file on this machine: the stored path, else the same file name under the local LiTS
    folders (lets a run made on Colab be recomputed on a PC that has its own copy of LiTS)."""
    if path and Path(str(path)).exists():
        return str(path)
    name = Path(str(path).replace("\\", "/")).name
    try:
        dirs = resolve_lits_dirs(cfg, allow_download=False, verbose=False)
    except FileNotFoundError:
        return str(path)
    for d in dirs:
        hit = next(Path(d).rglob(name), None)
        if hit is not None:
            return str(hit)
    return str(path)


def random_split(ids: list[int], fractions: dict[str, float], seed: int = 13) -> dict[int, str]:
    arr = np.array(sorted(ids))
    np.random.RandomState(seed).shuffle(arr)
    names = ["train", "val", "calib", "test"]
    fr = np.array([fractions.get(n, 0.0) for n in names], dtype=float)
    fr = fr / fr.sum()
    counts = np.floor(fr * len(arr)).astype(int)
    counts[-1] = len(arr) - counts[:-1].sum()
    split, k = {}, 0
    for name, n in zip(names, counts):
        for i in arr[k:k + n]:
            split[int(i)] = name
        k += n
    return split


def _portable(p: str) -> str:
    """Relative to the working directory when inside it (portable repo), absolute otherwise."""
    ap = Path(p).resolve()
    try:
        return str(ap.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(ap)


def build_manifest(cfg: dict, allow_download: bool = True) -> pd.DataFrame:
    d = cfg["data"]
    if d.get("source", "lits") != "lits":
        raise ValueError(f"unsupported data.source {d['source']!r} - this project uses LiTS17")
    dirs = resolve_lits_dirs(cfg, allow_download=allow_download)
    pairs = discover_pairs(dirs)
    if not pairs:
        raise FileNotFoundError(f"no volume-*/segmentation-* pairs found under {dirs}")
    ids = sorted(pairs)
    split = random_split(ids, d["split_fractions"], seed=cfg["seed"])
    rows = [{"case_id": f"lits_{i:03d}", "number": i, "ct_path": _portable(pairs[i][0]),
             "mask_path": _portable(pairs[i][1]), "split": split[i]} for i in ids]
    df = pd.DataFrame(rows)
    if d.get("max_cases"):
        # keep the subset proportional per split (deterministic)
        keep = []
        frac = min(1.0, d["max_cases"] / len(df))
        for _s, g in df.groupby("split", sort=False):
            keep.append(g.head(max(1, int(round(frac * len(g))))))
        df = pd.concat(keep).sort_values("number").reset_index(drop=True)
    return df
