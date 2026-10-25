"""LiTS discovery, case pairing and leakage-safe splits.

The Kaggle mirrors of LiTS (``andrewmvd/liver-tumor-segmentation`` and
``...-part-2``) store ``volume-<n>.nii`` CTs and ``segmentation-<n>.nii`` masks
(0 background, 1 liver, 2 tumour). The synthetic phantoms use the same names.

Split strategies
----------------
``random``
    Patient-level random split into train / val / calib / test.
``notebook_compatible``
    Reproduces the split of the ``ViTUNETR25D_LiverSegmentation`` notebook exactly
    (``RandomState(42)`` shuffle, first 80 cases = training pool, 15 % of them =
    validation). Its 51 held-out cases are then divided into **calib** and **test**.
    Use this when warm-starting from that notebook's checkpoint: the checkpoint has
    seen the 80 training-pool cases, so none of them may be used for calibration,
    reader-policy fitting or testing.
"""
from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


def case_number(path: str) -> int:
    """Case number = last number in the file name (``volume-12.nii`` -> 12), as in the notebook."""
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


def notebook_compatible_split(ids: list[int], seed: int = 42, n_train: int = 80, val_frac: float = 0.15,
                              calib_fraction: float = 0.5, heldout_seed: int = 13) -> dict[int, str]:
    arr = np.array(sorted(ids))
    np.random.RandomState(seed).shuffle(arr)
    train_pool = arr[:n_train].tolist()
    heldout = sorted(arr[n_train:].tolist())
    n_val = int(val_frac * len(train_pool))
    val_ids, train_ids = train_pool[:n_val], train_pool[n_val:]
    h = np.array(heldout)
    np.random.RandomState(heldout_seed).shuffle(h)
    n_cal = int(round(calib_fraction * len(h)))
    split = {i: "train" for i in train_ids}
    split.update({i: "val" for i in val_ids})
    split.update({int(i): "calib" for i in h[:n_cal]})
    split.update({int(i): "test" for i in h[n_cal:]})
    return split


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


def build_manifest(cfg: dict) -> pd.DataFrame:
    d = cfg["data"]
    if d["source"] == "synthetic":
        dirs, prefix = [d["root"]], "syn"
    elif d["source"] == "lits":
        dirs, prefix = list(d.get("lits_dirs") or []), "lits"
        if not dirs:
            raise ValueError("data.lits_dirs is empty - point it at the kagglehub LiTS folders")
    else:
        raise ValueError(f"unknown data.source {d['source']!r}")
    pairs = discover_pairs(dirs)
    if not pairs:
        raise FileNotFoundError(f"no volume-*/segmentation-* pairs found under {dirs}")
    ids = sorted(pairs)
    if d["split_strategy"] == "notebook_compatible":
        split = notebook_compatible_split(ids, seed=d["notebook_seed"], n_train=d["notebook_n_train"],
                                          val_frac=d["notebook_val_frac"],
                                          calib_fraction=d["calib_fraction_of_heldout"], heldout_seed=cfg["seed"])
    else:
        split = random_split(ids, d["split_fractions"], seed=cfg["seed"])
    def portable(p: str) -> str:
        """Relative to the working directory when inside it (portable repo), absolute otherwise."""
        ap = Path(p).resolve()
        try:
            return str(ap.relative_to(Path.cwd().resolve()))
        except ValueError:
            return str(ap)

    rows = [{"case_id": f"{prefix}_{i:03d}", "number": i, "ct_path": portable(pairs[i][0]),
             "mask_path": portable(pairs[i][1]), "split": split[i]} for i in ids]
    df = pd.DataFrame(rows)
    if d.get("max_cases"):
        # keep the subset proportional per split (deterministic)
        keep = []
        frac = min(1.0, d["max_cases"] / len(df))
        for _s, g in df.groupby("split", sort=False):
            keep.append(g.head(max(1, int(round(frac * len(g))))))
        df = pd.concat(keep).sort_values("number").reset_index(drop=True)
    return df
