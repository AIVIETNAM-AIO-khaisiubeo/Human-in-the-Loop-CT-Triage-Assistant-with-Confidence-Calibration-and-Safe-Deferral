"""Training of the segmentation tool (any backend).

The torch path follows the ViT-UNETR notebook closely (windowing, 2.5D stacks, the
same augmentation, Dice + CE, AdamW with a smaller encoder LR, AMP, gradient
clipping) with three changes:

1. three classes (background / liver / tumour) with a class-weighted CE term, a
   batch-level soft Dice on *both* liver and tumour, and tumour-slice oversampling;
2. the cosine schedule spans the *total number of optimiser steps* - in the notebook
   ``CosineAnnealingLR(T_max=epochs)`` was stepped every batch, so the learning rate
   cycled every 30 steps instead of decaying once over training;
3. model selection on the validation split (best mean of liver and tumour Dice).
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from radagent.config import run_dir, save_config
from radagent.geometry import load_volume
from radagent.models import checkpoint_path
from radagent.preprocess import preprocess_volume, resize_stack


# --------------------------------------------------------------------------- data
def load_training_case(row, cfg: dict, cache_dir: Path | None = None) -> dict | None:
    S = cfg["preprocess"]["img_size"]
    if cache_dir is not None:
        f = cache_dir / f"prep_{row['case_id']}_{S}.npz"
        if f.exists():
            d = np.load(f)
            return {"case_id": row["case_id"], "img": d["img"], "lab": d["lab"], "z_keep": np.arange(len(d["img"]))}
    ct = load_volume(row["ct_path"])
    mk = load_volume(row["mask_path"])
    if ct.view.shape != mk.view.shape:
        print(f"[train] skipping {row['case_id']}: CT {ct.view.shape} vs mask {mk.view.shape}")
        return None
    img = preprocess_volume(ct.view, tuple(cfg["preprocess"]["hu_window"]), S)
    lab = resize_stack(np.round(mk.view).astype(np.uint8), S, nearest=True)
    zs = np.where(lab.any(axis=(1, 2)))[0]
    if len(zs) == 0:
        return None
    m = cfg["preprocess"]["margin_slices"]
    z0, z1 = max(zs.min() - m, 0), min(zs.max() + m + 1, len(lab))
    img, lab = img[z0:z1], lab[z0:z1]
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_dir / f"prep_{row['case_id']}_{S}.npz", img=img, lab=lab)
    return {"case_id": row["case_id"], "img": img, "lab": lab, "z_keep": np.arange(len(img))}


def load_split(manifest: pd.DataFrame, split: str, cfg: dict, verbose: bool = True) -> list[dict]:
    cache = run_dir(cfg) / "cache" / "train_slices"
    out, t0 = [], time.time()
    rows = manifest[manifest["split"] == split]
    for k, (_, row) in enumerate(rows.iterrows()):
        c = load_training_case(row, cfg, cache)
        if c is not None:
            out.append(c)
        if verbose and (k % 10 == 0 or k == len(rows) - 1):
            print(f"[train] prepared {k + 1}/{len(rows)} {split} cases ({time.time() - t0:.0f}s)")
    return out


def dice_from_counts(inter: float, a: float, b: float) -> float:
    return float(2 * inter / (a + b)) if (a + b) > 0 else 1.0


# --------------------------------------------------------------------------- GBM
def train_voxel_gbm(cfg: dict, manifest: pd.DataFrame) -> dict:
    from radagent.models.voxel_gbm import FEATURE_NAMES, VoxelGBMSegmenter
    from radagent.preprocess import stacks_25d

    train_cases = load_split(manifest, "train", cfg)
    val_cases = load_split(manifest, "val", cfg)
    seg = VoxelGBMSegmenter(img_size=cfg["preprocess"]["img_size"]).fit(train_cases, cfg["segmenter"]["gbm"], seed=cfg["seed"])
    # validation: ensemble-mean probabilities, pooled slice-wise Dice; also picks the
    # tumour prior-shift correction (tumour pixels were over-sampled during training)
    probs, gts = [], []
    for case in val_cases:
        img, lab = case["img"], case["lab"]
        for b0 in range(0, len(img), 16):
            zs = list(range(b0, min(len(img), b0 + 16)))
            probs.append(seg.predict_batch(stacks_25d(img, zs)).mean(0))
            gts.append(lab[zs])
    P, G = np.concatenate(probs), np.concatenate(gts)

    def val_dice(scale: float) -> dict:
        q = P.copy()
        q[:, 2] *= scale
        pred = q.argmax(1)
        out = {}
        for name, pm, gm in (("liver", pred > 0, G > 0), ("tumour", pred == 2, G == 2)):
            out[name] = round(dice_from_counts(int((pm & gm).sum()), int(pm.sum()), int(gm.sum())), 4)
        return out

    grid = {r: val_dice(r) for r in (1.0, 0.7, 0.5, 0.35, 0.25)}
    best_r = max(grid, key=lambda r: grid[r]["tumour"])
    seg.tumour_prior_scale = float(best_r)
    val = grid[best_r]
    print("[train] tumour prior scale on validation: " + ", ".join(f"{r}: {d['tumour']:.3f}" for r, d in grid.items())
          + f" -> {best_r}")
    path = checkpoint_path(cfg)
    seg.save(path)
    card = {"backend": "voxel_gbm", "checkpoint": str(path), "n_train_cases": len(train_cases),
            "n_val_cases": len(val_cases), "val_dice_slicewise": val, "members": seg.n_passes,
            "tumour_prior_scale": seg.tumour_prior_scale,
            "features": FEATURE_NAMES}
    _write_card(cfg, card)
    print(f"[train] voxel_gbm saved to {path} | validation Dice liver {val['liver']:.3f} tumour {val['tumour']:.3f}")
    return card


def _write_card(cfg: dict, card: dict) -> None:
    d = run_dir(cfg) / "model"
    d.mkdir(parents=True, exist_ok=True)
    (d / "model_card.json").write_text(json.dumps(card, indent=2))
    save_config(cfg, run_dir(cfg) / "config_used.yaml")


# --------------------------------------------------------------------------- torch
def _augment(x: np.ndarray, y: np.ndarray):
    """The notebook's augmentation: flip, rotation+zoom+shift, brightness/contrast, noise."""
    import cv2

    C, H, W = x.shape
    if random.random() < 0.5:
        x = np.ascontiguousarray(np.flip(x, axis=2))
        y = np.ascontiguousarray(np.flip(y, axis=1))
    if random.random() < 0.6:
        M = cv2.getRotationMatrix2D((W / 2, H / 2), random.uniform(-15, 15), random.uniform(0.85, 1.15))
        M[0, 2] += random.uniform(-0.08, 0.08) * W
        M[1, 2] += random.uniform(-0.08, 0.08) * H
        x = np.array([cv2.warpAffine(np.ascontiguousarray(x[c]), M, (W, H), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0) for c in range(C)])
        y = cv2.warpAffine(np.ascontiguousarray(y), M, (W, H), flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if random.random() < 0.5:
        x = np.clip(x.astype(np.float32) * random.uniform(0.9, 1.1) + random.uniform(-10, 10), 0, 255)
    if random.random() < 0.3:
        rng = np.random.default_rng(random.getrandbits(32))
        x = np.clip(x.astype(np.float32) + rng.normal(0, random.uniform(2, 8), x.shape), 0, 255)
    return x, np.ascontiguousarray(y)


def train_torch(cfg: dict, manifest: pd.DataFrame) -> dict:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

    from radagent.models.torch_models import TorchSegmenter, amp_context, build_network, load_warm_start, pick_device
    from radagent.preprocess import to_model_range

    sc = cfg["segmenter"]
    backend = sc["backend"]
    S = cfg["preprocess"]["img_size"]
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    device = pick_device(sc.get("device", "auto"))
    use_amp = device.type == "cuda"
    print(f"[train] backend {backend} on {device} (AMP {use_amp}) | torch {torch.__version__}")

    train_cases = load_split(manifest, "train", cfg)
    val_cases = load_split(manifest, "val", cfg)

    class SliceDataset(Dataset):
        def __init__(self, cases, augment):
            self.cases, self.augment = cases, augment
            self.samples = [(i, z) for i, c in enumerate(cases) for z in range(len(c["img"]))]

        def __len__(self):
            return len(self.samples)

        def has_tumour(self):
            return np.array([bool((self.cases[i]["lab"][z] == 2).any()) for i, z in self.samples])

        def __getitem__(self, idx):
            i, z = self.samples[idx]
            img, lab = self.cases[i]["img"], self.cases[i]["lab"]
            x = np.stack([img[max(0, z - 1)], img[z], img[min(len(img) - 1, z + 1)]], axis=0)
            y = lab[z]
            if self.augment:
                x, y = _augment(x, y)
            x = to_model_range(np.asarray(x))
            return torch.from_numpy(np.ascontiguousarray(x)).float(), torch.from_numpy(y.astype(np.int64))

    train_ds, val_ds = SliceDataset(train_cases, True), SliceDataset(val_cases, False)
    weights = np.where(train_ds.has_tumour(), float(sc["tumour_oversample"]), 1.0)
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(train_ds), replacement=True)
    nw = int(sc.get("num_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=sc["batch_size"], sampler=sampler, drop_last=True,
                              num_workers=nw, pin_memory=use_amp, persistent_workers=nw > 0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=use_amp)
    print(f"[train] slices: train {len(train_ds):,} ({int(train_ds.has_tumour().sum()):,} with tumour) | val {len(val_ds):,}")

    model = build_network(backend, sc, S).to(device)
    if sc.get("init_from"):
        load_warm_start(model, sc["init_from"])
    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    class DiceCELoss3(nn.Module):
        def __init__(self, class_weights, smooth=1.0):
            super().__init__()
            self.register_buffer("w", torch.tensor(class_weights, dtype=torch.float32))
            self.smooth = smooth

        def forward(self, logits, target):
            ce = F.cross_entropy(logits.float(), target, weight=self.w)
            probs = F.softmax(logits.float(), dim=1)
            dices = []
            for c in (1, 2):
                p, t = probs[:, c], (target == c).float()
                inter = (p * t).sum()
                dices.append((2 * inter + self.smooth) / (p.sum() + t.sum() + self.smooth))
            return 0.5 * ce + 0.5 * (1.0 - torch.stack(dices).mean())

    criterion = DiceCELoss3(sc["class_weights"]).to(device)
    enc = list(model.encoder_parameters())
    enc_ids = {id(p) for p in enc}
    dec = [p for p in model.parameters() if id(p) not in enc_ids]
    groups = [{"params": dec, "lr": sc["lr_decoder"]}]
    if enc:
        groups.insert(0, {"params": enc, "lr": sc["lr_encoder"]})
    optimizer = torch.optim.AdamW(groups, weight_decay=sc["weight_decay"])
    total_steps = max(1, sc["epochs"] * len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    try:
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    except (AttributeError, TypeError):  # torch < 2.3
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    @torch.no_grad()
    def evaluate():
        model.eval()
        tot, counts = 0.0, {"liver": [0, 0, 0], "tumour": [0, 0, 0]}
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            with amp_context(device):
                logits = model(x)
            tot += criterion(logits, y).item()
            pred = logits.argmax(1)
            for name, pm, gm in (("liver", pred > 0, y > 0), ("tumour", pred == 2, y == 2)):
                counts[name][0] += int((pm & gm).sum()); counts[name][1] += int(pm.sum()); counts[name][2] += int(gm.sum())
        return tot / max(1, len(val_loader)), {k: dice_from_counts(*v) for k, v in counts.items()}

    path = checkpoint_path(cfg)
    best, history = -1.0, []
    for ep in range(1, sc["epochs"] + 1):
        model.train()
        t0, tr = time.time(), 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), sc["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            tr += loss.item()
        tr /= max(1, len(train_loader))
        vl, vd = evaluate()
        score = 0.5 * (vd["liver"] + vd["tumour"])
        history.append({"epoch": ep, "train_loss": tr, "val_loss": vl, "val_dice_liver": vd["liver"],
                        "val_dice_tumour": vd["tumour"], "lr": optimizer.param_groups[-1]["lr"], "sec": time.time() - t0})
        tag = ""
        if score > best:
            best = score
            TorchSegmenter(model, backend, device, cfg["inference"]["tta_flip"], cfg["inference"]["mc_samples"]).save(
                path, extra={"img_size": S, "epoch": ep, "val": vd, "classes": ["background", "liver", "tumour"]})
            tag = "  <- best, saved"
        print(f"Epoch {ep:02d}/{sc['epochs']:02d} [{time.time() - t0:5.1f}s] | train {tr:.4f} | val {vl:.4f} | "
              f"Dice liver {vd['liver']:.4f} tumour {vd['tumour']:.4f}{tag}")
    hist = pd.DataFrame(history)
    hist.to_csv(run_dir(cfg) / "model" / "training_history.csv", index=False)
    card = {"backend": backend, "checkpoint": str(path), "params_millions": round(n_params, 2),
            "warm_start": sc.get("init_from"), "n_train_cases": len(train_cases), "n_val_cases": len(val_cases),
            "best_epoch": int(hist.loc[(0.5 * (hist.val_dice_liver + hist.val_dice_tumour)).idxmax(), "epoch"]),
            "val_dice_slicewise": {"liver": round(float(hist.val_dice_liver.max()), 4),
                                   "tumour": round(float(hist.val_dice_tumour.max()), 4)}}
    _write_card(cfg, card)
    return card


def train(cfg: dict, manifest: pd.DataFrame) -> dict:
    if cfg["segmenter"]["backend"] == "voxel_gbm":
        return train_voxel_gbm(cfg, manifest)
    return train_torch(cfg, manifest)

