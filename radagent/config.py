"""Configuration: built-in defaults deep-merged with a YAML file.

Every experiment is driven by one YAML file (see ``configs/``). Anything not set in
the YAML falls back to ``DEFAULTS`` below, so a config only needs to state what it
changes.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "run_name": "demo",
    "output_root": "outputs",
    "seed": 13,
    # ------------------------------------------------------------------ data
    "data": {
        "source": "synthetic",            # synthetic | lits
        "root": "data/synthetic",         # synthetic phantoms are written here
        "lits_dirs": [],                   # LiTS folders (kagglehub part1 + part2)
        "n_cases": 120,                    # synthetic only
        "max_cases": None,                 # optional cap (LiTS subset)
        "split_strategy": "random",       # random | notebook_compatible
        "split_fractions": {"train": 0.40, "val": 0.07, "calib": 0.265, "test": 0.265},
        # notebook_compatible: reproduce the ViT-UNETR notebook split so that a
        # warm start from that checkpoint never sees a calibration/test case.
        "notebook_seed": 42,
        "notebook_n_train": 80,
        "notebook_val_frac": 0.15,
        "calib_fraction_of_heldout": 0.5,
    },
    # ------------------------------------------------------------ preprocess
    "preprocess": {
        "hu_window": [-100, 250],          # soft-tissue / liver window (HU)
        "img_size": 128,                   # in-plane size seen by the segmenter
        "margin_slices": 8,                # slices kept around the liver for training
    },
    # ------------------------------------------------------------- segmenter
    "segmenter": {
        "backend": "voxel_gbm",           # vitunetr | unet25d | voxel_gbm
        "checkpoint": None,                # default: <run>/model/<backend>.pt|.pkl
        "init_from": None,                 # warm start, e.g. the 2-class ViT-UNETR .pt
        "vit_name": "vit_small_patch16_224.augreg_in21k_ft_in1k",
        "pretrained": True,                # ImageNet weights for the ViT encoder
        "decoder_dropout": 0.1,            # enables MC-dropout uncertainty
        "unet_base": 16,
        "epochs": 30,
        "batch_size": 16,
        "lr_encoder": 1.0e-4,
        "lr_decoder": 5.0e-4,
        "weight_decay": 1.0e-2,
        "grad_clip": 1.0,
        "num_workers": 2,
        "class_weights": [1.0, 1.0, 3.0],  # background, liver, tumour (CE term)
        "tumour_oversample": 3.0,          # sampling weight of slices with tumour
        "device": "auto",
        "gbm": {
            "n_members": 3,                # bootstrap ensemble -> epistemic spread
            "pixels_per_case": 6000,
            "max_iter": 150,
            "learning_rate": 0.1,
            "max_leaf_nodes": 31,
        },
    },
    # ------------------------------------------------------------- inference
    "inference": {
        "tta_flip": True,
        "mc_samples": 2,                   # stochastic passes per TTA view (torch)
        "batch_size": 16,
        "min_lesion_ml": 0.1,              # smallest lesion reported / looked for by the case score
        "min_lesion_peak_prob": 0.7,       # a lesion candidate must reach this tumour probability somewhere
        "viewer_size": None,               # None -> img_size
        "viewer_max_slices": 96,
        "eval_splits": ["calib", "test"],
    },
    # ------------------------------------------------------------ confidence
    "confidence": {
        "method": "auto",                  # temperature | platt | auto (CV on calib)
        "n_bins": 10,
        "target_min_tumour_ml": 0.0,       # reference "tumour present" if volume > this
    },
    # --------------------------------------------------------------- quality
    "quality": {
        "liver_ml_range": [500.0, 4500.0],
        "spacing_mm_range": [0.3, 8.0],
        "min_slices": 16,
        "max_tumour_outside_liver_frac": 0.25,
        "min_outside_ml": 1.0,             # ... ignored if the volume outside the liver is tiny
    },
    # ---------------------------------------------------------------- triage
    "triage": {
        "suggest_threshold": 0.5,
        "urgent_probability": 0.85,
        "urgent_volume_ml": 5.0,
        "urgent_diameter_mm": 20.0,
        "uncertain_flag": 0.85,            # bits of predictive entropy (p between ~0.28 and ~0.72)
        "volume_ref_ml": 50.0,
        "weights": {"probability": 0.60, "volume": 0.25, "uncertainty": 0.15},
    },
    # ---------------------------------------------------------------- policy
    "policy": {
        "universal_threshold": 0.5,
        "threshold_grid": [round(0.10 + 0.05 * i, 2) for i in range(17)],
        "defer_grid": [None, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
        "w_fn": 2.0,                       # a missed tumour costs twice a false alarm
        "w_fp": 1.0,
        "burden_weight": 0.02,             # loss per review action
        "action_cost": {"accept": 1, "reject": 2, "edit": 3, "manual": 4},
        "min_feedback": 10,                # personalise only after this many reads
        "min_improvement": 0.005,          # ... and only if it helps by at least this much
        "improvement_se": 1.0,             # ... and by more than 1 standard error of the paired difference
        "shrink_rho": 0.02,                # pull towards the universal policy
        "trust_prior": {"alpha_mean": 0.0, "alpha_sd": 1.5, "beta_mean": 4.0, "beta_sd": 3.0},
    },
    # --------------------------------------------------------------- readers
    "readers": {
        "seeds": 30,
        "learning_curve_n": [0, 5, 10, 15, 20, 25, 32],
        "profiles": {
            "reader_a": {
                "name": "Reader A - over-reliant",
                "description": "Accepts most AI suggestions, switches to the AI easily.",
                "sens_a": 0.8, "sens_b": 1.5, "sens_c": 0.8, "fpr": 0.12,
                "trust_k": 8.0, "trust_tau": 0.55,
                "lapse": 0.02, "edit_poor": 0.2, "edit_noise": 0.02,
            },
            "reader_b": {
                "name": "Reader B - sceptical",
                "description": "Conservative caller; rejects many suggestions, rarely switches.",
                "sens_a": 0.9, "sens_b": 1.8, "sens_c": 0.8, "fpr": 0.04,
                "trust_k": 6.0, "trust_tau": 0.97,
                "lapse": 0.02, "edit_poor": 0.8, "edit_noise": 0.15,
            },
            "reader_c": {
                "name": "Reader C - inconsistent",
                "description": "Noisy, inconsistent decisions and corrections.",
                "sens_a": 0.7, "sens_b": 1.0, "sens_c": 0.6, "fpr": 0.15,
                "trust_k": 1.5, "trust_tau": 0.6,
                "lapse": 0.15, "edit_poor": 0.5, "edit_noise": 0.3,
            },
        },
    },
    # ----------------------------------------------------------------- agent
    "agent": {
        "planner": "rule",                 # rule | llm
        "max_steps": 12,
        "llm": {
            "base_url": "http://localhost:11434/v1",   # any OpenAI-compatible server (Ollama, vLLM, LM Studio)
            "model": "qwen2.5:7b-instruct",
            "api_key_env": "LLM_API_KEY",
            "timeout_s": 60,
            "temperature": 0.0,
        },
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        with open(path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        cfg = deep_merge(cfg, user)
        cfg["_config_path"] = str(path)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    if cfg["inference"].get("viewer_size") is None:
        cfg["inference"]["viewer_size"] = cfg["preprocess"]["img_size"]
    return cfg


def run_dir(cfg: dict) -> Path:
    return Path(cfg["output_root"]) / cfg["run_name"]


def save_config(cfg: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(json.loads(json.dumps(clean)), f, sort_keys=False)
