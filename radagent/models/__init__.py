"""Segmenter registry. Torch is imported lazily so the website runs without it.

Backends
--------
``vitunetr``  ViT-UNETR 2.5D (pretrained ViT-Small encoder + U-Net decoder, 3 classes) - default
``unet25d``   small 2.5D U-Net (3 classes)
``plugin``    any adapter class, ``segmenter.plugin: "package.module:ClassName"``, implementing
              :class:`radagent.models.base.Segmenter` and constructed as ``ClassName(cfg)``
"""
from __future__ import annotations

from pathlib import Path

from radagent.config import run_dir

TORCH_BACKENDS = ("vitunetr", "unet25d")


def checkpoint_path(cfg: dict) -> Path:
    if cfg["segmenter"].get("checkpoint"):
        return Path(cfg["segmenter"]["checkpoint"])
    backend = cfg["segmenter"]["backend"]
    if backend == "plugin":
        return run_dir(cfg) / "model" / "plugin.json"
    return run_dir(cfg) / "model" / f"{backend}.pt"


def load_segmenter(cfg: dict):
    backend = cfg["segmenter"]["backend"]
    if backend == "plugin":
        from radagent.plugins import load_class

        return load_class(cfg["segmenter"]["plugin"])(cfg)
    if backend not in TORCH_BACKENDS:
        raise ValueError(f"unknown segmenter backend {backend!r} (use vitunetr, unet25d or plugin)")
    path = checkpoint_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"segmenter checkpoint not found: {path} - run `python -m radagent train` first")
    from radagent.models.torch_models import TorchSegmenter

    return TorchSegmenter.load(path, cfg["segmenter"], cfg["preprocess"]["img_size"], cfg["inference"])
