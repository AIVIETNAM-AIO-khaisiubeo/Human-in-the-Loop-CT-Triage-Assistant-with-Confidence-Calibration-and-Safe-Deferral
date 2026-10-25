"""Segmenter registry. Torch is imported lazily so CPU-only installs still work."""
from __future__ import annotations

from pathlib import Path

from radagent.config import run_dir


def checkpoint_path(cfg: dict) -> Path:
    if cfg["segmenter"].get("checkpoint"):
        return Path(cfg["segmenter"]["checkpoint"])
    ext = ".pkl" if cfg["segmenter"]["backend"] == "voxel_gbm" else ".pt"
    return run_dir(cfg) / "model" / f"{cfg['segmenter']['backend']}{ext}"


def load_segmenter(cfg: dict):
    backend = cfg["segmenter"]["backend"]
    path = checkpoint_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"segmenter checkpoint not found: {path} - run `python -m radagent train` first")
    if backend == "voxel_gbm":
        from radagent.models.voxel_gbm import VoxelGBMSegmenter

        try:
            return VoxelGBMSegmenter.load(path)
        except Exception as e:  # e.g. pickle from an incompatible scikit-learn version
            raise RuntimeError(f"could not load {path} ({type(e).__name__}: {e}). Delete it and retrain with "
                               f"`python -m radagent train --config ...`") from e
    from radagent.models.torch_models import TorchSegmenter

    return TorchSegmenter.load(path, cfg["segmenter"], cfg["preprocess"]["img_size"], cfg["inference"])
