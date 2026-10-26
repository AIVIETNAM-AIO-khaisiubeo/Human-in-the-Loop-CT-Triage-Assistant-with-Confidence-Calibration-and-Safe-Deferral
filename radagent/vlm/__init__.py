"""Vision-language second reader (MedGemma 1.5 by default).

``vlm.backend``: ``medgemma`` | ``none`` | ``plugin`` (``vlm.plugin: "package.module:ClassName"``,
a :class:`radagent.vlm.base.VLMReader` constructed as ``ClassName(cfg)``).
"""
from __future__ import annotations

import hashlib
import json

from radagent.vlm.prompts import PROMPT_VERSION


def vlm_enabled(cfg: dict) -> bool:
    return cfg.get("vlm", {}).get("backend", "none") not in (None, "none", "")


def vlm_signature(cfg: dict) -> str:
    """Identifies a cached read: model, prompt version and slice preparation."""
    v = cfg["vlm"]
    key = {"backend": v["backend"], "plugin": v.get("plugin"), "model": v.get("model_id"), "prompt": PROMPT_VERSION,
           "n": v["n_slices"], "windows": v["windows"], "describe": bool(v.get("describe", True))}
    return hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]


def load_vlm(cfg: dict):
    backend = cfg["vlm"]["backend"]
    if backend == "medgemma":
        from radagent.vlm.medgemma import MedGemmaReader

        return MedGemmaReader(cfg)
    if backend == "plugin":
        from radagent.plugins import load_class

        return load_class(cfg["vlm"]["plugin"])(cfg)
    raise ValueError(f"unknown vlm backend {backend!r} (use medgemma, none or plugin)")
