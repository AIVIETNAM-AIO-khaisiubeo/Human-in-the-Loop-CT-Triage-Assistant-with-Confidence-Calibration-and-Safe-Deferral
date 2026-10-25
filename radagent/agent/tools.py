"""The agent's tools.

Each tool is a plain Python function ``fn(ctx, **args) -> summary`` with a
description (shown to an LLM planner) and preconditions (enforced by the runner -
a planner can *propose* any order, but the runner only executes a tool whose
inputs exist). Tools never write free text about the patient; the only text they
produce is the deterministic report template.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from radagent.calibration import Calibrator
from radagent.confidence import case_confidence
from radagent.quality import quality_check as _qc
from radagent.report import draft_report as _draft
from radagent.store import RunStore
from radagent.triage import triage as _triage


_SIG_CACHE: dict = {}


def model_signature(cfg: dict) -> str:
    """Content-based id of the segmenter checkpoint (survives copying/zipping, unlike mtimes)."""
    import hashlib

    from radagent.models import checkpoint_path

    p = checkpoint_path(cfg)
    if not p.exists():
        return f"{cfg['segmenter']['backend']}:missing"
    st = p.stat()
    key = (str(p.resolve()), st.st_size, st.st_mtime)
    if key not in _SIG_CACHE:
        h = hashlib.sha1()
        with open(p, "rb") as f:
            h.update(f.read(1 << 20))
            if st.st_size > (2 << 20):
                f.seek(-(1 << 20), 2)
                h.update(f.read(1 << 20))
        _SIG_CACHE[key] = h.hexdigest()[:12]
    return f"{cfg['segmenter']['backend']}:{st.st_size}:{_SIG_CACHE[key]}:{cfg['preprocess']['img_size']}"


@dataclass
class CaseContext:
    case_id: str
    row: dict
    cfg: dict
    store: RunStore
    calibrator: Calibrator
    get_segmenter: Callable[[], object]
    research_mode: bool = True
    reader_policy: object | None = None
    force_recompute: bool = False
    persist: bool = True                  # False for ad-hoc re-runs (UI / CLI demo): do not overwrite reports
    # state
    volume: object | None = None
    seg: object | None = None
    analysis: dict = field(default_factory=dict)
    record: dict = field(default_factory=dict)
    done: dict = field(default_factory=dict)

    def cached(self) -> bool:
        return (not self.force_recompute and self.store.has_analysis(self.case_id)
                and self.analysis.get("model_signature") == model_signature(self.cfg)
                and "measurements" in self.analysis and self.store.viewer_path(self.case_id).exists())


# --------------------------------------------------------------------------- tools
def load_study(ctx: CaseContext) -> dict:
    from radagent.geometry import load_volume

    if ctx.store.has_analysis(ctx.case_id):
        ctx.analysis = ctx.store.load_analysis(ctx.case_id)
    if ctx.cached():
        geo = ctx.analysis["geometry"]
        src = "cache"
    else:
        ctx.analysis = {"case_id": ctx.case_id}
        vol = load_volume(ctx.row["ct_path"])
        ctx.volume = vol
        geo = {"shape_on_disk": list(vol.shape_on_disk), "shape_view": list(vol.view.shape),
               "spacing_mm": [round(s, 5) for s in vol.spacing], "orientation_on_disk": list(vol.orientation_on_disk),
               "canonical": "RAS+", "view": "radiological axial: row 0 anterior, col 0 patient right"}
        ctx.analysis["geometry"] = geo
        src = "NIfTI"
    ctx.record["geometry"] = geo
    sp = geo["spacing_mm"]
    return {"source": src, "shape": geo["shape_on_disk"], "orientation": "".join(geo["orientation_on_disk"]),
            "spacing_mm": f"{sp[2]:.2f} x {sp[1]:.2f} x {sp[0]:.2f}"}


def segment_liver_tumour(ctx: CaseContext) -> dict:
    from radagent.inference import segment_volume

    if ctx.cached():
        s = ctx.analysis["segmentation"]
        ctx.record["model"] = ctx.analysis["model"]
        return {"source": "cache", "found_liver": s["found_liver"], "passes": s["n_passes"],
                "raw_tumour_score": s["tumour_score_raw"]}
    seg_model = ctx.get_segmenter()
    vol = ctx.volume
    ctx.seg = segment_volume(vol.view, vol.spacing, seg_model, ctx.cfg)
    ctx.analysis["segmentation"] = ctx.seg.summary()
    ctx.analysis["model"] = {**seg_model.describe(), "signature": model_signature(ctx.cfg)}
    ctx.record["model"] = ctx.analysis["model"]
    return {"source": "model", "found_liver": ctx.seg.found_liver, "passes": ctx.seg.n_passes,
            "raw_tumour_score": round(ctx.seg.p_raw, 4), "seconds": round(sum(ctx.seg.timings.values()), 2)}


def measure_findings(ctx: CaseContext) -> dict:
    from radagent.geometry import load_volume
    from radagent.measure import measure, reference_metrics, viewer_arrays

    if ctx.cached():
        m = ctx.analysis["measurements"]
        src = "cache"
    else:
        vol = ctx.volume
        res = measure(vol.view, vol.spacing, ctx.seg, ctx.cfg)
        m = res["measurements"]
        gt_view = None
        mask_path = ctx.row.get("mask_path")
        if ctx.research_mode and mask_path and os.path.exists(str(mask_path)):
            gt = load_volume(mask_path)
            if gt.view.shape == vol.view.shape:
                gt_view = gt.view
                ctx.analysis["reference"] = reference_metrics(
                    gt_view, vol.spacing, ctx.seg, res["maps"], ctx.cfg["confidence"]["target_min_tumour_ml"], vol.view)
        ctx.store.save_viewer(ctx.case_id, viewer_arrays(vol.view, ctx.seg, res["maps"], ctx.cfg, gt_view))
        ctx.analysis["measurements"] = m
        ctx.analysis["model_signature"] = model_signature(ctx.cfg)
        ctx.store.save_analysis(ctx.case_id, ctx.analysis)
        src = "computed"
    ctx.record["measurements"] = m
    if "reference" in ctx.analysis:
        ctx.record["reference"] = ctx.analysis["reference"]
    return {"source": src, "liver_ml": m.get("liver_volume_ml"), "tumour_ml": m.get("predicted_tumour_volume_ml"),
            "n_lesions": m.get("n_lesions"), "largest_diameter_mm": m.get("largest_lesion_diameter_mm", 0.0)}


def quality_check(ctx: CaseContext) -> dict:
    q = _qc(ctx.record["geometry"], ctx.record["measurements"], ctx.cfg)
    ctx.record["quality"] = q
    return {"passed": q["passed"], "failed": q["failed_reasons"]}


def estimate_confidence(ctx: CaseContext) -> dict:
    s = ctx.analysis["segmentation"]
    c = case_confidence(s["tumour_score_raw"], s.get("pass_scores_raw", []), ctx.calibrator, ctx.cfg)
    ctx.record["confidence"] = c
    return {"tumour_probability": c["tumour_probability"], "uncertainty": c["uncertainty"],
            "level": c["uncertainty_level"], "calibrator": c["calibrator"]}


def triage_case(ctx: CaseContext) -> dict:
    t = _triage(ctx.record["confidence"], ctx.record["measurements"], ctx.record["quality"], ctx.cfg)
    ctx.record["triage"] = t
    return {"category": t["category"], "priority": t["priority"]}


def apply_reader_policy(ctx: CaseContext) -> dict:
    pol = ctx.reader_policy
    c = ctx.record["confidence"]
    d = pol.decide(c["tumour_probability"], c["uncertainty"], ctx.record["quality"]["passed"])
    ctx.record.setdefault("policy_decisions", {})[pol.reader_id or "universal"] = d
    return {"reader": pol.reader_id, "decision": d["decision"], "threshold": d["threshold"], "reason": d["reason"]}


def draft_report(ctx: CaseContext) -> dict:
    policy = None
    if ctx.reader_policy is not None:
        policy = ctx.record.get("policy_decisions", {}).get(ctx.reader_policy.reader_id or "universal")
    text = _draft(ctx.record, review=None, policy=policy)
    ctx.record["report_text"] = text
    if not ctx.persist:
        return {"lines": text.count("\n") + 1, "path": None}
    path = ctx.store.report_path(ctx.case_id)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text)
    ctx.record["report_path"] = str(path)
    return {"lines": text.count("\n") + 1, "path": str(path)}


def finish(ctx: CaseContext) -> dict:
    return {"status": "first read complete"}


@dataclass
class Tool:
    name: str
    fn: Callable
    description: str
    requires: tuple = ()


TOOLS: dict[str, Tool] = {t.name: t for t in [
    Tool("load_study", load_study, "Load the CT study and its header geometry (orientation, voxel spacing)."),
    Tool("segment_liver_tumour", segment_liver_tumour,
         "Run the segmentation tool (liver + tumour) with stochastic passes for uncertainty.", ("load_study",)),
    Tool("measure_findings", measure_findings,
         "Measure liver and lesion volumes/diameters at original resolution from the voxel spacing.",
         ("segment_liver_tumour",)),
    Tool("quality_check", quality_check, "Run automatic quality-control checks on the first read.", ("measure_findings",)),
    Tool("estimate_confidence", estimate_confidence,
         "Turn the raw tumour score into a calibrated probability and an uncertainty estimate.",
         ("segment_liver_tumour",)),
    Tool("triage_case", triage_case, "Assign a worklist priority and triage category.",
         ("estimate_confidence", "quality_check")),
    Tool("apply_reader_policy", apply_reader_policy,
         "Apply the current reader's learned policy (suggestion threshold / deferral).",
         ("estimate_confidence", "quality_check")),
    Tool("draft_report", draft_report, "Fill the deterministic structured report template.", ("triage_case",)),
    Tool("finish", finish, "End the first read.", ("draft_report",)),
]}


def requirements(name: str, ctx: CaseContext) -> list[str]:
    req = list(TOOLS[name].requires)
    if name == "draft_report" and ctx.reader_policy is not None:
        req.append("apply_reader_policy")
    return req


def ready_tools(ctx: CaseContext) -> list[str]:
    out = []
    for name in TOOLS:
        if name in ctx.done:
            continue
        if name == "apply_reader_policy" and ctx.reader_policy is None:
            continue
        if all(r in ctx.done for r in requirements(name, ctx)):
            out.append(name)
    return out
