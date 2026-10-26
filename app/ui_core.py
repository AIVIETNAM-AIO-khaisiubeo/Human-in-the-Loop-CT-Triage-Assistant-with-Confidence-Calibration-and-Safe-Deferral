"""UI logic without Streamlit (rendering, worklist, live policy adaptation) - unit-testable."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from radagent.confidence import uncertainty_level  # noqa: E402
from radagent.config import load_config  # noqa: E402
from radagent.policy import ReaderPolicy, case_table, fit_policy, universal_policy  # noqa: E402
from radagent.readers.feedback import FeedbackEvent, FeedbackStore  # noqa: E402
from radagent.readers.profiles import N_DRAWS, ReaderProfile, load_profiles, simulate_read  # noqa: E402
from radagent.report import suggestion_text  # noqa: E402
from radagent.store import RunStore, load_json  # noqa: E402
from radagent.triage import CATEGORY_ORDER, triage  # noqa: E402
from ui_theme import CATEGORY_STYLE, OVERLAY  # noqa: E402

LIVE_READER = "live"
POLICY_CHOICES = {
    "universal": "Universal threshold",
    "universal_defer": "Universal + deferral",
    "reader_threshold": "Reader-specific threshold",
    "reader_threshold_defer": "Reader-specific + deferral",
}
# text glyphs for places that cannot render HTML (tables, downloads); shape-coded like the badges
CATEGORY_ICON = {"QC FAIL": "■", "URGENT": "▲", "SUSPICIOUS": "◆", "UNCERTAIN": "◐", "LIKELY NORMAL": "○",
                 "DEFERRED": "●"}
CATEGORY_LABEL = {k: v[1] for k, v in CATEGORY_STYLE.items()}

# overlay colours (RGB) - see ui_theme.OVERLAY
C_LIVER = np.array(OVERLAY["liver"])
C_LESION = np.array(OVERLAY["lesion"])
C_SELECTED = np.array(OVERLAY["selected"])
C_REF = np.array(OVERLAY["reference"])
C_UNC = np.array(OVERLAY["uncertainty"])


# --------------------------------------------------------------------------- run loading
def find_runs(root: str | Path = "outputs") -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if (p / "cases").is_dir() and any((p / "cases").glob("*.json"))],
                  key=lambda p: -p.stat().st_mtime)


def load_run_config(run: Path) -> dict:
    f = run / "config_used.yaml"
    cfg = load_config(f if f.exists() else None)
    cfg["output_root"] = str(run.parent)
    cfg["run_name"] = run.name
    return cfg


def load_run(run: Path) -> dict:
    st = RunStore(run)
    records = st.records()
    out = {"store": st, "cfg": load_run_config(run), "records": {r["case_id"]: r for r in records},
           "cases": case_table(records)}
    for key, rel in (("calibration", ("calibration", "calibration_metrics.json")), ("model_card", ("model", "model_card.json")),
                     ("exp_meta", ("experiments", "meta.json"))):
        f = st.p(*rel)
        out[key] = load_json(f) if f.exists() else None
    for key in ("summary", "benefit_spread", "results", "learning_curve", "policies"):
        f = st.p("experiments", f"{key}.csv")
        out[key] = pd.read_csv(f) if f.exists() else None
    return out


# --------------------------------------------------------------------------- rendering
def _edges(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    return m & ~ndi.binary_erosion(m, iterations=1, border_value=0)


def _blend(img: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float) -> None:
    if mask.any():
        img[mask] = (1 - alpha) * img[mask] + alpha * color


def render_slice(arrs: dict, i: int, show_liver: bool = True, show_lesions: bool = True, kept_lesions=None,
                 selected_lesion: int | None = None, show_uncertainty: bool = False, show_reference: bool = False,
                 upscale: int = 4, contrast: tuple[int, int] = (0, 255)) -> np.ndarray:
    """Compose an RGB uint8 image of viewer slice ``i`` with the requested overlays."""
    lo, hi = contrast
    ct = arrs["ct"][i].astype(np.float32)
    ct = np.clip((ct - lo) / max(1, hi - lo) * 255.0, 0, 255)
    img = np.repeat(ct[..., None], 3, axis=2)
    if show_uncertainty:
        h = arrs["entropy"][i].astype(np.float32) / 255.0
        a = np.clip(h * 0.85, 0, 0.85)[..., None]
        img = (1 - a) * img + a * C_UNC
    lesions = arrs["lesions"][i]
    if show_lesions and lesions.any():
        ids = np.unique(lesions[lesions > 0])
        if kept_lesions is not None:
            ids = [k for k in ids if int(k) in set(kept_lesions)]
        for k in ids:
            m = lesions == k
            col = C_SELECTED if selected_lesion is not None and int(k) == selected_lesion else C_LESION
            _blend(img, m, col, 0.35)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if upscale > 1:
        img = np.repeat(np.repeat(img, upscale, axis=0), upscale, axis=1)

    def up(m):
        return np.repeat(np.repeat(m, upscale, axis=0), upscale, axis=1) if upscale > 1 else m

    if show_liver:
        img[_edges(up(arrs["liver"][i] > 0))] = C_LIVER
    if show_lesions and lesions.any():
        ids = np.unique(lesions[lesions > 0])
        if kept_lesions is not None:
            ids = [k for k in ids if int(k) in set(kept_lesions)]
        for k in ids:
            col = C_SELECTED if selected_lesion is not None and int(k) == selected_lesion else C_LESION
            img[_edges(up(lesions == k))] = col
    if show_reference and "reference" in arrs:
        ref = arrs["reference"][i]
        img[_edges(up(ref == 2))] = C_REF
        img[_edges(up(ref > 0)) & (np.indices(img.shape[:2]).sum(0) % 6 < 3)] = C_REF   # dotted liver outline
    return img


def key_slice_index(arrs: dict, record: dict) -> int:
    z = arrs["z"]
    les = record.get("measurements", {}).get("lesions", [])
    if les:
        target = les[0]["key_slice"]
        return int(np.argmin(np.abs(z - target)))
    area = (arrs["liver"] > 0).sum(axis=(1, 2))
    return int(np.argmax(area)) if area.any() else len(z) // 2


# --------------------------------------------------------------------------- policies
def all_events(store: RunStore, reader_ids: list[str]) -> dict[str, list[dict]]:
    fs = FeedbackStore(store.p("feedback"))
    return {r: fs.events(r) for r in reader_ids}


def current_policy(policy_name: str, reader_id: str, events_by_reader: dict[str, list[dict]], cases: dict,
                   cfg: dict) -> ReaderPolicy:
    if policy_name == "universal":
        return universal_policy(cfg)
    if policy_name == "universal_defer":
        pooled = [e for evs in events_by_reader.values() for e in evs]
        return fit_policy(pooled, cases, cfg, None, "universal_defer")
    return fit_policy(events_by_reader.get(reader_id, []), cases, cfg, reader_id, policy_name)


def policy_trajectory(events: list[dict], cases: dict, cfg: dict, variant: str = "reader_threshold_defer") -> pd.DataFrame:
    rows = []
    for n in range(0, len(events) + 1):
        p = fit_policy(events[:n], cases, cfg, None, variant)
        rows.append({"n_feedback": n, "threshold": p.threshold,
                     "defer_uncertainty": np.nan if p.defer_uncertainty is None else p.defer_uncertainty,
                     "personalised": p.personalised, "trust_alpha": p.trust_alpha, "trust_beta": p.trust_beta})
    return pd.DataFrame(rows)


def worklist(records: dict, cases: dict, policy: ReaderPolicy, cfg: dict, reviews: dict, splits=("test",)) -> pd.DataFrame:
    rows = []
    for cid, r in records.items():
        if r.get("split") not in splits:
            continue
        c = r["confidence"]
        d = policy.decide(c["tumour_probability"], c["uncertainty"], r["quality"]["passed"])
        t = triage(c, r["measurements"], r["quality"], cfg, threshold=policy.threshold)
        cat = t["category"]
        if d["decision"] == "defer" and cat != "QC FAIL":
            shown_cat = "DEFERRED"
            suggestion = "Withheld - independent read"
        elif d["decision"] == "defer":
            shown_cat = cat
            suggestion = "Manual read (QC failed)"
        else:
            shown_cat = cat
            suggestion = suggestion_text(c["tumour_probability"], policy.threshold)
        rv = reviews.get(cid)
        vp = c.get("vlm_probability")
        reasons = list(t.get("reasons", []))
        if shown_cat == "DEFERRED":
            reasons = [f"AI suggestion withheld for this reader ({d.get('reason', 'reader policy')})"] + reasons
        reasons = [humanize(x) for x in reasons]
        rows.append({"case_id": cid, "priority": t["priority"], "category": shown_cat,
                     "medgemma": float(vp) if vp is not None else float("nan"),
                     "agree": ("disagree" if c.get("tools_disagree") else "agree") if vp is not None else "-",
                     "reasons": "; ".join(str(x) for x in reasons),
                     "unc_level": uncertainty_level(float(c["uncertainty"])),
                     "triage": f"{CATEGORY_ICON.get(shown_cat, '')} {CATEGORY_LABEL.get(shown_cat, shown_cat)}",
                     "ai_suggestion": suggestion,
                     "tumour_probability": c["tumour_probability"], "uncertainty": c["uncertainty"],
                     "tumour_ml": r["measurements"].get("predicted_tumour_volume_ml", 0.0),
                     "lesions": r["measurements"].get("n_lesions", 0),
                     "universal_category": r["triage"]["category"],
                     "status": "pending" if rv is None else rv["action"]})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["priority", "case_id"], ascending=[False, True]).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


# --------------------------------------------------------------------------- simulated / live reads
def _draws(reader_id: str, case_id: str, salt: int = 0) -> np.ndarray:
    h = int(hashlib.sha256(f"{reader_id}|{case_id}|{salt}".encode()).hexdigest()[:12], 16)
    return np.random.default_rng(h).random(N_DRAWS)


def simulate_and_log(store: RunStore, profile: ReaderProfile, case: dict, policy: ReaderPolicy, salt: int = 0) -> dict:
    d = policy.decide(case["p"], case["u"], case["qc_passed"])
    r = simulate_read(profile, case, d, _draws(profile.reader_id, case["case_id"], salt))
    ev = FeedbackEvent(reader_id=profile.reader_id, case_id=case["case_id"], policy=policy.name, shown=r["shown"],
                       ai_label=d["ai_label"], ai_probability=case["p"], ai_confidence=d["confidence"],
                       uncertainty=case["u"], threshold=policy.threshold, initial_read=r["initial_read"],
                       action=r["action"], final_label=r["final_label"], source="simulated-ui").to_dict()
    FeedbackStore(store.p("feedback")).append(ev)
    return {**ev, "decision": d, "gt_present": case["gt_present"], "switched": r["switched"]}


def log_live(store: RunStore, case: dict, policy: ReaderPolicy, initial_read: int | None, action: str,
             final_label: int, edit: dict | None = None) -> dict:
    d = policy.decide(case["p"], case["u"], case["qc_passed"])
    ev = FeedbackEvent(reader_id=LIVE_READER, case_id=case["case_id"], policy=policy.name,
                       shown=d["decision"] == "show", ai_label=d["ai_label"], ai_probability=case["p"],
                       ai_confidence=d["confidence"], uncertainty=case["u"], threshold=policy.threshold,
                       initial_read=initial_read, action=action, final_label=int(final_label), source="live",
                       edit=edit).to_dict()
    FeedbackStore(store.p("feedback")).append(ev)
    return ev


def reader_profiles(cfg: dict) -> dict:
    return load_profiles(cfg)


def outcome_label(ev: dict, gt_present: bool) -> str:
    ok = int(ev["final_label"]) == int(gt_present)
    ai_ok = ev.get("ai_label") is not None and int(ev["ai_label"]) == int(gt_present)
    if not ev["shown"]:
        return "correct (independent read)" if ok else "incorrect (independent read)"
    if ev["action"] in ("accept", "edit"):
        return "correct acceptance" if ai_ok else "HARMFUL acceptance (AI was wrong)"
    return "correct rejection (AI was wrong)" if not ai_ok else "wrong rejection (AI was right)"


def humanize(reason: str) -> str:
    """Make a triage/policy reason readable: symbols and plain words instead of code-like notation."""
    r = str(reason).replace(">=", "≥").replace("<=", "≤")
    r = r.replace("p = ", "probability ").replace("p ", "probability ", 1) if r.startswith("p ") else \
        r.replace("p = ", "probability ")
    return r[:1].upper() + r[1:]


def neighbours(order: list[str], case_id: str) -> tuple[str | None, str | None]:
    """Previous / next case in worklist order."""
    if case_id not in order:
        return None, (order[0] if order else None)
    i = order.index(case_id)
    return (order[i - 1] if i > 0 else None), (order[i + 1] if i + 1 < len(order) else None)


__all__ = ["CATEGORY_ORDER", "LIVE_READER", "POLICY_CHOICES", "find_runs", "load_run", "render_slice", "worklist",
           "current_policy", "policy_trajectory", "simulate_and_log", "log_live", "key_slice_index", "all_events"]
