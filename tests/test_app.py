"""Tests for the reading interface (app/): wording helpers, HTML components and the full Streamlit app.

The app tests drive the real Streamlit script with ``streamlit.testing.v1.AppTest`` against a small
run built from LiTS-layout test CTs with stand-in models (no GPU, no download). They check the
safety rules of the design review as behaviour, not just looks:

* blind first read: no AI output (prediction, %, disagreement, AI priority) for unread cases, arrival order;
* the AI evidence opens only after a saved impression, and is shown next to it;
* every decision is stored with the recommendation on screen, session, policy and model-set versions;
* the decision policy is frozen for the session and changes only by an explicit, logged action;
* QC failures withhold the AI output and are explained; "defer" records no final finding.

Run:  pytest -q tests/test_app.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import ui_core as U  # noqa: E402
import ui_theme as T  # noqa: E402


# --------------------------------------------------------------------------- wording and components
def _ratio(a, b):
    def lum(h):
        r, g, b_ = (int(h.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4))
        f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4  # noqa: E731
        return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b_)

    hi, lo = sorted([lum(a), lum(b)], reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_semantic_palette_meets_wcag_aa():
    pairs = [(T.NAVY, "#FFFFFF"), (T.INK, "#FFFFFF"), (T.SLATE, "#FFFFFF"), (T.SLATE, T.SURFACE),
             (T.SLATE, T.SLATE_BG), ("#FFFFFF", T.TEAL), (T.TEAL_DARK, T.TEAL_TINT), (T.RED, T.RED_BG),
             (T.AMBER, T.AMBER_BG), (T.AMBER, "#FFFFFF"), ("#7A2E0E", T.AMBER_BG), ("#DCE6F2", T.NAVY),
             ("#FDE7B0", T.NAVY), ("#D7E1EC", T.VIEWER_BG), ("#E6EDF5", "#132540")]
    for fg, bg in pairs:
        assert _ratio(fg, bg) >= 4.5, (fg, bg, round(_ratio(fg, bg), 2))


def test_overlays_visible_on_ct_and_distinct():
    rgb = {k: np.array(v, float) for k, v in T.OVERLAY.items() if k in ("liver", "lesion", "uncertainty")}
    hexes = {k: "#%02X%02X%02X" % tuple(int(x) for x in v) for k, v in rgb.items()}
    for k, h in hexes.items():
        assert _ratio(h, "#000000") >= 7.0, k                     # visible on the dark CT background / air
    keys = list(rgb)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert np.linalg.norm(rgb[keys[i]] - rgb[keys[j]]) > 120, (keys[i], keys[j])


def test_badges_codes_and_escaping():
    shapes = {v[2] for v in T.CATEGORY_STYLE.values() if v[1] != "AI withheld (QC)"}
    assert len(shapes) == len(T.CATEGORY_STYLE) - 1              # distinct shapes (withheld kinds share a family)
    for cat, (_, label, _) in T.CATEGORY_STYLE.items():
        h = T.badge(cat)
        assert label in h and "<svg" in h and 'aria-hidden="true"' in h
    assert "AI:" in T.badge("SUSPICIOUS")                        # AI outputs are labelled as such
    assert "&lt;b&gt;" in T.header("<b>", "S-1", "M-1", None)
    assert "&lt;script&gt;" in T.withheld_card("<script>", "") and T.esc("a & b") == "a &amp; b"
    assert "&lt;i&gt;" in T.comparison(1, "<i>", 0, "x")


def test_stepper_meter_measures():
    s = T.stepper(3, {1, 2})
    assert s.count("ra-step-done") == 2 and 'aria-current="step"' in s and "Reconcile and decide" in s
    assert "width:100.0%" in T.meter(1.7, 0.5) and "threshold 50%" in T.meter(0.2, 0.5)
    assert T.measures([("Liver volume", "1500", "mL")]).count("<dt>") == 1


def test_clinical_wording_helpers():
    assert U.humanize("p = 0.72 >= suggestion threshold 0.50") == "Probability 0.72 ≥ suggestion threshold 0.50"
    assert U.neighbours(["a", "b", "c"], "b") == ("a", "c") and U.neighbours(["a"], "z") == (None, "a")
    assert U.pct(0.734) == "73%"
    a = U.agreement(0.73, 0.91)
    assert a["kind"] == "differ" and "73% vs 91%" in a["text"]           # same class, different scores
    assert U.agreement(0.72, 0.29)["kind"] == "disagree"
    assert U.agreement(0.70, 0.78)["kind"] == "agree"
    assert U.agreement(0.70, None)["kind"] == "none"
    assert "0.84 on a 0-1 scale" in U.uncertainty_phrase(0.84)
    rec = {"quality": {"passed": False, "checks": [{"name": "tumour_inside_liver", "passed": False,
                                                    "message": "43% (167 ml) of tumour-class voxels fall outside"}]}}
    q = U.qc_status(rec)
    assert q["label"] == "Segmentation QC failed" and "withheld" in q["consequence"] and "43%" in q["detail"]
    rec["quality"]["checks"].append({"name": "enough_slices", "passed": False, "message": "12 axial slices"})
    assert U.qc_status(rec)["label"] == "Image QC failed"
    assert U.qc_status({"quality": {"passed": True}})["kind"] == "ok"


def test_viewer_rendering_window_zoom_markers():
    lo, hi = U.window_to_u8(350, 75, (-100, 250))
    assert abs(lo) < 1e-6 and abs(hi - 255) < 1e-6
    arrs = {"ct": np.full((3, 64, 64), 100, np.uint8), "liver": np.zeros((3, 64, 64), np.uint8),
            "lesions": np.zeros((3, 64, 64), np.uint8), "entropy": np.zeros((3, 64, 64), np.uint8), "z": np.arange(3)}
    arrs["lesions"][1, 20:30, 20:30] = 1
    plain = U.render_slice(arrs, 1, upscale=2)
    zoomed = U.render_slice(arrs, 1, upscale=2, zoom=2.0, markers=True)
    assert plain.shape == zoomed.shape == (128, 128, 3) and not np.array_equal(plain, zoomed)
    no_fill = U.render_slice(arrs, 1, upscale=1, lesion_alpha=0.0, show_liver=False)
    filled = U.render_slice(arrs, 1, upscale=1, lesion_alpha=0.5, show_liver=False)
    assert tuple(no_fill[25, 25]) == tuple(no_fill[0, 0]) and tuple(filled[25, 25]) != tuple(filled[0, 0])
    assert U.neutral_slice_index(arrs) == 1


def test_blind_order_ignores_ai_priority():
    import pandas as pd

    wl = pd.DataFrame({"case_id": ["lits_009", "lits_002", "lits_005"], "rank": [1, 2, 3], "arrival": [9, 2, 5]})
    assert list(U.order_worklist(wl, blind=True).case_id) == ["lits_002", "lits_005", "lits_009"]
    assert list(U.order_worklist(wl, blind=False).case_id) == ["lits_009", "lits_002", "lits_005"]


# --------------------------------------------------------------------------- the Streamlit app
st_testing = pytest.importorskip("streamlit.testing.v1")


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    from radagent import pipeline as P
    from tests.test_lits_medgemma import _e2e_cfg

    cfg = _e2e_cfg(tmp_path_factory.mktemp("apprun"))
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        for step in (P.step_data, P.step_train, P.step_infer, P.step_vlm, P.step_calibrate):
            step(cfg)
        P.step_first_read(cfg, "rule")
        P.step_simulate(cfg, 2)
        P.step_figures(cfg)
    finally:
        P.release_models()
        os.chdir(cwd)
    return Path(cfg["output_root"]) / cfg["run_name"]


@pytest.fixture
def app(run_dir, monkeypatch):
    monkeypatch.setenv("RA_RUN", str(run_dir))
    monkeypatch.chdir(ROOT)
    d = run_dir / "feedback"
    if d.exists():
        for f in d.glob("*"):
            f.unlink()
    at = st_testing.AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def _md(at) -> str:
    return "\n".join(m.value for m in at.markdown)


def _records(run_dir: Path) -> dict:
    out = {}
    for f in (run_dir / "cases").glob("*.json"):
        r = json.loads(f.read_text())
        out[r["case_id"]] = r
    return out


def _events(run_dir: Path, reader: str = "live") -> list:
    from radagent.readers.feedback import FeedbackStore

    return FeedbackStore(run_dir / "feedback").events(reader)


def _order(at) -> list:
    return [b.key[5:] for b in at.button if b.key and b.key.startswith("open_")]


def test_blind_worklist_shows_no_ai_output(app, run_dir):
    assert app.session_state["view"] == "worklist" and app.session_state["reading_mode"] == "blind"
    md = _md(app)
    assert "Hidden until your impression" in md and "AI triage hidden" in md
    status = next(m.value for m in app.markdown if "ra-statusline" in m.value)
    assert "AI tools disagree" not in status and "Urgent" not in status
    ai_cells = [m.value for m in app.markdown if '<span class="ra-cell-l">AI</span>' in m.value]
    assert ai_cells and all("Hidden until your impression" in v and "%" not in v for v in ai_cells)
    order = _order(app)
    assert order == sorted(order, key=lambda c: int(c.split("_")[1]))      # arrival order, not AI priority
    assert sum(app.button(key=f"open_{c}").proto.type == "primary" for c in order) == 1


def test_blind_journey_impression_reveal_decide(app, run_dir):
    recs = _records(run_dir)
    cid = next(c for c in _order(app) if recs[c]["quality"]["passed"])
    app.button(key=f"open_{cid}").click().run()
    assert app.session_state["view"] == "read" and app.session_state["case"] == cid
    md = _md(app)
    assert "Independent read" in md and 'class="ra-ai"' not in md and '<span class="ra-pn">2</span>AI evidence' not in md
    assert all(t.disabled for t in app.toggle if t.key in ("ov_liver", "ov_lesions", "ov_unc"))
    app.segmented_control(key=f"imp_{cid}").set_value("present").run()
    app.button(key="record_imp").click().run()
    md = _md(app)
    assert not app.exception and 'class="ra-ai"' in md and "Your impression" in md and "Patient-level probability" in md
    app.segmented_control(key=f"final_{cid}").set_value("present").run()
    app.button(key="save_decision").click().run()
    ev = _events(run_dir)[-1]
    assert ev["case_id"] == cid and ev["initial_read"] == 1 and ev["final_label"] == 1
    assert ev["action"] in ("accept", "reject") and ev["recommendation"]["text"]
    assert ev["session_id"].startswith("S-") and ev["policy_version"].startswith("P-") \
        and ev["model_version"].startswith("M-") and ev["impression_at"]
    assert "Decision saved" in _md(app)
    app.button(key="after_report").click().run()
    assert app.session_state["view"] == "report" and any("AI recommendation shown" in c.value for c in app.code)


def test_qc_failure_and_defer(app, run_dir):
    recs = _records(run_dir)
    order = _order(app)
    qc_fail = [c for c in order if not recs[c]["quality"]["passed"]]
    if qc_fail:
        app.button(key=f"open_{qc_fail[0]}").click().run()
        md = _md(app)
        assert "AI output withheld" in md and "What it means" in md and "independent read" in md
        app.segmented_control(key=f"final_{qc_fail[0]}").set_value("absent").run()
        app.button(key="save_decision").click().run()
        assert _events(run_dir)[-1]["action"] == "manual"
    ok = [c for c in order if recs[c]["quality"]["passed"]]
    app.session_state["case"], app.session_state["view"] = ok[-1], "read"
    app.run()
    app.segmented_control(key=f"imp_{ok[-1]}").set_value("absent").run()
    app.button(key="record_imp").click().run()
    app.segmented_control(key=f"final_{ok[-1]}").set_value("defer").run()
    app.button(key="save_decision").click().run()
    ev = _events(run_dir)[-1]
    assert ev["action"] == "defer" and ev["final_label"] is None
    app.segmented_control(key="nav").set_value("worklist").run()
    assert "Deferred · second opinion" in _md(app)


def test_report_and_audit_stay_closed_before_impression(app, run_dir):
    cid = _order(app)[0]
    app.session_state["case"], app.session_state["view"] = cid, "report"
    app.run()
    assert "available after you save your independent impression" in _md(app) and not app.code
    app.button(key="go_audit").click().run()
    assert app.session_state["view"] == "audit" and "opens after your impression" in _md(app)


def test_policy_frozen_until_explicitly_applied(app, run_dir):
    app.selectbox(key="reader").set_value("reader_b").run()
    key = next(k for k in app.session_state["active_policies"] if k.startswith("reader_b|"))
    before = U.policy_version(app.session_state["active_policies"][key]["policy"])
    app.button(key="go_adapt").click().run()
    app.number_input(key="nmore").set_value(6).run()
    app.button(key="simn").click().run()
    n_sim = len(_events(run_dir, "reader_b"))
    assert not app.exception and 1 <= n_sim <= 6
    assert U.policy_version(app.session_state["active_policies"][key]["policy"]) == before   # no silent change
    btn = app.button(key="apply_policy")
    if not btn.disabled:
        btn.click().run()
        log = (run_dir / "feedback" / "policy_changes.jsonl").read_text().splitlines()
        assert json.loads(log[-1])["from"] == before
    app.button(key="reset_confirm").click().run()
    assert _events(run_dir, "reader_b") == []


def test_assisted_mode_and_every_view(app):
    app.radio(key="reading_mode").set_value("assisted").run()
    md = _md(app)
    assert "AI-assisted triage" in md and "AI tools disagree" in md and "%" in md
    app.selectbox(key="wl_sort").set_value("arrival").run()
    assert not app.exception
    for reader in ("live", "reader_a"):
        app.selectbox(key="reader").set_value(reader).run()
        for view in ("worklist", "read"):
            app.segmented_control(key="nav").set_value(view).run()
            assert not app.exception, (reader, view, app.exception)
        for v in ("adapt", "perf", "audit", "about"):
            app.button(key=f"go_{v}").click().run()
            assert not app.exception and app.session_state["view"] == v, (reader, v, app.exception)
            assert app.session_state["nav"] is None                     # research area is outside the two modes
        app.button(key="back_wl").click().run()
        assert app.session_state["view"] == "worklist"
    app.segmented_control(key="nav").set_value(None).run()             # deselecting keeps the page
    assert app.session_state["view"] == "worklist" and not app.exception


def test_simulated_reader_in_read_view(app, run_dir):
    app.selectbox(key="reader").set_value("reader_c").run()
    app.segmented_control(key="nav").set_value("read").run()
    cid = app.session_state["case"]
    app.button(key=f"sim_{cid}").click().run()
    ev = _events(run_dir, "reader_c")
    assert not app.exception and len(ev) == 1 and ev[0]["session_id"].startswith("S-")


def test_worklist_search_filter_and_viewer_controls(app):
    app.text_input(key="wl_search").input("lits_0").run()
    assert not app.exception
    app.segmented_control(key="wl_filter").set_value("decided").run()
    assert not app.exception
    app.segmented_control(key="wl_filter").set_value("all").run()
    app.toggle(key="wl_details").set_value(True).run()
    cid = _order(app)[0]
    app.button(key=f"open_{cid}").click().run()
    s0 = app.session_state[f"slice_{cid}"]
    app.button(key="slice_next").click().run()
    assert app.session_state[f"slice_{cid}"] != s0 or not app.exception
    app.segmented_control(key="window").set_value("custom").run()
    app.button(key="reset_view").click().run()
    assert not app.exception and app.session_state["window"] == "liver"


def test_onboarding_without_runs(tmp_path, monkeypatch):
    if U.find_runs(ROOT / "outputs"):
        pytest.skip("this checkout has runs in outputs/")
    monkeypatch.delenv("RA_RUN", raising=False)
    monkeypatch.chdir(tmp_path)
    at = st_testing.AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=60)
    at.run()
    assert not at.exception and "No reading session yet" in _md(at)
