"""Tests for the reading interface (app/): HTML building blocks and the full Streamlit app.

The app tests drive the real Streamlit script with ``streamlit.testing.v1.AppTest`` against a small
run built from LiTS-layout test CTs with stand-in models (no GPU, no download). They walk the
radiologist journey: worklist -> open case -> independent impression -> AI suggestion -> decision ->
next case / report, plus every other view, the simulated readers and the compact worklist.

Run:  pytest -q tests/test_app.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import ui_core as U  # noqa: E402
import ui_theme as T  # noqa: E402


# --------------------------------------------------------------------------- HTML building blocks
def test_badges_are_shape_word_and_colour_coded():
    seen = set()
    for cat, (mod, label, ico) in T.CATEGORY_STYLE.items():
        h = T.badge(cat)
        assert f"ra-{mod}" in h and label in h and "<svg" in h and 'aria-hidden="true"' in h
        seen.add(ico)
    assert len(seen) == len(T.CATEGORY_STYLE)          # every category has its own shape


def test_dynamic_text_is_escaped():
    h = T.case_card_main("<script>x</script>", "a & b", "p < q")
    assert "<script>" not in h and "&lt;script&gt;" in h and "a &amp; b" in h
    assert "&lt;b&gt;" in T.header("<b>", "pol", "research", "run")
    assert "<b>" not in T.withheld_card("<b>", "x")


def test_stepper_meter_and_flow():
    s = T.stepper(3, {1, 2})
    assert s.count("ra-step-done") == 2 and 'aria-current="step"' in s and "ra-step-todo" in s
    m = T.meter(1.7, 0.5)
    assert "width:100.0%" in m and "left:50.0%" in m and "threshold 0.50" in m
    assert "width:0.0%" in T.meter(float("nan"))
    f = T.flow([("A", "a", "data"), ("B", "b", "human")])
    assert f.count("<li") == 2 and "ra-flow-human" in f


def test_palette_meets_wcag_aa():
    def lum(h):
        r, g, b = (int(h.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4))
        f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4  # noqa: E731
        return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)

    def ratio(a, b):
        hi, lo = sorted([lum(a), lum(b)], reverse=True)
        return (hi + 0.05) / (lo + 0.05)

    pairs = [(T.NAVY, "#FFFFFF"), (T.INK, "#FFFFFF"), (T.MUTED, "#FFFFFF"), (T.MUTED, T.SURFACE),
             ("#FFFFFF", T.TEAL), (T.TEAL_DARK, T.TEAL_TINT), (T.TEAL_SOFT, T.NAVY),
             ("#B42318", "#FEF3F2"), ("#B54708", "#FFF6E5"), ("#5B3CC4", "#F3F1FF"), ("#067647", "#ECF8F1"),
             ("#1D4E89", "#EAF1FA"), ("#344054", "#EEF1F4")]
    for fg, bg in pairs:
        assert ratio(fg, bg) >= 4.5, (fg, bg, ratio(fg, bg))


def test_worklist_helpers():
    assert U.humanize("p = 0.72 >= suggestion threshold 0.50") == "Probability 0.72 ≥ suggestion threshold 0.50"
    assert U.neighbours(["a", "b", "c"], "b") == ("a", "c")
    assert U.neighbours(["a", "b", "c"], "a") == (None, "b")
    assert U.neighbours(["a"], "z") == (None, "a")


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
    for sub in ("feedback",):
        d = run_dir / sub
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
    return {json.loads(f.read_text())["case_id"]: json.loads(f.read_text()) for f in (run_dir / "cases").glob("*.json")}


def _events(run_dir: Path, reader: str = "live") -> list:
    from radagent.readers.feedback import FeedbackStore

    return FeedbackStore(run_dir / "feedback").events(reader)


def test_worklist_first(app):
    assert app.session_state["view"] == "worklist"
    md = _md(app)
    assert "ReaderAdapt · Liver CT" in md and "Research prototype" in md and "To read" in md
    opens = [b for b in app.button if b.key and b.key.startswith("open_")]
    assert opens and sum(b.proto.type == "primary" for b in opens) == 1     # one clear next action


def test_read_journey_impression_ai_decision_next(app, run_dir):
    recs = _records(run_dir)
    order = [b.key[5:] for b in app.button if b.key and b.key.startswith("open_")]
    cid = next(c for c in order if recs[c]["quality"]["passed"])
    app.button(key=f"open_{cid}").click().run()
    assert app.session_state["view"] == "read" and app.session_state["case"] == cid
    assert "Your independent impression" in _md(app) and 'class="ra-ai ' not in _md(app)
    app.segmented_control(key=f"imp_{cid}").set_value("present").run()
    app.button(key="record_imp").click().run()
    assert not app.exception and 'class="ra-ai ' in _md(app)            # AI suggestion revealed
    app.button(key="accept").click().run()
    ev = _events(run_dir)
    assert len(ev) == 1 and ev[0]["case_id"] == cid and ev[0]["action"] == "accept" and ev[0]["initial_read"] == 1
    assert "Decision saved" in _md(app)
    nxt = app.button(key="after_next")
    if not nxt.disabled:
        nxt.click().run()
        assert app.session_state["case"] != cid and app.session_state["view"] == "read"
    app.segmented_control(key="nav").set_value("report").run()
    assert not app.exception and any("RESEARCH PREDICTION" in c.value for c in app.code)


def test_independent_read_and_edit_paths(app, run_dir):
    recs = _records(run_dir)
    order = [b.key[5:] for b in app.button if b.key and b.key.startswith("open_")]
    qc_fail = [c for c in order if not recs[c]["quality"]["passed"]]
    if qc_fail:                                     # AI suggestion withheld -> independent read
        app.button(key=f"open_{qc_fail[0]}").click().run()
        assert "AI suggestion withheld" in _md(app)
        app.segmented_control(key=f"fin_{qc_fail[0]}").set_value("absent").run()
        app.button(key="save_manual").click().run()
        assert _events(run_dir)[-1]["action"] == "manual"
    ok = [c for c in order if recs[c]["quality"]["passed"]]
    app.session_state["case"], app.session_state["view"] = ok[-1], "read"
    app.toggle(key="sequential").set_value(False).run()
    app.segmented_control(key=f"pres_{ok[-1]}").set_value("no").run()
    app.button(key="save_edit").click().run()
    assert _events(run_dir)[-1]["case_id"] == ok[-1] and _events(run_dir)[-1]["action"] in ("edit", "reject")


def test_every_view_renders_for_every_reader(app):
    for reader in ("live", "reader_a"):
        app.selectbox(key="reader").set_value(reader).run()
        for view in ("worklist", "read", "report", "adapt", "perf", "audit", "about"):
            app.segmented_control(key="nav").set_value(view).run()
            assert not app.exception, (reader, view, app.exception)
            assert app.session_state["view"] == view
    app.segmented_control(key="nav").set_value(None).run()             # deselecting keeps the page
    assert app.session_state["view"] == "about" and not app.exception


def test_simulated_reader_and_adaptation(app, run_dir):
    app.selectbox(key="reader").set_value("reader_b").run()
    app.segmented_control(key="nav").set_value("read").run()
    cid = app.session_state["case"]
    app.button(key=f"sim_{cid}").click().run()
    assert not app.exception and len(_events(run_dir, "reader_b")) == 1
    app.number_input(key="nmore").set_value(3).run()
    app.button(key="simn").click().run()
    assert not app.exception and len(_events(run_dir, "reader_b")) >= 2
    app.segmented_control(key="nav").set_value("adapt").run()
    assert not app.exception and "How the assistant adapts" in _md(app)
    app.button(key="reset_confirm").click().run()
    assert _events(run_dir, "reader_b") == []


def test_compact_worklist_and_filters(app):
    app.segmented_control(key="wl_filter").set_value("reviewed").run()
    assert not app.exception
    app.segmented_control(key="wl_filter").set_value("all").run()
    app.toggle(key="wl_compact").set_value(True).run()
    assert len(app.dataframe) >= 1
    cid = app.selectbox(key="wl_pick").options[0].split(" · ")[1]
    app.selectbox(key="wl_pick").set_value(cid).run()
    assert app.session_state["view"] == "read" and app.session_state["case"] == cid


def test_onboarding_without_runs(tmp_path, monkeypatch):
    if U.find_runs(ROOT / "outputs"):
        pytest.skip("this checkout has runs in outputs/")
    monkeypatch.delenv("RA_RUN", raising=False)
    monkeypatch.chdir(tmp_path)
    at = st_testing.AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=60)
    at.run()
    assert not at.exception and "No reading session yet" in _md(at)
