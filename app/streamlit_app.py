"""ReaderAdapt-Liver - the radiologist's reading interface.

    python -m streamlit run app/streamlit_app.py -- --run outputs/lits

(or set the environment variable RA_RUN=outputs/lits). Data: LiTS17. AI tools: ViT-UNETR 2.5D
segmenter + MedGemma 1.5 second reader. The layout follows the reading journey described in
docs/UX_DESIGN.md: worklist -> independent impression -> AI suggestion -> decision -> report,
with adaptation, performance and audit views kept out of the reading flow.

Research prototype. Every output is a research prediction, not for clinical use.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(APP_DIR.parent))

import ui_core as U  # noqa: E402
import ui_theme as T  # noqa: E402
from radagent.experiments import POLICY_LABELS  # noqa: E402
from radagent.readers.feedback import FeedbackStore  # noqa: E402
from radagent.report import draft_report  # noqa: E402
from radagent.triage import CATEGORY_HELP  # noqa: E402

RESEARCH_LABEL = "Research prototype · not for clinical use"

st.set_page_config(page_title="ReaderAdapt · Liver CT", page_icon="🩻", layout="wide",
                   initial_sidebar_state="auto")
st.markdown(T.style_tag(), unsafe_allow_html=True)


def html(s: str) -> None:
    st.markdown(s, unsafe_allow_html=True)


def cols(name: str, spec, **kw):
    """Columns that reflow by the space they actually have (see ui_theme.REFLOW), not only by screen width:
    with the sidebar open on a tablet, the viewer and the AI panel stack instead of being squeezed."""
    return st.container(key=f"rx_{name}").columns(spec, **kw)


# --------------------------------------------------------------------------- run selection
def _cli_run() -> str | None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--run", default=None)
    args, _ = ap.parse_known_args(sys.argv[1:])
    return args.run or os.environ.get("RA_RUN")


@st.cache_data(show_spinner=False)
def _load_viewer(path: str, mtime: float) -> dict:
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def _cases_mtime(run: Path) -> float:
    files = list((run / "cases").glob("*.json"))
    return max((f.stat().st_mtime for f in files), default=0.0)


@st.cache_resource(show_spinner="Loading the reading session ...")
def _load_run_cached(run: str, mtime: float) -> dict:
    return U.load_run(Path(run))


runs = U.find_runs(APP_DIR.parent / "outputs") + U.find_runs("outputs")
runs = list(dict.fromkeys(p.resolve() for p in runs))
cli = _cli_run()
if cli and Path(cli).resolve() not in runs and (Path(cli) / "cases").exists():
    runs.insert(0, Path(cli).resolve())

if not runs:
    html(T.header(None, None, RESEARCH_LABEL, "-"))
    html(T.section_title("No reading session yet",
                         "The website shows the results of a pipeline run on the LiTS17 liver CT data."))
    html(T.note("No finished run was found in <code>outputs/</code>.", "warn"))
    c1, c2 = cols("onboard", 2, gap="large")
    with c1.container(border=True):
        st.markdown("""
#### Create a run on Google Colab (GPU)
1. Open `notebooks/colab_lits_pipeline.ipynb` in Colab with a GPU runtime.
2. Add the secrets `KAGGLE_API_TOKEN` (LiTS download) and `HF_TOKEN` (MedGemma - accept the terms on the
   `google/medgemma-1.5-4b-it` model page first).
3. Run all cells. The last cells open this website from Colab or download `lits_run.zip`.
""")
    with c2.container(border=True):
        st.markdown("""
#### Open a run on this computer
1. Unzip `lits_run.zip` into the `outputs/` folder of this repository.
2. Start `python -m streamlit run app/streamlit_app.py -- --run outputs/lits`.

On a GPU machine you can also run every step yourself:
`python -m radagent all --config configs/lits.yaml`.
""")
    st.stop()

# --------------------------------------------------------------------------- session settings (sidebar)
ss = st.session_state
with st.sidebar:
    html("<div class='ra-side-t'>Session settings</div>"
         "<div class='ra-side-s'>Usually set once at the start of a reading session.</div>")
    default_idx = 0
    if cli:
        for i, r in enumerate(runs):
            if r == Path(cli).resolve():
                default_idx = i
    run = st.selectbox("Results run", runs, index=default_idx, format_func=lambda p: p.name,
                       help="A finished pipeline run (records, images, calibration, experiments).")

R = _load_run_cached(str(run), _cases_mtime(run))
cfg, store, records, cases = R["cfg"], R["store"], R["records"], R["cases"]
profiles = U.reader_profiles(cfg)
reader_options = [U.LIVE_READER] + list(profiles)
reader_names = {**{k: v.name for k, v in profiles.items()}, U.LIVE_READER: "You (live reader)"}

with st.sidebar:
    reader_id = st.selectbox("Reading as", reader_options, format_func=lambda r: reader_names[r], key="reader",
                             help="'You' records your own decisions. Readers A, B and C are simulated readers "
                                  "with fixed behaviour, used to study how the assistant adapts.")
    if reader_id in profiles:
        st.caption(profiles[reader_id].description)
    policy_name = st.radio("Decision policy", list(U.POLICY_CHOICES), index=3, key="policy",
                           format_func=lambda p: U.POLICY_CHOICES[p],
                           help="How the AI suggestion is shown: one threshold for everyone, or adapted to this "
                                "reader, with or without deferring uncertain cases to an independent read.")
    splits = st.multiselect("Worklist", ["test", "calib"], default=["test"], key="splits",
                            format_func={"test": "Held-out test cases", "calib": "Calibration cases"}.get,
                            help="The test cases are the held-out reading session; calibration cases were used "
                                 "to fit the confidence calibration.")
    sequential = st.toggle("Independent impression first", value=True, key="sequential",
                           help="Sequential reading protocol: record your own impression before the AI suggestion "
                                "is revealed. This avoids anchoring and makes your trust in the AI learnable.")
    research = st.toggle("Research mode (show reference)", value=False, key="research",
                         help="Shows the dataset's reference annotation and whether each decision was correct. "
                              "Keep it off during a real read.")

events_by_reader = U.all_events(store, reader_options)
policy = U.current_policy(policy_name, reader_id, events_by_reader, cases, cfg)
reviews = FeedbackStore(store.p("feedback")).latest_by_case(reader_id)
wl = U.worklist(records, cases, policy, cfg, reviews, splits=tuple(splits or ["test"]))
reader_name = reader_names[reader_id]
short_reader = reader_name.split(" - ")[0]


def _reset_feedback(rid: str) -> None:
    FeedbackStore(store.p("feedback")).reset(rid)
    ss.pop("impressions", None)


with st.sidebar:
    st.divider()
    n_ev = len(events_by_reader.get(reader_id, []))
    html(f"<div class='ra-side-t'>Feedback</div><div class='ra-side-s'>{n_ev} decisions logged for "
         f"{T.esc(short_reader)}. The assistant adapts to these.</div>")
    with st.popover("Reset feedback", icon=":material/restart_alt:", disabled=n_ev == 0):
        st.markdown(f"Delete the **{n_ev}** logged decisions of **{short_reader}**? "
                    "The decision policy returns to the universal one.")
        st.button("Delete feedback", type="primary", key="reset_confirm", on_click=_reset_feedback, args=(reader_id,), width="stretch")
    st.caption(f"Run: `{run.name}`")

# --------------------------------------------------------------------------- header + status
policy_chip = f"{U.POLICY_CHOICES[policy_name]} · threshold {policy.threshold:.2f}" + \
              (" (personalised)" if policy.personalised else "")
html(T.header(reader_name, policy_chip, RESEARCH_LABEL, run.name))

if wl.empty:
    html(T.note("No cases in the selected worklist. Choose a worklist in <b>Session settings</b>.", "warn"))
    st.stop()

pending = int((wl.status == "pending").sum())
html(T.kpi_strip([
    ("To read", f"{pending}<span>/ {len(wl)}</span>", "cases waiting", "todo"),
    ("Urgent", str(int(((wl.category == "URGENT") & (wl.status == "pending")).sum())), "high-confidence findings",
     "urgent"),
    ("AI tools disagree", str(int((wl.agree == "disagree").sum())), "segmenter vs MedGemma", "disagree"),
    ("Independent reads", str(int((wl.category == "DEFERRED").sum())), "AI suggestion withheld", "deferred"),
    ("Reviewed", str(len(wl) - pending), "decisions saved", "done"),
]))

# --------------------------------------------------------------------------- navigation state
VIEWS = {"worklist": ":material/format_list_bulleted: Worklist", "read": ":material/visibility: Read case",
         "report": ":material/description: Report", "adapt": ":material/tune: AI adaptation",
         "perf": ":material/monitoring: Performance", "audit": ":material/fact_check: Audit trail",
         "about": ":material/info: About"}
order = list(wl.case_id)
if "view" not in ss:
    qv, qc = st.query_params.get("view"), st.query_params.get("case")
    ss.view = qv if qv in VIEWS else "worklist"
    if qc in order:
        ss.case = qc
if ss.get("case") not in order:
    ss.case = order[0]
case_id = ss.case


def _open(cid: str, view: str = "read") -> None:
    ss.case = cid
    ss.view = view


def _go(view: str) -> None:
    ss.view = view


def _nav_changed() -> None:
    if ss.nav is None:          # clicking the active section again must not leave the page empty
        ss.nav = ss.view
    ss.view = ss.nav


ss.nav = ss.view
st.segmented_control("Section", list(VIEWS), format_func=VIEWS.get, key="nav", on_change=_nav_changed,
                     label_visibility="collapsed")
st.query_params["view"] = ss.view
st.query_params["case"] = case_id

record = records[case_id]
case = cases[case_id]
decision = policy.decide(case["p"], case["u"], case["qc_passed"])
row = wl.set_index("case_id").loc[case_id]


def _case_label(cid: str) -> str:
    r = wl.set_index("case_id").loc[cid]
    return f"#{r['rank']} · {cid} · {U.CATEGORY_LABEL.get(r['category'], r['category'])}"


# =========================================================================== views
def view_worklist() -> None:
    html(T.section_title("Worklist", f"{len(wl)} cases in priority order. Uncertain cases and cases where the two "
                                     "AI tools disagree are deliberately placed higher."))
    changed = wl[(wl.category != wl.universal_category)]
    if len(changed) and policy_name != "universal":
        html(T.note(f"For {T.esc(short_reader)}, the adapted policy changes the category of {len(changed)} case(s) "
                    f"({int((changed.category == 'DEFERRED').sum())} sent for an independent read).", "info"))
    f1, f2 = cols("wlfilter", [3, 1.3], vertical_alignment="bottom")
    with f1:
        show = st.segmented_control("Show", ["all", "pending", "reviewed"], default="all", key="wl_filter",
                                    format_func={"all": "All cases", "pending": "To read",
                                                 "reviewed": "Reviewed"}.get) or "all"
    with f2:
        compact = st.toggle("Compact table", key="wl_compact", help="One row per case - for large screens.")
    rows = wl if show == "all" else wl[(wl.status == "pending") == (show == "pending")]
    if rows.empty:
        html(T.note("No cases here." if show == "reviewed" else "All cases on this worklist have been reviewed.",
                    "ok"))
        return
    if compact:
        tcols = ["rank", "case_id", "triage", "ai_suggestion", "tumour_probability", "uncertainty", "medgemma",
                "agree", "tumour_ml", "lesions", "status"]
        st.dataframe(rows[tcols], hide_index=True, height=min(720, 40 + 36 * len(rows)), column_config={
            "rank": st.column_config.NumberColumn("#", width="small"), "case_id": "Case", "triage": "Category",
            "ai_suggestion": "AI suggestion",
            "tumour_probability": st.column_config.ProgressColumn("Probability", min_value=0.0, max_value=1.0,
                                                                  format="%.2f"),
            "uncertainty": st.column_config.ProgressColumn("Uncertainty", min_value=0.0, max_value=1.0, format="%.2f"),
            "medgemma": st.column_config.NumberColumn("MedGemma", format="%.2f",
                                                      help="MedGemma's calibrated probability of a liver tumour"),
            "agree": "AI tools", "tumour_ml": st.column_config.NumberColumn("Tumour (ml)", format="%.1f"),
            "lesions": st.column_config.NumberColumn("Lesions"), "status": "Status"})
        st.selectbox("Open a case", list(rows.case_id), format_func=_case_label, key="wl_pick", index=None,
                     placeholder="Choose a case to read ...",
                     on_change=lambda: ss.get("wl_pick") and _open(ss.wl_pick))
    else:
        first = next((c for c, s in zip(wl.case_id, wl.status) if s == "pending"), None)
        for r in rows.itertuples():
            with st.container(border=True, key=f"wl_{r.case_id}"):
                a, b, c, d = st.columns([1.0, 2.6, 2.2, 1.3], vertical_alignment="center", gap="medium")
                a.markdown(f"<div class='ra-rank'><span>Priority</span>#{r.rank}</div>{T.badge(r.category)}",
                           unsafe_allow_html=True)
                b.markdown(T.case_card_main(r.case_id, r.ai_suggestion, r.reasons), unsafe_allow_html=True)
                c.markdown(T.case_card_ai(r.tumour_probability, policy.threshold, r.unc_level,
                                          None if np.isnan(r.medgemma) else r.medgemma, r.agree),
                           unsafe_allow_html=True)
                with d:
                    html(T.status_badge(r.status))
                    st.button("Open case" if r.status == "pending" else "Review again", key=f"open_{r.case_id}",
                              type="primary" if r.case_id == first else "secondary", on_click=_open,
                              args=(r.case_id,), icon=":material/arrow_forward:", width="stretch")
    with st.expander("What do the categories mean?"):
        for k, v in CATEGORY_HELP.items():
            html(f"<p>{T.badge(k)} &nbsp;{T.esc(v)}</p>")
        html(f"<p>{T.badge('DEFERRED')} &nbsp;The adapted policy withholds the AI suggestion for this reader and asks "
             "for an independent read.</p>")
        w_ = cfg["triage"]["weights"]
        st.caption(f"Priority = {w_['probability']:.2f} × calibrated probability + {w_['volume']:.2f} × lesion-volume "
                   f"term + {w_['uncertainty']:.2f} × uncertainty, + {cfg['triage'].get('disagreement_bonus', 0.1):.2f} "
                   "when the segmenter and MedGemma disagree.")


def _record_impression(key: tuple, value: str | None) -> None:
    if value is not None:
        ss.setdefault("impressions", {})[key] = int(value == "present")


def _log(action: str, final: int, init, edit: dict | None = None) -> None:
    U.log_live(store, case, policy, init, action, final, edit=edit)


def view_read() -> None:
    prev_id, next_id = U.neighbours(order, case_id)
    n1, n2, n3 = st.container(key="casenav").columns([1, 3, 1], vertical_alignment="bottom")
    n1.button("Previous", key="prev_case", disabled=prev_id is None, on_click=_open, args=(prev_id,),
              icon=":material/arrow_back:", width="stretch")
    ss.case_pick = case_id
    n2.selectbox("Case (worklist order)", order, key="case_pick", format_func=_case_label,
                 on_change=lambda: _open(ss.case_pick))
    n3.button("Next", key="next_case", disabled=next_id is None, on_click=_open, args=(next_id,),
              icon=":material/arrow_forward:", width="stretch")

    vpath = store.viewer_path(case_id)
    if not vpath.exists():
        html(T.note(f"Image data for {T.esc(case_id)} is missing in this run.", "warn"))
        return
    arrs = _load_viewer(str(vpath), vpath.stat().st_mtime)
    imp_key = (reader_id, case_id)
    impressions = ss.setdefault("impressions", {})
    reviewed = case_id in reviews
    need_impression = (reader_id == U.LIVE_READER and sequential and decision["decision"] == "show"
                       and imp_key not in impressions and not reviewed)
    ai_visible = not need_impression
    if need_impression:
        html(T.stepper(1, set()))
    elif reviewed:
        html(T.stepper(4, {1, 2, 3}))
    else:
        html(T.stepper(3, {1, 2}))

    left, right = cols("read", [1.2, 1.0], gap="large")
    m, c, q = record["measurements"], record["confidence"], record["quality"]
    with left, st.container(border=True, key="viewer"):
        html("<div class='ra-card-t'>CT viewer</div><div class='ra-card-s'>Axial slices through the liver · "
             "radiological view (patient's right on the left, anterior up)</div>")
        n = len(arrs["z"])
        sl = st.slider("Slice", 0, n - 1, U.key_slice_index(arrs, record), key=f"slice_{case_id}",
                       help=f"Slices {int(arrs['z'][0])}-{int(arrs['z'][-1])} of the original volume. Starts at the "
                            "AI's key slice. Use the arrow keys after clicking the slider.")
        win = st.segmented_control("Window", ["wide", "liver", "narrow"], default="liver", key="window",
                                   format_func={"wide": "Wide", "liver": "Liver", "narrow": "Narrow"}.get) or "liver"
        t1, t2, t3 = cols("overlays", 3)
        show_liver = t1.toggle("AI liver outline", True, disabled=not ai_visible, key="ov_liver")
        show_les = t2.toggle("AI lesions", True, disabled=not ai_visible, key="ov_lesions")
        show_unc = t3.toggle("Uncertainty map", False, disabled=not ai_visible, key="ov_unc")
        contrast = {"wide": (0, 255), "liver": (40, 230), "narrow": (80, 200)}[win]
        kept = ss.get(f"kept_{reader_id}_{case_id}")
        img = U.render_slice(arrs, sl, show_liver=show_liver and ai_visible, show_lesions=show_les and ai_visible,
                             kept_lesions=kept, show_uncertainty=show_unc and ai_visible, show_reference=research,
                             upscale=max(1, 512 // arrs["ct"].shape[1]), contrast=contrast)
        st.image(img, width="stretch",
                 caption=f"{case_id} · slice {int(arrs['z'][sl])} of the original CT · window {win}")
        items = [("AI liver outline", T.OVERLAY["liver"], "line"), ("AI lesion", T.OVERLAY["lesion"], "fill"),
                 ("Uncertainty", T.OVERLAY["uncertainty"], "fill")]
        if research:
            items.append(("Reference (tumour solid, liver dotted)", T.OVERLAY["reference"], "dotted"))
        html(T.legend(items))
        if not ai_visible:
            st.caption("AI overlays are hidden until you record your impression.")

    with right:
        if need_impression:
            with st.container(border=True, key="impression"):
                html("<div class='ra-card-t'>Step 1 · Your independent impression</div>"
                     "<div class='ra-card-s'>Read the scan first. The AI suggestion stays hidden until you commit to "
                     "your own impression.</div>")
                imp = st.segmented_control("Is there a liver tumour?", ["present", "absent"], key=f"imp_{case_id}", width="stretch",
                                           format_func={"present": "Tumour present", "absent": "No tumour"}.get)
                st.button("Record impression and show the AI suggestion", type="primary", disabled=imp is None,
                          key="record_imp", on_click=_record_impression, args=(imp_key, imp),
                          icon=":material/lock_open:", width="stretch")
        else:
            _ai_panel(m, c, q, arrs)
        _decision_panel(m, imp_key, reviewed, ai_visible, next_id)


def _ai_panel(m: dict, c: dict, q: dict, arrs: dict) -> None:
    seg_name, vlm_name = _tool_names()
    if decision["decision"] == "defer":
        html(T.withheld_card(decision["reason"], policy.summary()))
    else:
        positive = bool(decision["ai_label"])
        prov = (f"{seg_name} segmentation + {vlm_name} second read; probability calibrated on held-out "
                f"cases ({c.get('calibrator', 'calibrated')}).")
        html(T.ai_summary(
            headline="Liver tumour suspected" if positive else "No liver tumour suspected", positive=positive,
            p=c["tumour_probability"], threshold=policy.threshold, unc_level=c["uncertainty_level"],
            unc=c["uncertainty"], seg_p=c.get("segmenter_probability"), vlm_p=c.get("vlm_probability"),
            disagree=c.get("tools_disagree"), reasons=str(row["reasons"]).split("; ") if row["reasons"] else [],
            provenance=prov))
    html(T.tiles([("Liver volume", f"{m.get('liver_volume_ml', 0):.0f}", "ml"),
                  ("Tumour volume", f"{m.get('predicted_tumour_volume_ml', 0):.1f}", "ml"),
                  ("Lesions", f"{m.get('n_lesions', 0)}", "")]))
    if not q["passed"]:
        html(T.note("<b>Quality control failed.</b> " + T.esc("; ".join(q["failed_reasons"])) +
                    " Read this case fully yourself.", "warn"))
    if m.get("lesions"):
        les = pd.DataFrame(m["lesions"])[["id", "volume_ml", "max_axial_diameter_mm", "mean_hu", "mean_probability",
                                          "slice_range"]]
        html("<div class='ra-card-t' style='margin-top:6px'>Lesions found by the AI</div>")
        st.dataframe(les, hide_index=True, height=min(260, 40 + 36 * len(les)), column_config={
            "id": "#", "volume_ml": st.column_config.NumberColumn("Volume (ml)", format="%.2f"),
            "max_axial_diameter_mm": st.column_config.NumberColumn("Longest diameter (mm)", format="%.0f"),
            "mean_hu": st.column_config.NumberColumn("Mean HU", format="%.0f"),
            "mean_probability": st.column_config.NumberColumn("Mean prob.", format="%.2f"),
            "slice_range": "Slices"})
    v = record.get("vlm") or {}
    with st.expander(f"Second reader · {vlm_name}", icon=":material/neurology:"):
        if v.get("status") != "ok":
            st.caption(f"No second read for this case: {v.get('reason', v.get('status', 'not run'))}.")
        else:
            st.markdown(f"**{v.get('model', {}).get('model_id', 'VLM')}** looked at {v.get('n_images')} axial slices "
                        f"through the predicted liver. Probability of \"yes, liver tumour\": **{v['p_yes']:.2f}**"
                        + (f" (calibrated {c['vlm_probability']:.2f})" if c.get("vlm_probability") is not None else "")
                        + ".")
            if "vlm_rgb" in arrs:
                st.image([im for im in arrs["vlm_rgb"]], width=104,
                         caption=[f"slice {int(z)}" for z in arrs["vlm_z"]])
                st.caption("Input format of MedGemma 1.5 for CT: red = wide window, green = soft tissue, "
                           "blue = brain window.")
            if v.get("text"):
                html(T.note("<b>Unverified model text.</b> Quoted as written; never used for measurements or triage.",
                            "info"))
                st.text(v["text"])
            for note_ in (v.get("guard") or {}).get("notes", []):
                html(T.note("<b>Guardrail:</b> " + T.esc(note_), "warn"))
    with st.expander("Technical details", icon=":material/settings:"):
        mi = c.get("uncertainty_components", {}).get("mutual_information", 0.0)
        st.markdown(f"""
- Calibration: {c.get('calibrator', '-')}
- Segmenter raw score {c['tumour_probability_raw']:.3f} → segmenter probability {c.get('segmenter_probability', float('nan')):.3f}
- Uncertainty {c['uncertainty']:.3f} bits (model disagreement between passes {mi:.3f} bits)
- Policy: {policy.summary()}
""")
        if policy.notes:
            st.caption(" · ".join(policy.notes))


def _decision_panel(m: dict, imp_key: tuple, reviewed: bool, ai_visible: bool, next_id: str | None) -> None:
    prev = reviews.get(case_id)
    if prev:
        with st.container(border=True, key="saved"):
            fin = "tumour present" if prev["final_label"] else "no tumour"
            act = T.STATUS_STYLE.get(prev["action"], ("", prev["action"]))[1]
            html(T.note(f"<b>Decision saved: {T.esc(act)}.</b> Final finding: <b>{fin}</b> "
                        f"<span class='ra-muted'>({T.esc(str(prev['timestamp'])[:16].replace('T', ' '))})</span>", "ok"))
            if research:
                st.caption("Outcome vs reference: " + U.outcome_label(prev, case["gt_present"]))
            b1, b2 = cols("after", 2)
            b1.button("Next case", type="primary", key="after_next", disabled=next_id is None, on_click=_open,
                      args=(next_id,), icon=":material/arrow_forward:", width="stretch")
            b2.button("View report", key="after_report", on_click=_go, args=("report",), icon=":material/description:", width="stretch")
    if reader_id in profiles:
        _simulation_panel()
        return
    if prev or not ai_visible:
        return
    init = ss.get("impressions", {}).get(imp_key)
    with st.container(border=True, key="decision"):
        html("<div class='ra-card-t'>Step 3 · Your decision</div>")
        if decision["decision"] == "defer":
            html("<div class='ra-card-s'>Record your own finding for this case.</div>")
            fin = st.segmented_control("Final finding", ["present", "absent"], key=f"fin_{case_id}", width="stretch",
                                       format_func={"present": "Tumour present", "absent": "No tumour"}.get)
            st.button("Save independent read", type="primary", disabled=fin is None, key="save_manual",
                      on_click=_log, args=("manual", int(fin == "present"), init), icon=":material/check:", width="stretch")
            return
        ai_lab = int(decision["ai_label"])
        lab = "tumour present" if ai_lab else "no tumour"
        html(f"<div class='ra-card-s'>Accepting records <b>{lab}</b>; rejecting records the opposite. "
             "You can also edit which lesions are kept.</div>")
        a1, a2 = cols("decide", 2)
        a1.button("Accept suggestion", type="primary", key="accept", on_click=_log, args=("accept", ai_lab, init),
                  icon=":material/check:", width="stretch")
        a2.button("Reject suggestion", key="reject", on_click=_log, args=("reject", 1 - ai_lab, init),
                  icon=":material/close:", width="stretch")
        with st.expander("Edit the suggestion", icon=":material/edit:"):
            les_ids = [les["id"] for les in m.get("lesions", [])]
            keep = st.multiselect("Lesions to keep", les_ids, default=les_ids, key=f"keep_{case_id}")
            ss[f"kept_{reader_id}_{case_id}"] = keep
            vol = sum(les["volume_ml"] for les in m.get("lesions", []) if les["id"] in keep)
            st.caption(f"Edited tumour volume: {vol:.2f} ml (from the original-resolution lesion volumes).")
            pres = st.segmented_control("Tumour present after your edit?", ["yes", "no"],
                                        default="yes" if (keep or ai_lab) else "no", key=f"pres_{case_id}", width="stretch",
                                        format_func={"yes": "Yes", "no": "No"}.get) or "no"
            note_ = st.text_input("Note (optional)", key=f"note_{case_id}")
            final = int(pres == "yes")
            st.button("Save edit", key="save_edit", on_click=_log,
                      args=("edit" if final == ai_lab else "reject", final, init,
                            {"kept_lesions": keep, "edited_volume_ml": round(vol, 3), "note": note_}),
                      icon=":material/save:", width="stretch")


def _simulation_panel() -> None:
    prof = profiles[reader_id]
    with st.container(border=True, key="simulation"):
        html(f"<div class='ra-card-t'>Research simulation · {T.esc(prof.name)}</div>"
             f"<div class='ra-card-s'>{T.esc(prof.description)}</div>")
        b1, b2 = cols("sim", 2, vertical_alignment="bottom")
        if b1.button("Simulate this read", type="primary", key=f"sim_{case_id}", icon=":material/play_arrow:", width="stretch"):
            before = policy.threshold
            ev = U.simulate_and_log(store, prof, case, policy, salt=len(events_by_reader.get(reader_id, [])))
            ss.last_sim = (ev, before)
            st.rerun()
        nmore = b2.number_input("Next N unread cases", 1, 40, 10, key="nmore")
        if st.button(f"Simulate the next {int(nmore)} reads", key="simn", icon=":material/fast_forward:",
                     help="The policy is refitted after every read.", width="stretch"):
            evs = list(events_by_reader.get(reader_id, []))
            done = {e["case_id"] for e in evs}
            todo = [cid for cid in wl.case_id if cid not in done][: int(nmore)]
            for cid in todo:
                pol = U.current_policy(policy_name, reader_id, {**events_by_reader, reader_id: evs}, cases, cfg)
                evs.append(U.simulate_and_log(store, prof, cases[cid], pol, salt=len(evs)))
            ss.last_sim = (None, len(todo))
            st.rerun()
        if "last_sim" in ss:
            ev, info = ss.pop("last_sim")
            if ev is None:
                html(T.note(f"Simulated {info} reads. The policy was refitted after each one - see "
                            "<b>AI adaptation</b>.", "ok"))
            else:
                txt = (f"Impression <b>{'tumour' if ev['initial_read'] else 'no tumour'}</b> → action "
                       f"<b>{T.esc(ev['action'])}</b> → final <b>{'tumour' if ev['final_label'] else 'no tumour'}</b>"
                       + (" (switched to the AI)" if ev.get("switched") else ""))
                html(T.note(txt, "ok"))
                if research:
                    st.caption("Outcome vs reference: " + U.outcome_label(ev, ev["gt_present"]))


def _tool_names() -> tuple[str, str]:
    from radagent.plots import tool_names

    return tool_names(cfg)


def view_report() -> None:
    rv = reviews.get(case_id)
    html(T.section_title(f"Report draft · {case_id}",
                         "A fixed template filled with the measured values. No language model writes clinical "
                         "text; MedGemma's words are quoted and marked unverified."))
    s1, s2 = cols("reportbar", [3, 1.2], vertical_alignment="center")
    if rv:
        act = T.STATUS_STYLE.get(rv["action"], ("", rv["action"]))[1]
        s1.markdown(T.note(f"Reviewed by <b>{T.esc(reader_name)}</b>: {T.esc(act)}.", "ok"), unsafe_allow_html=True)
    else:
        s1.markdown(T.note("Draft - not yet reviewed. Your decision is added once you save it in <b>Read case</b>.",
                           "info"), unsafe_allow_html=True)
    s2.button("Back to the case", key="report_back", on_click=_go, args=("read",), icon=":material/visibility:", width="stretch")
    review = {**rv, "reader_name": reader_name} if rv else None
    text = draft_report(record, review=review, policy={**decision, "personalised": policy.personalised})
    with st.container(key="report_doc"):
        st.code(text, language=None, wrap_lines=True)
    d1, d2 = cols("downloads", 2)
    d1.download_button("Download report (.txt)", text, file_name=f"{case_id}_report.txt", icon=":material/download:", width="stretch")
    d2.download_button("Download case record (.json)", json.dumps(record, indent=2, default=str),
                       file_name=f"{case_id}.json", mime="application/json", icon=":material/data_object:", width="stretch")


def _chart_style(ax, title: str) -> None:
    ax.set_title(title, loc="left", color=T.NAVY, fontsize=12, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(T.BORDER_STRONG)
    ax.tick_params(colors=T.MUTED, labelsize=10)
    ax.xaxis.label.set_color(T.INK)
    ax.yaxis.label.set_color(T.INK)
    ax.grid(color="#E4EBF1", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.figure.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")


def view_adapt() -> None:
    evs = events_by_reader.get(reader_id, [])
    html(T.section_title(f"How the assistant adapts to {short_reader}",
                         "From each saved decision it learns how this reader uses the AI, then sets the suggestion "
                         "threshold and decides when to withhold the suggestion."))
    defer_txt = "never" if policy.defer_uncertainty is None else f"uncertainty ≥ {policy.defer_uncertainty:.2f}"
    html(T.tiles([("Decisions logged", str(len(evs)), ""),
                  ("Suggestion threshold", f"{policy.threshold:.2f}", "personalised" if policy.personalised else "universal"),
                  ("Withhold suggestion if", defer_txt, "")]))
    min_fb = cfg["policy"]["min_feedback"]
    if not policy.personalised:
        html(T.note(f"The policy is not personalised yet. It adapts after at least {min_fb} decisions and only when "
                    "that clearly lowers the expected error for this reader.", "info"))
    if policy.notes:
        st.caption(" · ".join(policy.notes))
    if not evs:
        html(T.note("No decisions yet. Read or simulate cases in <b>Read case</b>; the policy is refitted after every "
                    "decision.", "info"))
        return
    variant = policy_name if policy_name.startswith("reader") else "reader_threshold_defer"
    traj = U.policy_trajectory(evs, cases, cfg, variant)
    col1, col2 = cols("charts", 2, gap="large")
    with col1:
        fig, ax = plt.subplots(figsize=(5.8, 3.5))
        ax.step(traj.n_feedback, traj.threshold, where="post", color=T.TEAL, lw=2.4, label="Suggestion threshold")
        if traj.defer_uncertainty.notna().any():
            ax.step(traj.n_feedback, traj.defer_uncertainty, where="post", color="#1D4E89", lw=2.0, ls="--",
                    label="Withhold if uncertainty ≥")
        ax.axhline(cfg["policy"]["universal_threshold"], color=T.BORDER_STRONG, lw=1.2)
        ax.set_ylim(0, 1.02)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_xlabel("Decisions logged")
        _chart_style(ax, "How the policy evolved")
        ax.legend(loc="lower right", frameon=False, fontsize=9)
        st.pyplot(fig, clear_figure=True)
    with col2:
        fig, ax = plt.subplots(figsize=(5.8, 3.5))
        cgrid = np.linspace(0.5, 1.0, 50)
        fitted = policy.trust(cgrid) if policy.trust_alpha is not None else \
            U.fit_policy(evs, cases, cfg, reader_id, "reader_threshold").trust(cgrid)
        ax.plot(cgrid, fitted, color=T.TEAL, lw=2.4, label="Learned from decisions")
        dis = [e for e in evs if e.get("shown") and e.get("initial_read") is not None and e["ai_label"] is not None
               and e["initial_read"] != e["ai_label"]]
        if dis:
            x = np.array([e["ai_confidence"] for e in dis])
            y = np.array([e["final_label"] == e["ai_label"] for e in dis], dtype=float)
            ax.scatter(x, y + np.random.default_rng(0).uniform(-0.03, 0.03, len(y)), s=42, color=T.NAVY,
                       edgecolor="#FFFFFF", linewidth=1.2, alpha=0.85, label="Disagreements (1 = followed the AI)")
        if reader_id in profiles:
            ax.plot(cgrid, [profiles[reader_id].trust(v) for v in cgrid], color=T.BORDER_STRONG, lw=1.6, ls=":",
                    label="True behaviour (simulation)")
        ax.set_ylim(-0.08, 1.08)
        ax.set_xlabel("AI confidence in its suggestion")
        ax.set_ylabel("P(follow the AI | disagree)")
        _chart_style(ax, "Trust curve")
        ax.legend(loc="upper left", frameon=False, fontsize=8.5)
        st.pyplot(fig, clear_figure=True)
    df = pd.DataFrame(evs)
    acts = df["action"].value_counts().reindex(["accept", "edit", "reject", "manual"]).fillna(0).astype(int)
    html(T.tiles([(T.STATUS_STYLE[k][1], str(int(v)), "") for k, v in acts.items()]))
    if research:
        df["reference"] = df["case_id"].map(lambda cid: int(cases[cid]["gt_present"]))
        df["outcome"] = [U.outcome_label(e, cases[e["case_id"]]["gt_present"]) for e in evs]
    ecols = [c for c in ["timestamp", "case_id", "policy", "ai_label", "ai_probability", "initial_read", "action",
                        "final_label", "reference", "outcome", "source"] if c in df.columns]
    with st.expander("All logged decisions"):
        st.dataframe(df[ecols].iloc[::-1], hide_index=True, height=320)
    st.caption("Decisions teach the assistant how this reader uses the AI. Whether a decision was correct always comes "
               "from the reference annotation - a reader's correction never becomes ground truth.")


def view_perf() -> None:
    cal = R.get("calibration") or {}
    mc = R.get("model_card") or {}
    tools = cal.get("tools_test") or {}
    html(T.section_title("Performance and evidence",
                         "For quality assurance. Held-out LiTS17 test cases; reader results come from simulated "
                         "readers and are not evidence of clinical benefit."))
    t = []
    if cal.get("test"):
        a, b = cal["test"]["after"], cal["test"]["before"]
        t += [("Test AUROC", f"{a['auroc']:.2f}", ""), ("Calibration error (ECE)", f"{a['ece']:.3f}",
                                                         f"raw {b['ece']:.3f}"),
              ("Brier score", f"{a['brier']:.3f}", f"raw {b['brier']:.3f}")]
    dice = (mc.get("val_dice_slicewise") or {}).get("tumour")
    if dice is not None:
        t.append(("Segmenter tumour Dice (validation)", f"{dice:.2f}", ""))
    if t:
        html(T.tiles(t[:3]) + (T.tiles(t[3:]) if len(t) > 3 else ""))
    if tools:
        seg_n, vlm_n = _tool_names()
        names = {"segmenter": f"Segmenter score · {seg_n}", "medgemma": f"{vlm_n} P(yes)",
                 "fused": f"Combined: segmenter + {vlm_n}"}
        st.markdown("#### The AI tools on the test cases")
        st.dataframe(pd.DataFrame([{"Tool": names.get(k, k), "Cases": r["n"], "With tumour": r["n_pos"],
                                    "AUROC": r["auroc"], "Brier": r["brier"], "ECE": r["ece"], "NLL": r["nll"]}
                                   for k, r in tools.items()]).round(3), hide_index=True)
        st.caption(f"Each tool calibrated on the calibration cases only. Used for the worklist: "
                   f"{cal.get('calibrator', {}).get('description', '-')}.")
    s = R.get("summary")
    if s is not None:
        st.markdown("#### Decision policies with simulated readers")
        show = s.copy()
        show["policy"] = show["policy"].map(POLICY_LABELS)
        show["reader"] = show["reader"].map(lambda r: reader_names.get(r, r))
        labels = {"reader": "Reader", "policy": "Policy", "accuracy": "Accuracy", "sensitivity": "Sensitivity",
                  "specificity": "Specificity", "harmful_acceptance_rate": "Harmful acceptance",
                  "over_reliance": "Over-reliance", "deferral_rate": "Deferral rate",
                  "actions_per_case": "Actions / case", "threshold": "Threshold"}
        tab = show[list(labels)].rename(columns=labels)
        for col in tab.columns[2:]:                     # one type per column; "-" where not applicable
            tab[col] = [("-" if pd.isna(x) else f"{x:.3f}".rstrip("0").rstrip(".")) for x in tab[col]]
        st.dataframe(tab, hide_index=True)
        st.caption("Harmful acceptance: an incorrect AI suggestion was accepted. Actions per case: accept 1, reject 2, "
                   "edit 3, independent read 4.")
    figs = [(n, title) for n, title in (
        ("ai_tools.png", "Segmenter vs MedGemma"), ("calibration_reliability.png", "Calibration"),
        ("segmentation_quality.png", "Segmentation"), ("triage_lift.png", "Worklist triage"),
        ("risk_coverage.png", "Selective prediction"), ("policy_comparison.png", "Policies × readers"),
        ("benefit_by_reader.png", "Benefit by reader (RQ1)"), ("learning_curve.png", "Learning curve (RQ3)"))
        if (store.p("figures") / n).exists()]
    if figs:
        st.markdown("#### Figures")
        for i in range(0, len(figs), 2):
            row = cols(f"figs{i}", 2, gap="large")
            for col, (n, title) in zip(row, figs[i:i + 2]):
                with col, st.container(border=True):
                    html(f"<div class='ra-card-t'>{T.esc(title)}</div>")
                    st.image(str(store.p("figures") / n), width="stretch")
    b = R.get("benefit_spread")
    if b is not None:
        with st.expander("Spread of benefit across readers (RQ1)"):
            b = b.copy()
            b["policy"] = b["policy"].map(POLICY_LABELS)
            st.dataframe(b.round(3), hide_index=True)


def view_audit() -> None:
    html(T.section_title(f"Audit trail · {case_id}",
                         "Every tool the agent called for this case, in order, with its reason, result and timing."))
    tp = store.trace_path(case_id)
    if tp.exists():
        trace = json.loads(tp.read_text())
        html(T.tiles([("Planner", str(trace["planner"]), ""), ("Status", str(trace["status"]), ""),
                      ("Guardrail overrides", str(trace.get("n_guardrail_overrides", 0)), "")]))
        rows = [{"Step": s_["step"], "Tool": s_["tool"], "Planned by": s_["planned_by"], "Reason": s_.get("reason", ""),
                 "ms": s_.get("duration_ms"), "Result": json.dumps(s_.get("summary"), default=str)[:160],
                 "Guardrail": s_.get("guardrail") or ""} for s_ in trace["steps"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    else:
        html(T.note("No stored trace for this case.", "info"))
    from radagent.data.lits import locate

    ct_ok = Path(locate(str(store.manifest().set_index("case_id").loc[case_id, "ct_path"]), cfg)).exists()
    torch_ok = importlib.util.find_spec("torch") is not None
    can_recompute = ct_ok and (torch_ok or cfg["segmenter"]["backend"] == "plugin")
    with st.container(border=True, key="rerun"):
        html("<div class='ra-card-t'>Run the agent again on this case</div>")
        c1, c2 = cols("rerunrow", [1.4, 1], vertical_alignment="bottom")
        planner_choice = c1.segmented_control("Planner", ["rule", "llm"], default="rule", key="planner",
                                              format_func={"rule": "Rules", "llm": "Open LLM"}.get,
                                              help="Open LLM = any OpenAI-compatible server (e.g. Ollama) configured "
                                                   "under agent.llm") or "rule"
        recompute = c1.toggle("Recompute from the CT file (runs the segmenter and MedGemma now)", value=False,
                              disabled=not can_recompute, key="recompute",
                              help="Needs the LiTS CT on this machine and PyTorch (a GPU for MedGemma). "
                                   + ("" if can_recompute else ("CT file not found here. " if not ct_ok else "")
                                      + ("PyTorch not installed. " if not torch_ok else "")))
        label = "Run segmentation + MedGemma now" if recompute else "Re-run the agent"
        if c2.button(label, type="primary", key="rerun_btn", icon=":material/replay:", width="stretch"):
            from radagent.pipeline import run_single_case

            with st.spinner("Segmenting the CT and asking MedGemma (the first call loads the models) ..." if recompute
                            else "Agent running ..."):
                _, tr2 = run_single_case(cfg, case_id, planner_choice, reader_policy=policy, recompute=recompute)
            ss["rerun_trace"] = tr2
            if recompute:
                _load_run_cached.clear()
                st.rerun()
    if "rerun_trace" in ss and ss["rerun_trace"]["case_id"] == case_id:
        tr2 = ss["rerun_trace"]
        html(T.note(f"Re-run with the {T.esc(tr2['planner'])} planner: {T.esc(tr2['status'])}, "
                    f"{tr2.get('n_guardrail_overrides', 0)} guardrail override(s).", "ok"))
        st.dataframe(pd.DataFrame([{"Step": s_["step"], "Tool": s_["tool"], "Planned by": s_["planned_by"],
                                    "Reason": s_.get("reason", ""), "Guardrail": s_.get("guardrail") or "",
                                    "Result": json.dumps(s_.get("summary"), default=str)[:160]}
                                   for s_ in tr2["steps"]]), hide_index=True)


def view_about() -> None:
    seg_n, vlm_n = _tool_names()
    html(T.section_title("About this assistant", "What it does, what it is not for, and who decides."))
    c1, c2 = cols("about", 2, gap="large")
    with c1.container(border=True):
        st.markdown(f"""
#### Intended use (research)
- Prioritise a liver-CT worklist and pre-read each case: liver and tumour segmentation, measurements at the
  original resolution, a calibrated probability and its uncertainty.
- Offer a second opinion from **{vlm_n}** and flag cases where the two AI tools disagree.
- Learn how each reader uses the AI and adapt the suggestion threshold, or withhold suggestions, accordingly.
""")
    with c2.container(border=True):
        st.markdown("""
#### Not intended for
- Clinical diagnosis or patient management - this is a research prototype on public, de-identified data.
- Replacing a radiologist's read: **the radiologist decides every finding**.
- Measurements from the language model: all sizes come from the segmentation; model text is marked unverified.
""")
    with st.container(border=True):
        st.markdown(f"""
#### How a case is pre-read
1. **Load** the CT and its voxel spacing; **segment** liver and tumour with **{seg_n}** (several stochastic passes).
2. **Measure** lesion volumes and diameters; run **quality control**.
3. **Second read** by **{vlm_n}** on slices through the liver.
4. **Calibrate** the probability on held-out cases; estimate **uncertainty**; **triage** the case.
5. Apply **your** decision policy and draft a **fixed-template report**.

Data: **LiTS17** (Liver Tumor Segmentation Challenge, ISBI / MICCAI 2017; Kaggle mirror, 131 CT volumes).
""")
    html(T.flow([("CT volume", "LiTS NIfTI, voxel spacing", "data"), ("Segmenter", seg_n, "ai"),
                 ("Measure", "volumes, diameters", "step"), ("Quality control", "plausibility checks", "step"),
                 ("Second reader", vlm_n, "ai"), ("Calibrated confidence", "probability + uncertainty", "step"),
                 ("Triage", "worklist priority", "step"), ("Your policy", "threshold, deferral", "step"),
                 ("Report draft", "fixed template", "step"), ("You decide", "accept / edit / reject", "human"),
                 ("Feedback", "adapts your policy", "data")]))
    st.caption(f"Run directory: {store.root}")


{"worklist": view_worklist, "read": view_read, "report": view_report, "adapt": view_adapt, "perf": view_perf,
 "audit": view_audit, "about": view_about}[ss.view]()
