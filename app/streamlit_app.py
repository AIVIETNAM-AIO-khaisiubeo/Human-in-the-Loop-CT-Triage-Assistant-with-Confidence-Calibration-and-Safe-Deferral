"""ReaderAdapt-Liver - interactive review app.

    python -m streamlit run app/streamlit_app.py -- --run outputs/lits

(or set the environment variable RA_RUN=outputs/lits). Data: LiTS17. Models: ViT-UNETR 2.5D
segmenter + MedGemma 1.5 second reader.

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

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(APP_DIR.parent))

import ui_core as U  # noqa: E402
from radagent import RESEARCH_BANNER  # noqa: E402
from radagent.experiments import POLICY_LABELS  # noqa: E402
from radagent.plots import POLICY_COLORS, SERIES, style  # noqa: E402
from radagent.readers.feedback import FeedbackStore  # noqa: E402
from radagent.report import draft_report  # noqa: E402
from radagent.triage import CATEGORY_HELP  # noqa: E402

st.set_page_config(page_title="ReaderAdapt-Liver", page_icon="🩻", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1500px;}
.ra-banner {background: rgba(227,73,72,0.12); border: 1px solid rgba(227,73,72,0.45); color: inherit;
  padding: 6px 12px; border-radius: 8px; font-size: 0.82rem; letter-spacing: .02em; margin-bottom: .6rem;}
.ra-card {border: 1px solid rgba(128,128,128,0.28); border-radius: 10px; padding: 14px 16px; margin-bottom: 10px;}
.ra-card h4 {margin: 0 0 6px 0; font-size: 1.02rem;}
.ra-muted {opacity: .72; font-size: .86rem;}
.ra-pill {display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem; font-weight: 600;
  border: 1px solid rgba(128,128,128,.35); margin-right: 6px;}
.ra-legend span {display:inline-block; margin-right: 14px; font-size: .82rem;}
.ra-legend i {display:inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 5px; vertical-align: -1px;}
</style>
""", unsafe_allow_html=True)


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


@st.cache_resource(show_spinner="Loading run ...")
def _load_run_cached(run: str, mtime: float) -> dict:
    return U.load_run(Path(run))


runs = U.find_runs(APP_DIR.parent / "outputs") + U.find_runs("outputs")
runs = list(dict.fromkeys(p.resolve() for p in runs))
cli = _cli_run()
if cli and Path(cli).resolve() not in runs and (Path(cli) / "cases").exists():
    runs.insert(0, Path(cli).resolve())
if not runs:
    st.title("ReaderAdapt-Liver")
    st.markdown(f"<div class='ra-banner'>⚠️ {RESEARCH_BANNER}</div>", unsafe_allow_html=True)
    st.warning("No finished run was found in `outputs/`. The website shows the results of a pipeline run on LiTS17.")
    st.markdown("""
**Create a run (GPU needed, about 1-2 hours on Colab):**
1. Open `notebooks/colab_lits_pipeline.ipynb` in Google Colab with a GPU runtime.
2. Add the Colab secrets `KAGGLE_API_TOKEN` (LiTS download) and `HF_TOKEN` (MedGemma, after accepting the
   terms on the `google/medgemma-1.5-4b-it` model page).
3. Run all cells. The last cell either opens this website directly from Colab, or downloads `lits_run.zip`.

**Then, on your computer:** unzip `lits_run.zip` into the `outputs/` folder of this repository and start
`python -m streamlit run app/streamlit_app.py -- --run outputs/lits`.

Or run the steps yourself on a GPU machine: `python -m radagent all --config configs/lits.yaml`.
""")
    st.stop()

with st.sidebar:
    st.markdown("### 🩻 ReaderAdapt-Liver")
    default_idx = 0
    if cli:
        for i, r in enumerate(runs):
            if r == Path(cli).resolve():
                default_idx = i
    run = st.selectbox("Run", runs, index=default_idx, format_func=lambda p: p.name)

R = _load_run_cached(str(run), _cases_mtime(run))
cfg, store, records, cases = R["cfg"], R["store"], R["records"], R["cases"]
profiles = U.reader_profiles(cfg)
reader_options = list(profiles) + [U.LIVE_READER]
reader_names = {**{k: v.name for k, v in profiles.items()}, U.LIVE_READER: "Live reader (you)"}

# --------------------------------------------------------------------------- sidebar controls
with st.sidebar:
    reader_id = st.selectbox("Reader", reader_options, format_func=lambda r: reader_names[r],
                             help="Simulated readers A/B/C have fixed behaviour models; 'Live reader' is you.")
    if reader_id in profiles:
        st.caption(profiles[reader_id].description)
    policy_name = st.radio("Decision policy", list(U.POLICY_CHOICES), index=3,
                           format_func=lambda p: U.POLICY_CHOICES[p])
    splits = st.multiselect("Worklist splits", ["test", "calib"], default=["test"],
                            help="The test split is the held-out reading session; calib cases were used to fit the calibrator.")
    sequential = st.toggle("Sequential reading protocol", value=True,
                           help="Record your independent impression before the AI suggestion is revealed. "
                                "This is what makes reader-specific trust learnable.")
    research = st.toggle("Research mode: show reference annotation", value=False,
                         help="Shows the dataset's reference outline and correctness. Off during a real read.")

events_by_reader = U.all_events(store, reader_options)
policy = U.current_policy(policy_name, reader_id, events_by_reader, cases, cfg)
reviews = FeedbackStore(store.p("feedback")).latest_by_case(reader_id)
wl = U.worklist(records, cases, policy, cfg, reviews, splits=tuple(splits or ["test"]))

with st.sidebar:
    if wl.empty:
        st.warning("No cases in the selected splits.")
        st.stop()
    labels = {row.case_id: f"#{row.rank}  {row.case_id}  ·  {row.category.lower()}" for row in wl.itertuples()}
    if "case_id" not in st.session_state or st.session_state.case_id not in labels:
        st.session_state.case_id = wl.case_id.iloc[0]
    case_id = st.selectbox("Current case (worklist order)", list(labels), key="case_id", format_func=labels.get)
    n_ev = len(events_by_reader.get(reader_id, []))
    st.markdown(f"**Feedback from {reader_names[reader_id].split(' - ')[0]}:** {n_ev} reads")
    if st.button("Reset this reader's feedback"):
        FeedbackStore(store.p("feedback")).reset(reader_id)
        st.session_state.pop("impressions", None)
        st.rerun()
    st.markdown(f"<div class='ra-banner'>⚠️ {RESEARCH_BANNER}</div>", unsafe_allow_html=True)

record = records[case_id]
case = cases[case_id]
decision = policy.decide(case["p"], case["u"], case["qc_passed"])

# --------------------------------------------------------------------------- header
st.markdown(f"<div class='ra-banner'>⚠️ {RESEARCH_BANNER} · public de-identified data · "
            f"the radiologist decides every finding</div>", unsafe_allow_html=True)
st.title("Reader-adaptive first read · liver CT")
cal = R.get("calibration") or {}
mc = R.get("model_card") or {}
k1, k2, k6, k3, k4, k5 = st.columns(6)
k1.metric("Worklist", f"{len(wl)} cases", f"{int((wl.category == 'DEFERRED').sum())} deferred", delta_color="off")
k2.metric("Segmenter", mc.get("backend", record.get("model", {}).get("backend", "-")),
          f"val Dice T {mc.get('val_dice_slicewise', {}).get('tumour', float('nan')):.2f}", delta_color="off",
          help="Tumour Dice on the validation scans (slice-wise, model resolution).")
_tools = cal.get("tools_test") or {}
_n_vlm = sum(1 for r in records.values() if (r.get("vlm") or {}).get("status") == "ok")
if _tools.get("medgemma"):
    k6.metric("MedGemma", f"AUROC {_tools['medgemma']['auroc']:.2f}",
              f"segmenter {_tools['segmenter']['auroc']:.2f}", delta_color="off",
              help="Case-level AUROC on the test split: MedGemma's P(yes) vs the segmenter score.")
else:
    k6.metric("MedGemma", f"{_n_vlm} reads", "second reader", delta_color="off")
if cal.get("test"):
    k3.metric("Test ECE", f"{cal['test']['after']['ece']:.3f}", f"{cal['test']['after']['ece'] - cal['test']['before']['ece']:+.3f} vs raw",
              delta_color="inverse")
    k4.metric("Test Brier", f"{cal['test']['after']['brier']:.3f}",
              f"{cal['test']['after']['brier'] - cal['test']['before']['brier']:+.3f} vs raw", delta_color="inverse")
k5.metric("Suggestion threshold", f"{policy.threshold:.2f}",
          "personalised" if policy.personalised else "universal", delta_color="off")

tabs = st.tabs(["🗂 Worklist", "🔍 Case review", "📝 Report draft", "🧠 Reader adaptation", "📊 Evaluation",
                "🤖 Agent trace", "ℹ️ About"])

# =========================================================================== worklist
with tabs[0]:
    st.markdown(f"**Policy for {reader_names[reader_id]}:** {policy.summary()}")
    if policy.notes:
        st.caption(" · ".join(policy.notes))
    changed = wl[(wl.category != wl.universal_category)]
    if len(changed) and policy_name != "universal":
        st.info(f"Compared with the universal policy, {len(changed)} case(s) change category for this reader "
                f"({int((changed.category == 'DEFERRED').sum())} deferred to an independent read).")
    show = wl[["rank", "case_id", "triage", "ai_suggestion", "tumour_probability", "uncertainty", "medgemma",
               "agree", "tumour_ml", "lesions", "status"]]
    st.dataframe(show, hide_index=True, height=min(640, 38 + 35 * len(show)), column_config={
        "rank": st.column_config.NumberColumn("#", width="small"),
        "case_id": "Case",
        "triage": "Triage",
        "ai_suggestion": "AI suggestion",
        "tumour_probability": st.column_config.ProgressColumn("Tumour probability", min_value=0.0, max_value=1.0, format="%.2f"),
        "uncertainty": st.column_config.ProgressColumn("Uncertainty", min_value=0.0, max_value=1.0, format="%.2f"),
        "medgemma": st.column_config.NumberColumn("MedGemma P", format="%.2f",
                                                  help="MedGemma's calibrated probability of a liver tumour"),
        "agree": st.column_config.TextColumn("Segmenter vs MedGemma"),
        "tumour_ml": st.column_config.NumberColumn("Tumour (ml)", format="%.2f"),
        "lesions": st.column_config.NumberColumn("Lesions"),
        "status": "Review status",
    })
    w_ = cfg["triage"]["weights"]
    st.caption(f"Priority = {w_['probability']:.2f} × calibrated probability + {w_['volume']:.2f} × lesion-volume term "
               f"+ {w_['uncertainty']:.2f} × uncertainty, + {cfg['triage'].get('disagreement_bonus', 0.1):.2f} when the "
               "segmenter and MedGemma disagree (uncertain and disputed cases are pulled up on purpose). "
               "Pick a case in the sidebar, then open **Case review**.")
    with st.expander("Triage categories"):
        for k, v in CATEGORY_HELP.items():
            st.markdown(f"- {U.CATEGORY_ICON.get(k, '')} **{k}** - {v}")
        st.markdown(f"- {U.CATEGORY_ICON['DEFERRED']} **DEFERRED** - the reader-adapted policy withholds the AI "
                    "suggestion and asks for an independent read.")

# =========================================================================== case review
with tabs[1]:
    vpath = store.viewer_path(case_id)
    if not vpath.exists():
        st.error(f"Viewer data missing for {case_id}")
        st.stop()
    arrs = _load_viewer(str(vpath), vpath.stat().st_mtime)
    imp_key = (reader_id, case_id)
    st.session_state.setdefault("impressions", {})
    need_impression = (reader_id == U.LIVE_READER and sequential and decision["decision"] == "show"
                       and imp_key not in st.session_state.impressions and case_id not in reviews)
    ai_visible = not need_impression

    left, right = st.columns([1.15, 1.0], gap="large")
    with left:
        n = len(arrs["z"])
        default = U.key_slice_index(arrs, record)
        sl = st.slider("Axial slice", 0, n - 1, default, key=f"slice_{case_id}",
                       help=f"Liver slab, slice {int(arrs['z'][0])}-{int(arrs['z'][-1])} of the original volume")
        c1, c2, c3, c4 = st.columns(4)
        show_liver = c1.toggle("Liver", True, disabled=not ai_visible)
        show_les = c2.toggle("Lesions", True, disabled=not ai_visible)
        show_unc = c3.toggle("Uncertainty", False, disabled=not ai_visible)
        win = c4.select_slider("Contrast", options=["wide", "liver", "narrow"], value="liver")
        contrast = {"wide": (0, 255), "liver": (40, 230), "narrow": (80, 200)}[win]
        kept = st.session_state.get(f"kept_{reader_id}_{case_id}")
        img = U.render_slice(arrs, sl, show_liver=show_liver and ai_visible, show_lesions=show_les and ai_visible,
                             kept_lesions=kept, show_uncertainty=show_unc and ai_visible, show_reference=research,
                             upscale=max(1, 512 // arrs["ct"].shape[1]), contrast=contrast)
        st.image(img, width=560, caption=f"{case_id} · slice {int(arrs['z'][sl])} · radiological view "
                                         f"(patient's right on the left, anterior up)")
        legend = ["<i style='background:#1baf7a'></i>AI liver", "<i style='background:#eb6834'></i>AI lesion",
                  "<i style='background:#e87ba4'></i>voxel entropy"]
        if research:
            legend.append("<i style='background:#f0efec'></i>reference (tumour solid, liver dotted)")
        st.markdown("<div class='ra-legend'>" + "".join(f"<span>{x}</span>" for x in legend) + "</div>",
                    unsafe_allow_html=True)

    with right:
        m, c, q = record["measurements"], record["confidence"], record["quality"]
        if need_impression:
            st.markdown("<div class='ra-card'><h4>Step 1 · Your independent impression</h4>"
                        "<div class='ra-muted'>The AI suggestion is hidden until you commit to your own read "
                        "(sequential reading protocol).</div></div>", unsafe_allow_html=True)
            imp = st.radio("Is there a liver tumour?", ["Tumour present", "No tumour"], index=None, horizontal=True,
                           key=f"imp_radio_{case_id}")
            if st.button("Record impression and reveal AI", type="primary", disabled=imp is None):
                st.session_state.impressions[imp_key] = int(imp == "Tumour present")
                st.rerun()
        else:
            if decision["decision"] == "defer":
                st.markdown(f"<div class='ra-card'><h4>🧑‍⚕️ AI suggestion withheld</h4>"
                            f"<div>Reason: {decision['reason']}. Please read this case independently.</div>"
                            f"<div class='ra-muted'>Policy: {policy.summary()}</div></div>", unsafe_allow_html=True)
            else:
                lab = "Liver tumour suspected" if decision["ai_label"] else "No liver tumour suspected"
                pill = "personalised" if policy.personalised else "universal"
                st.markdown(f"<div class='ra-card'><h4>AI suggestion · {lab}</h4>"
                            f"<span class='ra-pill'>p = {c['tumour_probability']:.2f}</span>"
                            f"<span class='ra-pill'>uncertainty {c['uncertainty']:.2f} ({c['uncertainty_level']})</span>"
                            f"<span class='ra-pill'>threshold {policy.threshold:.2f} · {pill}</span>"
                            f"<div class='ra-muted' style='margin-top:8px'>{c['calibrator']}; "
                            f"segmenter raw score {c['tumour_probability_raw']:.2f}; model disagreement "
                            f"{c['uncertainty_components'].get('mutual_information', 0.0):.2f} bits</div>"
                            + (f"<div style='margin-top:8px'><span class='ra-pill'>segmenter "
                               f"{c.get('segmenter_probability', 0):.2f}</span><span class='ra-pill'>MedGemma "
                               f"{c['vlm_probability']:.2f}</span>"
                               + ("<span class='ra-pill'>⚠️ the two AI tools disagree</span>" if c.get("tools_disagree")
                                  else "<span class='ra-pill'>tools agree</span>") + "</div>"
                               if c.get("vlm_probability") is not None else "")
                            + "</div>", unsafe_allow_html=True)
            mm1, mm2, mm3 = st.columns(3)
            mm1.metric("Liver volume", f"{m.get('liver_volume_ml', 0):.0f} ml")
            mm2.metric("Tumour volume (AI)", f"{m.get('predicted_tumour_volume_ml', 0):.2f} ml")
            mm3.metric("Lesions (AI)", m.get("n_lesions", 0))
            if not q["passed"]:
                st.warning("Quality control failed: " + "; ".join(q["failed_reasons"]))
            if m.get("lesions"):
                les = pd.DataFrame(m["lesions"])[["id", "volume_ml", "max_axial_diameter_mm", "mean_hu",
                                                  "mean_probability", "slice_range"]]
                st.dataframe(les, hide_index=True, height=min(250, 38 + 35 * len(les)), column_config={
                    "id": "#", "volume_ml": st.column_config.NumberColumn("Volume (ml)", format="%.2f"),
                    "max_axial_diameter_mm": st.column_config.NumberColumn("Max diam. (mm)", format="%.0f"),
                    "mean_hu": st.column_config.NumberColumn("Mean HU", format="%.0f"),
                    "mean_probability": st.column_config.NumberColumn("Mean prob.", format="%.2f"),
                    "slice_range": "Slices"})
            v = record.get("vlm") or {}
            with st.expander("🔬 MedGemma second read (vision-language model)", expanded=False):
                if v.get("status") != "ok":
                    st.caption(f"No MedGemma read for this case: {v.get('reason', v.get('status', 'not run'))}.")
                else:
                    st.markdown(f"**{v.get('model', {}).get('model_id', 'VLM')}** read {v.get('n_images')} axial "
                                f"slices through the predicted liver. P(\"yes, liver tumour\") = **{v['p_yes']:.2f}**"
                                + (f", calibrated {c['vlm_probability']:.2f}" if c.get("vlm_probability") is not None else "")
                                + ".")
                    if "vlm_rgb" in arrs:
                        st.image([im for im in arrs["vlm_rgb"]], width=96,
                                 caption=[f"slice {int(z)}" for z in arrs["vlm_z"]])
                        st.caption("What the model saw: red = wide window, green = soft tissue, blue = brain window "
                                   "(Google's CT input format for MedGemma 1.5).")
                    if v.get("text"):
                        st.markdown("**Model text** (unverified; never used for measurements or triage):")
                        st.text(v["text"])
                    for note in (v.get("guard") or {}).get("notes", []):
                        st.warning(f"Guardrail: {note}")

        st.divider()
        prev = reviews.get(case_id)
        if prev:
            st.success(f"Already reviewed: **{prev['action']}** → final finding "
                       f"**{'tumour' if prev['final_label'] else 'no tumour'}** ({prev['timestamp']}).")
            if research:
                st.caption("Outcome vs reference: " + U.outcome_label(prev, case["gt_present"]))

        if reader_id in profiles:
            prof = profiles[reader_id]
            st.markdown(f"#### Simulate {prof.name}")
            b1, b2 = st.columns([1, 1])
            if b1.button(f"▶ Simulate this read", type="primary", key=f"sim_{case_id}"):
                before = policy.threshold
                ev = U.simulate_and_log(store, prof, case, policy, salt=len(events_by_reader.get(reader_id, [])))
                st.session_state.last_sim = (ev, before)
                st.rerun()
            nmore = b2.number_input("Next N unread cases", 1, 40, 10, key="nmore")
            if b2.button("⏩ Simulate N reads (policy updates after each)", key="simn"):
                evs = list(events_by_reader.get(reader_id, []))
                done = {e["case_id"] for e in evs}
                todo = [cid for cid in wl.case_id if cid not in done][: int(nmore)]
                for cid in todo:
                    pol = U.current_policy(policy_name, reader_id, {**events_by_reader, reader_id: evs}, cases, cfg)
                    ev = U.simulate_and_log(store, prof, cases[cid], pol, salt=len(evs))
                    evs.append(ev)
                st.session_state.last_sim = (None, len(todo))
                st.rerun()
            if "last_sim" in st.session_state:
                ev, info = st.session_state.pop("last_sim")
                if ev is None:
                    st.info(f"Simulated {info} reads. The policy was refitted after each one - see **Reader adaptation**.")
                else:
                    txt = (f"Independent impression: **{'tumour' if ev['initial_read'] else 'no tumour'}** → "
                           f"action **{ev['action']}** → final **{'tumour' if ev['final_label'] else 'no tumour'}**")
                    if ev.get("switched"):
                        txt += " (switched to the AI)"
                    st.success(txt)
                    if research:
                        st.caption("Outcome vs reference: " + U.outcome_label(ev, ev["gt_present"]))
        else:
            st.markdown("#### Your decision")
            init = st.session_state.impressions.get(imp_key)
            if decision["decision"] == "defer" or not ai_visible:
                if ai_visible:
                    fin = st.radio("Final finding", ["Tumour present", "No tumour"], index=None, horizontal=True,
                                   key=f"fin_{case_id}")
                    if st.button("Submit independent read", type="primary", disabled=fin is None):
                        U.log_live(store, case, policy, init, "manual", int(fin == "Tumour present"))
                        st.rerun()
            else:
                a1, a2, a3 = st.columns(3)
                ai_lab = int(decision["ai_label"])
                if a1.button("✅ Accept", type="primary"):
                    U.log_live(store, case, policy, init, "accept", ai_lab)
                    st.rerun()
                if a3.button("❌ Reject"):
                    U.log_live(store, case, policy, init, "reject", 1 - ai_lab)
                    st.rerun()
                with st.expander("✏️ Edit the suggestion", expanded=False):
                    les_ids = [les["id"] for les in m.get("lesions", [])]
                    keep = st.multiselect("Lesions to keep", les_ids, default=les_ids, key=f"keep_{case_id}")
                    st.session_state[f"kept_{reader_id}_{case_id}"] = keep
                    vol = sum(les["volume_ml"] for les in m.get("lesions", []) if les["id"] in keep)
                    st.caption(f"Edited tumour volume: {vol:.2f} ml (exact, from original-resolution lesion volumes)")
                    pres = st.radio("Tumour present after edit?", ["Yes", "No"], index=0 if (keep or ai_lab) else 1,
                                    horizontal=True, key=f"pres_{case_id}")
                    note = st.text_input("Note (optional)", key=f"note_{case_id}")
                    if st.button("Save edit"):
                        final = int(pres == "Yes")
                        action = "edit" if final == ai_lab else "reject"
                        U.log_live(store, case, policy, init, action, final,
                                   edit={"kept_lesions": keep, "edited_volume_ml": round(vol, 3), "note": note})
                        st.rerun()

# =========================================================================== report
with tabs[2]:
    rv = reviews.get(case_id)
    review = None
    if rv:
        review = {**rv, "reader_name": reader_names[reader_id]}
    pol_info = {**decision, "personalised": policy.personalised}
    text = draft_report(record, review=review, policy=pol_info)
    st.code(text, language=None)
    d1, d2 = st.columns(2)
    d1.download_button("Download report (.txt)", text, file_name=f"{case_id}_report.txt")
    d2.download_button("Download case record (.json)", json.dumps(record, indent=2, default=str),
                       file_name=f"{case_id}.json", mime="application/json")
    st.caption("The report is a fixed template filled with measured values - no language model writes clinical text.")

# =========================================================================== reader adaptation
with tabs[3]:
    evs = events_by_reader.get(reader_id, [])
    st.markdown(f"#### {reader_names[reader_id]} · {len(evs)} feedback events")
    if not evs:
        st.info("No feedback yet. Review or simulate some cases in **Case review** - the policy refits after every read.")
    else:
        style()
        variant = policy_name if policy_name.startswith("reader") else "reader_threshold_defer"
        traj = U.policy_trajectory(evs, cases, cfg, variant)
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(5.6, 3.4))
            ax.step(traj.n_feedback, traj.threshold, where="post", color=SERIES[0], label="suggestion threshold")
            if traj.defer_uncertainty.notna().any():
                ax.step(traj.n_feedback, traj.defer_uncertainty, where="post", color=SERIES[1], label="defer if uncertainty ≥")
            ax.axhline(cfg["policy"]["universal_threshold"], color="#898781", lw=1)
            ax.set_ylim(0, 1.02)
            ax.set_xlabel("Feedback events")
            ax.set_title("How this reader's policy evolved", loc="left")
            ax.legend(loc="lower right")
            st.pyplot(fig, clear_figure=True)
        with col2:
            fig, ax = plt.subplots(figsize=(5.6, 3.4))
            cgrid = np.linspace(0.5, 1.0, 50)
            ax.plot(cgrid, policy.trust(cgrid) if policy.trust_alpha is not None else
                    U.fit_policy(evs, cases, cfg, reader_id, "reader_threshold").trust(cgrid), color=SERIES[2],
                    label="fitted trust curve")
            dis = [e for e in evs if e.get("shown") and e.get("initial_read") is not None and e["ai_label"] is not None
                   and e["initial_read"] != e["ai_label"]]
            if dis:
                x = np.array([e["ai_confidence"] for e in dis])
                y = np.array([e["final_label"] == e["ai_label"] for e in dis], dtype=float)
                ax.scatter(x, y + np.random.default_rng(0).uniform(-0.03, 0.03, len(y)), s=40, color=SERIES[2],
                           edgecolor="#fcfcfb", linewidth=1.5, alpha=0.8, label="disagreements (1 = switched to AI)")
            if reader_id in profiles:
                ax.plot(cgrid, [profiles[reader_id].trust(v) for v in cgrid], color="#898781", lw=1.5, label="true (simulation)")
            ax.set_ylim(-0.08, 1.08)
            ax.set_xlabel("AI confidence in its suggestion")
            ax.set_ylabel("P(switch to AI | disagree)")
            ax.set_title("Learned trust curve", loc="left")
            ax.legend(loc="upper left", fontsize=8)
            st.pyplot(fig, clear_figure=True)
        df = pd.DataFrame(evs)
        acts = df["action"].value_counts().reindex(["accept", "edit", "reject", "manual"]).fillna(0).astype(int)
        a1, a2, a3, a4 = st.columns(4)
        for col, (k, v) in zip((a1, a2, a3, a4), acts.items()):
            col.metric(k.capitalize(), int(v))
        if research:
            df["reference"] = df["case_id"].map(lambda cid: int(cases[cid]["gt_present"]))
            df["outcome"] = [U.outcome_label(e, cases[e["case_id"]]["gt_present"]) for e in evs]
        cols = [c for c in ["timestamp", "case_id", "policy", "ai_label", "ai_probability", "initial_read", "action",
                            "final_label", "reference", "outcome", "source"] if c in df.columns]
        st.dataframe(df[cols].iloc[::-1], hide_index=True, height=300)
    st.caption("Feedback is used to learn *how this reader uses the AI* (trust curve, lapse rate). Correctness for "
               "policy fitting comes from the reference annotation - a reader's correction never becomes ground truth.")

# =========================================================================== evaluation
with tabs[4]:
    fig_dir = store.p("figures")
    if _tools:
        st.markdown("#### The AI tools on the held-out test split")
        from radagent.plots import tool_names

        seg_n, vlm_n = tool_names(cfg)
        names = {"segmenter": f"Segmenter score - {seg_n} (Platt)", "medgemma": f"{vlm_n} P(yes) (Platt)",
                 "fused": f"Fused: segmenter + {vlm_n} (logistic)"}
        st.dataframe(pd.DataFrame([{"tool": names.get(k, k), "cases": r["n"], "with tumour": r["n_pos"],
                                    "AUROC": r["auroc"], "Brier": r["brier"], "ECE": r["ece"], "NLL": r["nll"]}
                                   for k, r in _tools.items()]).round(3), hide_index=True)
        fit = cal.get("fit", {})
        st.caption(f"Each tool calibrated on the calib split only. Used for the worklist: "
                   f"**{cal.get('calibrator', {}).get('description', '-')}**, selected by cross-validated NLL "
                   f"{fit.get('cv_nll', {})}. Few tumour-free LiTS cases make specificity estimates noisy.")
    s = R.get("summary")
    if s is not None:
        st.markdown("#### Policy comparison on the held-out test split (simulated readers)")
        show = s.copy()
        show["policy"] = show["policy"].map(POLICY_LABELS)
        show["reader"] = show["reader"].map(lambda r: reader_names.get(r, r))
        st.dataframe(show[["reader", "policy", "accuracy", "sensitivity", "specificity", "harmful_acceptance_rate",
                           "over_reliance", "deferral_rate", "actions_per_case", "threshold"]].round(3), hide_index=True)
    for name, title in (("ai_tools.png", "Segmenter vs MedGemma"), ("policy_comparison.png", "Policies × readers"),
                        ("benefit_by_reader.png", "RQ1 benefit spread"), ("learning_curve.png", "RQ3 learning curve"), ("calibration_reliability.png", "Calibration"),
                        ("risk_coverage.png", "Selective prediction"), ("triage_lift.png", "Triage"),
                        ("segmentation_quality.png", "Segmentation")):
        f = fig_dir / name
        if f.exists():
            st.markdown(f"##### {title}")
            st.image(str(f), width=900)
    b = R.get("benefit_spread")
    if b is not None:
        st.markdown("#### RQ1 · spread of benefit across readers")
        b = b.copy()
        b["policy"] = b["policy"].map(POLICY_LABELS)
        st.dataframe(b.round(3), hide_index=True)
    st.caption("Simulated readers only: these numbers show the method works as designed; they are not evidence of "
               "clinical benefit or reading-time savings.")

# =========================================================================== agent trace
with tabs[5]:
    tp = store.trace_path(case_id)
    if tp.exists():
        trace = json.loads(tp.read_text())
        st.markdown(f"**Planner:** {trace['planner']} · status **{trace['status']}** · "
                    f"{trace.get('n_guardrail_overrides', 0)} guardrail override(s)")
        rows = [{"step": s_["step"], "tool": s_["tool"], "planned by": s_["planned_by"], "reason": s_.get("reason", ""),
                 "ms": s_.get("duration_ms"), "result": json.dumps(s_.get("summary"), default=str)[:160],
                 "guardrail": s_.get("guardrail") or ""} for s_ in trace["steps"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    cpl, cbtn = st.columns([1, 1])
    planner_choice = cpl.radio("Re-run the agent on this case with", ["rule", "llm"], horizontal=True,
                               help="LLM = any OpenAI-compatible server (e.g. Ollama) configured under agent.llm")
    from radagent.data.lits import locate

    ct_ok = Path(locate(str(store.manifest().set_index("case_id").loc[case_id, "ct_path"]), cfg)).exists()
    torch_ok = importlib.util.find_spec("torch") is not None
    can_recompute = ct_ok and (torch_ok or cfg["segmenter"]["backend"] == "plugin")
    recompute = cpl.toggle("Recompute from the CT file (runs the segmenter and MedGemma now)", value=False,
                           disabled=not can_recompute,
                           help="Needs the LiTS CT on this machine and PyTorch (a GPU for MedGemma). "
                                + ("" if can_recompute else
                                   ("CT file not found here. " if not ct_ok else "") +
                                   ("PyTorch not installed. " if not torch_ok else "")))
    label = "🤖 Run segmentation + MedGemma now" if recompute else "🤖 Re-run agent (uses cached perception)"
    if cbtn.button(label):
        from radagent.pipeline import run_single_case

        with st.spinner("Segmenting the CT and asking MedGemma (first call loads the models) ..." if recompute
                        else "Agent running ..."):
            rec2, tr2 = run_single_case(cfg, case_id, planner_choice, reader_policy=policy, recompute=recompute)
        st.session_state["rerun_trace"] = tr2
        if recompute:
            st.success("Recomputed - the case record, viewer images and report were updated.")
            _load_run_cached.clear()
            st.rerun()
    if "rerun_trace" in st.session_state and st.session_state["rerun_trace"]["case_id"] == case_id:
        tr2 = st.session_state["rerun_trace"]
        st.markdown(f"**Re-run:** planner {tr2['planner']} → {tr2['status']}, "
                    f"{tr2.get('n_guardrail_overrides', 0)} guardrail override(s)")
        st.dataframe(pd.DataFrame([{"step": s_["step"], "tool": s_["tool"], "planned by": s_["planned_by"],
                                    "reason": s_.get("reason", ""), "guardrail": s_.get("guardrail") or "",
                                    "result": json.dumps(s_.get("summary"), default=str)[:160]}
                                   for s_ in tr2["steps"]]), hide_index=True)

# =========================================================================== about
with tabs[6]:
    st.markdown("""
#### What this prototype does
Data: **LiTS17** (Liver Tumor Segmentation Challenge, ISBI/MICCAI 2017; Kaggle mirror, 131 CT volumes with
reference masks).

1. **Agentic first read** - a planner (deterministic rules, or an open LLM via any OpenAI-compatible server) calls
   tools in a guarded order: load study → segment liver/tumour with the **ViT-UNETR 2.5D** (stochastic passes) →
   measure at original resolution → quality control → **MedGemma 1.5 second read** of slices through the liver →
   calibrated confidence → triage → (reader policy) → report.
2. **Calibration** - the segmenter score and MedGemma's P(yes) are calibrated on a separate split (alone or fused,
   chosen by cross-validated NLL), with Brier score, ECE and reliability diagrams before/after. When the two AI tools
   disagree, the case is flagged and moved up the worklist.
3. **Learning from corrections** - accept/edit/reject actions (sequential protocol) teach the system each reader's
   *trust curve*; a reader-specific threshold and deferral rule are chosen to minimise that reader's expected error.
4. **Evaluation** - universal vs reader-specific vs deferral policies on the same held-out cases with common random
   numbers, plus a learning curve for how much feedback is needed.
""")
    st.graphviz_chart("""
digraph G { rankdir=LR; node [shape=box, style="rounded", fontname="Helvetica", fontsize=10];
  ct [label="CT volume\\n(NIfTI, LiTS)"]; plan [label="Planner\\nrule | open LLM"]; guard [label="Guardrails\\npreconditions"];
  seg [label="Segmenter tool\\nViT-UNETR 2.5D"]; meas [label="Measure\\n(voxel spacing)"]; qc [label="Quality\\ncontrol"];
  vlm [label="Second reader\\nMedGemma 1.5"];
  conf [label="Calibrated\\nconfidence"]; tri [label="Triage"]; pol [label="Reader policy\\nthreshold + defer"];
  rep [label="Report\\ntemplate"]; ui [label="Review UI\\naccept / edit / reject"]; fb [label="Feedback log"];
  fit [label="Per-reader\\nfit (trust curve)"];
  ct -> seg; ct -> vlm; plan -> guard -> seg; seg -> meas -> qc -> vlm -> conf -> tri -> pol -> rep -> ui -> fb -> fit -> pol;
}
""")
    st.markdown(f"Run directory: `{store.root}` · config: `{store.root / 'config_used.yaml'}`")
