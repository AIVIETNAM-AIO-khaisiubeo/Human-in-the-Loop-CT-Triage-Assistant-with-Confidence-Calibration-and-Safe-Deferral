"""ReaderAdapt-Liver - the radiologist's reading interface.

    python -m streamlit run app/streamlit_app.py -- --run outputs/lits

(or set the environment variable RA_RUN=outputs/lits). Data: LiTS17. AI tools: ViT-UNETR 2.5D
segmentation model + MedGemma 1.5 second-reader model.

Two working modes - Worklist and Read case - and a separate Research & quality area (reader
adaptation, model performance, audit trail, intended use). Design notes and the response to the
design review: docs/UX_DESIGN.md.

Safety rules built into the flow:
* Blind first read (default): no AI output - prediction, overlays, uncertainty, tool disagreement,
  AI priority - is shown for a case before the radiologist saves an independent impression, and
  the worklist is in arrival order.
* The decision policy is frozen for the session; a proposed update is applied only explicitly and
  the change is logged.
* Every decision is stored with the recommendation that was on screen, the session, policy and
  model-set versions.

Research prototype. Every output is a research prediction, not for clinical use.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import secrets
import sys
from datetime import datetime, timezone
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
from radagent.plots import tool_names  # noqa: E402
from radagent.readers.feedback import FeedbackStore  # noqa: E402
from radagent.report import draft_report  # noqa: E402
from radagent.triage import CATEGORY_HELP  # noqa: E402
from radagent.vlm.prompts import PROMPT_VERSION  # noqa: E402

PRIMARY = {"worklist": ":material/format_list_bulleted: Worklist", "read": ":material/visibility: Read case"}
RESEARCH = {"adapt": ("Reader adaptation", ":material/tune:"), "perf": ("Model performance", ":material/monitoring:"),
            "audit": ("Audit trail (this case)", ":material/fact_check:"),
            "about": ("About and intended use", ":material/info:")}
WINDOWS = {"abdomen": ("Abdomen", 350, 75), "liver": ("Liver", 150, 70), "lesion": ("Lesion", 100, 60)}
FINDING = {"present": "Tumour", "absent": "No tumour", "defer": "Defer"}

st.set_page_config(page_title="ReaderAdapt · Liver CT", page_icon="🩻", layout="wide", initial_sidebar_state="auto")


def html(s: str) -> None:
    st.markdown(s, unsafe_allow_html=True)


def cols(name: str, spec, **kw):
    """Columns that reflow by the space they actually have (ui_theme.REFLOW), not only by screen width."""
    return st.container(key=f"rx_{name}").columns(spec, **kw)


def now_local() -> str:
    return datetime.now().astimezone().strftime("%H:%M")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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


ss = st.session_state
if "session_id" not in ss:
    ss.session_id = "S-" + datetime.now().strftime("%m%d-%H%M") + "-" + secrets.token_hex(2).upper()
    ss.session_started = now_local()

runs = U.find_runs(APP_DIR.parent / "outputs") + U.find_runs("outputs")
runs = list(dict.fromkeys(p.resolve() for p in runs))
cli = _cli_run()
if cli and Path(cli).resolve() not in runs and (Path(cli) / "cases").exists():
    runs.insert(0, Path(cli).resolve())

if not runs:
    html(T.style_tag())
    html(T.header(None, None, None, None))
    html(T.page_title("No reading session yet",
                      "The website shows the results of a pipeline run on the LiTS17 liver CT data."))
    html(T.note("No finished run was found in <code>outputs/</code>.", "warn"))
    c1, c2 = cols("onboard", 2, gap="large")
    with c1:
        st.markdown("""
#### Create a run on Google Colab (GPU)
1. Open `notebooks/colab_lits_pipeline.ipynb` in Colab with a GPU runtime.
2. Add the secrets `KAGGLE_API_TOKEN` (LiTS download) and `HF_TOKEN` (MedGemma - accept the terms on the
   `google/medgemma-1.5-4b-it` model page first).
3. Run all cells. The last cells open this website from Colab or download `lits_run.zip`.
""")
    with c2:
        st.markdown("""
#### Open a run on this computer
1. Unzip `lits_run.zip` into the `outputs/` folder of this repository.
2. Start `python -m streamlit run app/streamlit_app.py -- --run outputs/lits`.

On a GPU machine you can also run every step yourself: `python -m radagent all --config configs/lits.yaml`.
""")
    st.stop()

# --------------------------------------------------------------------------- session settings (sidebar)
with st.sidebar:
    html("<div class='ra-side-t'>Session</div><div class='ra-side-s'>Set once at the start of a reading session.</div>")
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
reader_names = {**{k: v.name for k, v in profiles.items()}, U.LIVE_READER: "You"}

with st.sidebar:
    reader_id = st.selectbox("Reading as", reader_options, key="reader",
                             format_func=lambda r: "You (live reader)" if r == U.LIVE_READER
                             else f"{reader_names[r]} (simulated)",
                             help="'You' records your own decisions. Readers A, B and C are simulated readers used "
                                  "to study adaptation; their reads are research simulations.")
    live = reader_id == U.LIVE_READER
    if live:
        mode = st.radio("Reading mode", ["blind", "assisted"], key="reading_mode", horizontal=False,
                        format_func={"blind": "Blind first read", "assisted": "AI-assisted triage"}.get,
                        help="Blind first read: no AI output is shown for a case until you save your own "
                             "impression, and the worklist is in arrival order. AI-assisted triage: the AI suggestion "
                             "and AI priority are shown from the start.")
    else:
        mode = "assisted"
        st.caption(profiles[reader_id].description)
    policy_name = st.radio("Decision policy", list(U.POLICY_CHOICES), index=3, key="policy",
                           format_func=lambda p: U.POLICY_CHOICES[p],
                           help="How the AI suggestion is shown: one threshold for everyone, or adapted to this "
                                "reader, with or without withholding uncertain suggestions. Frozen for the session.")
    splits = st.multiselect("Worklist", ["test", "calib"], default=["test"], key="splits",
                            format_func={"test": "Held-out test cases", "calib": "Calibration cases"}.get)
    research = st.toggle("Research mode (show reference)", value=False, key="research",
                         help="Shows the dataset's reference annotation and whether each decision was correct. "
                              "Keep it off during a real read.")
blind = live and mode == "blind"

# --------------------------------------------------------------------------- versions, frozen policy, worklist
events_by_reader = U.all_events(store, reader_options)
proposed = U.current_policy(policy_name, reader_id, events_by_reader, cases, cfg)
active = ss.setdefault("active_policies", {})
pkey = f"{reader_id}|{policy_name}|{run.name}"
if pkey not in active:
    active[pkey] = {"policy": proposed, "since": now_local()}
policy = active[pkey]["policy"]
pol_ver = U.policy_version(policy)
mv = U.model_version(cfg, R.get("calibration"), R.get("model_card"))
reviews = FeedbackStore(store.p("feedback")).latest_by_case(reader_id)
impressions = ss.setdefault("impressions", {})
revealed = {cid for (rid, cid) in impressions if rid == reader_id}
wl = U.worklist(records, cases, policy, cfg, reviews, splits=tuple(splits or ["test"]), impressions=revealed)
reader_name = reader_names[reader_id]


def _reset_feedback(rid: str) -> None:
    FeedbackStore(store.p("feedback")).reset(rid)
    for k in [k for k in active if k.startswith(f"{rid}|")]:
        active.pop(k)
    for k in [k for k in impressions if k[0] == rid]:
        impressions.pop(k)


def _go(view: str) -> None:
    ss.view = view


with st.sidebar:
    st.divider()
    html("<div class='ra-side-t'>Session identity</div>"
         f"<div class='ra-kv'>Session <b>{T.esc(ss.session_id)}</b> · started {T.esc(ss.session_started)}<br>"
         f"Model set <b>{T.esc(mv['id'])}</b><br>Decision policy <b>{T.esc(pol_ver)}</b> · active since "
         f"{T.esc(active[pkey]['since'])}<br><span class='ra-muted'>The policy does not change during the session. "
         "Updates are applied only from Reader adaptation, and every change is logged.</span></div>")
    st.divider()
    html("<div class='ra-side-t'>Research and quality</div>")
    for v, (label, ico) in RESEARCH.items():
        st.button(label, key=f"go_{v}", icon=ico, type="tertiary", on_click=_go, args=(v,), width="stretch")
    n_ev = len(events_by_reader.get(reader_id, []))
    with st.popover("Reset logged decisions", icon=":material/restart_alt:", disabled=n_ev == 0, width="stretch"):
        st.markdown(f"Delete the **{n_ev}** logged decisions of **{reader_name}**? The decision policy returns to "
                    "the universal one.")
        st.button("Delete decisions", type="primary", key="reset_confirm", on_click=_reset_feedback,
                  args=(reader_id,), width="stretch")
    st.caption(f"Run: `{run.name}`")

# --------------------------------------------------------------------------- header + navigation state
html(T.style_tag())
mode_label = None if not live else ("Blind first read" if blind else "AI-assisted triage")
html(T.header(reader_name if live else f"{reader_name} (simulated)", ss.session_id, mv["id"], mode_label))

if wl.empty:
    html(T.note("No cases in the selected worklist. Choose a worklist in <b>Session</b> settings.", "warn"))
    st.stop()

sort_mode = ss.get("wl_sort", "priority")
wlo = U.order_worklist(wl, blind, sort_mode)
order = list(wlo.case_id)
if "view" not in ss:
    qv, qc = st.query_params.get("view"), st.query_params.get("case")
    ss.view = qv if qv in PRIMARY or qv in RESEARCH or qv == "report" else "worklist"
    if qc in order:
        ss.case = qc
if ss.get("case") not in order:
    ss.case = order[0]
case_id = ss.case


def _open(cid: str, view: str = "read") -> None:
    ss.case = cid
    ss.view = view


def _nav_changed() -> None:
    if ss.nav is None:
        if ss.view in PRIMARY:          # clicking the active mode again must not leave the page empty
            ss.nav = ss.view
        return
    ss.view = ss.nav


ss.nav = ss.view if ss.view in PRIMARY else None
st.segmented_control("Mode", list(PRIMARY), format_func=PRIMARY.get, key="nav", on_change=_nav_changed,
                     label_visibility="collapsed")
st.query_params["view"] = ss.view
st.query_params["case"] = case_id

record = records[case_id]
case = cases[case_id]
decision = policy.decide(case["p"], case["u"], case["qc_passed"])
row = wlo.set_index("case_id").loc[case_id]
seg_name, vlm_name = tool_names(cfg)


def _hide_ai(cid: str) -> bool:
    """Blind first read: AI output for a case stays hidden until an impression or a decision exists."""
    return blind and cid not in revealed and cid not in reviews


def _case_label(cid: str) -> str:
    r = wlo.set_index("case_id").loc[cid]
    if _hide_ai(cid) or blind:
        return f"{r['position']}. {cid} · {U.STATUS_TEXT.get(r['status'], r['status'])}"
    return (f"{r['position']}. {cid} · {U.CATEGORY_LABEL.get(r['category'], r['category'])} · "
            f"{U.pct(r['tumour_probability'])}")


def _next_unread(after: str | None) -> str | None:
    pend = [c for c, s in zip(wlo.case_id, wlo.status) if s == "pending"]
    if not pend:
        return None
    if after in order:
        later = [c for c in pend if order.index(c) > order.index(after)]
        if later:
            return later[0]
    return pend[0]


def _research_header(title: str, subtitle: str) -> None:
    c1, c2 = cols("reportbar", [4, 1.2], vertical_alignment="bottom")
    with c1:
        html("<div class='ra-crumb'>Research and quality area</div>" + T.page_title(title, subtitle))
    c2.button("Back to the worklist", key="back_wl", on_click=_go, args=("worklist",),
              icon=":material/arrow_back:", width="stretch")


# =========================================================================== Worklist
def view_worklist() -> None:
    sub = ("Blind first read: cases in arrival order; AI findings stay hidden until you save your impression."
           if blind else "AI-assisted triage: cases ordered by AI priority (probability, lesion size, uncertainty and "
                         "disagreement between the two AI tools).")
    if not live:
        sub = f"Research simulation as {reader_name}: AI triage is shown."
    html(T.page_title("Worklist", sub))
    unread = wlo[wlo.status == "pending"]
    decided = int(wlo.status.isin(["accept", "edit", "reject", "manual"]).sum())
    deferred = int((wlo.status == "defer").sum())
    qc_unread = int((unread.qc_kind != "ok").sum())
    items = [("Unread", str(len(unread)), "plain")]
    if not blind:
        items += [("Urgent", str(int((unread.category == "URGENT").sum())), "urgent"),
                  ("AI tools disagree", str(int((unread.agree == "disagree").sum())), "caution")]
    items += [("QC issues", str(int((wlo.qc_kind != "ok").sum())), "caution" if qc_unread else "plain"),
              ("Decided", str(decided), "plain"), ("Deferred", str(deferred), "plain")]
    extra = (f"<span class='ra-sl-note'>{T.icon('lock', 13)} AI triage hidden (blind first read)</span>" if blind
             else f"<span class='ra-sl-note' title='Urgent rule: {T.esc(U.urgent_rule(cfg))}'>"
                  f"{T.icon('info', 13)} Urgent = {T.esc(U.urgent_rule(cfg))}</span>")
    html(T.status_line(items, extra))
    if qc_unread:
        html(T.note(f"<b>{qc_unread} unread case(s) failed automatic quality control.</b> Their AI output is "
                    "withheld because it is unreliable - read these cases fully.", "warn"))
    if live and policy_name != "universal" and not blind:
        changed = wlo[(wlo.category != wlo.universal_category)]
        if len(changed):
            html(T.note(f"Your decision policy ({T.esc(pol_ver)}) changes the AI category of {len(changed)} case(s) "
                        "compared with the universal policy.", "info"))

    t1, t2, t3, t4 = cols("toolbar", [2, 1.4, 1, 0.8], vertical_alignment="bottom")
    with t1:
        flt = st.segmented_control("Show", ["all", "unread", "decided"], default="all", key="wl_filter",
                                   format_func={"all": "All", "unread": "Unread", "decided": "Decided or deferred"}.get) \
            or "all"
    with t2:
        q = st.text_input("Find case", key="wl_search", placeholder="Case ID, e.g. lits_028")
    with t3:
        if blind:
            st.selectbox("Order", ["arrival"], key="wl_sort_blind", disabled=True,
                         format_func=lambda _: "Arrival order",
                         help="Blind first read always uses arrival order, so hidden AI results cannot steer your "
                              "reading sequence.")
        else:
            st.selectbox("Order", ["priority", "arrival"], key="wl_sort",
                         format_func={"priority": "AI priority", "arrival": "Arrival order"}.get)
    with t4:
        details = st.toggle("Details", key="wl_details", help="Show QC explanations and priority reasons in each row.")

    rows = wlo
    if flt == "unread":
        rows = rows[rows.status == "pending"]
    elif flt == "decided":
        rows = rows[rows.status != "pending"]
    if q:
        rows = rows[rows.case_id.str.contains(q.strip(), case=False, regex=False)]
    if rows.empty:
        html(T.note("No cases match." if (q or flt != "unread") else "Every case on this worklist has been read.",
                    "neutral"))
        return
    nxt = _next_unread(None)
    if nxt:
        html(f"<style>.st-key-row_{nxt} {{ box-shadow: inset 4px 0 0 {T.NAVY}; background:#F7F9FC; }}</style>")
    spec = [0.4, 1.1, 1.9, 2.5, 2.0, 1.16]
    with st.container(key="rowhead"):
        hs = st.columns(spec, vertical_alignment="center")
        for col, h in zip(hs, ["#", "Case", "Image / AI QC", "AI finding", "Status", ""]):
            col.markdown(f"<span class='ra-th'>{h}</span>", unsafe_allow_html=True)
    for r in rows.itertuples():
        hide = _hide_ai(r.case_id)
        with st.container(key=f"row_{r.case_id}"):
            c = st.columns(spec, vertical_alignment="center")
            c[0].markdown(f"<span class='ra-pos'>{r.position}</span>", unsafe_allow_html=True)
            c[1].markdown(f"<span class='ra-case'>{T.esc(r.case_id)}</span>", unsafe_allow_html=True)
            qc_html = T.qc_badge(r.qc_kind, r.qc_label, r.qc_detail)
            if details and r.qc_kind != "ok":
                qc_html += f"<div class='ra-cell-d'>{T.esc(r.qc_detail)}</div>"
            c[2].markdown(T.cell("QC", qc_html), unsafe_allow_html=True)
            if hide:
                ai_html = T.hidden_finding()
            else:
                ai_html = T.badge(r.category)
                if r.category not in ("QC FAIL", "DEFERRED"):
                    ai_html += f"<span class='ra-p'>{U.pct(r.tumour_probability)}</span>"
                    if r.agree == "disagree":
                        ai_html += f"<span class='ra-flag'>{T.icon('alert', 12)} tools disagree</span>"
                    elif r.agree == "differ":
                        ai_html += f"<span class='ra-flag'>{T.icon('alert', 12)} scores differ</span>"
                if details and r.reasons:
                    ai_html += f"<div class='ra-cell-d'>{T.esc(r.reasons)}</div>"
            c[3].markdown(T.cell("AI", ai_html), unsafe_allow_html=True)
            c[4].markdown(T.cell("Status", T.status_badge(r.status)), unsafe_allow_html=True)
            c[5].button("Open", key=f"open_{r.case_id}", type="primary" if r.case_id == nxt else "secondary",
                        on_click=_open, args=(r.case_id,), icon=":material/arrow_forward:", width="stretch")
    with st.expander("How to read this list"):
        html("<p><b>Image / AI QC</b> is operational: whether the AI output can be used at all. It is never a clinical "
             "finding or an urgency level. A failed check withholds the AI output for that case.</p>")
        for k, v in CATEGORY_HELP.items():
            if k != "QC FAIL":
                html(f"<p>{T.badge(k)} &nbsp;{T.esc(v)}</p>")
        html(f"<p>{T.badge('DEFERRED')} &nbsp;This reader's decision policy withholds the AI suggestion; read "
             f"independently.</p><p>{T.badge('QC FAIL')} &nbsp;Automatic QC failed; the AI output is withheld.</p>")
        st.caption(f"Urgent rule: {U.urgent_rule(cfg)}. Probabilities are calibrated patient-level probabilities that "
                   f"the CT shows at least one liver tumour. {U.UNCERTAINTY_DEFINITION}")


# =========================================================================== Read case
def _record_impression(key: tuple, value: str | None) -> None:
    if value in ("present", "absent"):
        ss.setdefault("impressions", {})[key] = {"value": int(value == "present"), "at": now_iso(),
                                                 "at_local": now_local()}


def _save(action: str, final: int | None, init, audit: dict, edit: dict | None = None) -> None:
    U.log_live(store, case, policy, init, action, final, edit=edit, audit=audit)
    ss.pop(f"revise_{case_id}", None)


VIEW_DEFAULTS = {"window": "liver", "zoom": 1.0, "pan_x": 0.0, "pan_y": 0.0, "ov_liver": True, "ov_lesions": True,
                 "ov_unc": False, "op_lesion": 35, "op_unc": 70, "wl_w": 150, "wl_l": 70}


def _reset_view(slice_key: str, start: int) -> None:
    ss[slice_key] = start
    for k, v in VIEW_DEFAULTS.items():
        ss[k] = v


def _step_slice(slice_key: str, delta: int, nums: list) -> None:
    i = nums.index(ss[slice_key]) if ss.get(slice_key) in nums else 0
    ss[slice_key] = nums[int(np.clip(i + delta, 0, len(nums) - 1))]


def _jump(slice_key: str, idx: int) -> None:
    ss[slice_key] = int(idx)


def view_read() -> None:
    prev_id, next_id = U.neighbours(order, case_id)
    n1, n2, n3 = st.container(key="casenav").columns([1, 3, 1], vertical_alignment="bottom")
    n1.button("Previous", key="prev_case", disabled=prev_id is None, on_click=_open, args=(prev_id,),
              icon=":material/arrow_back:", width="stretch", shortcut="P", help="Previous case (P)")
    ss.case_pick = case_id
    n2.selectbox("Case", order, key="case_pick", format_func=_case_label, on_change=lambda: _open(ss.case_pick))
    n3.button("Next", key="next_case", disabled=next_id is None, on_click=_open, args=(next_id,),
              icon=":material/arrow_forward:", width="stretch", shortcut="N", help="Next case (N)")

    g = record.get("geometry", {})
    nz = int((g.get("shape_view") or [0])[0])
    sp = (g.get("spacing_mm") or [0, 0, 0])
    imp = impressions.get((reader_id, case_id))
    reviewed = case_id in reviews
    ai_withheld = decision["decision"] != "show"
    need_impression = blind and not ai_withheld and imp is None and not reviewed and not ss.get(f"revise_{case_id}")
    hide = blind and imp is None and not reviewed
    state = ("Decided" if reviewed and reviews[case_id]["action"] != "defer" else "Deferred" if reviewed
             else "AI evidence revealed" if imp else "Unread")
    html(f"<div class='ra-casehead'><span class='ra-case'>{T.esc(case_id)}</span>"
         f"<span class='ra-meta'>CT abdomen · axial · {nz} slices · {float(sp[0]):.1f} mm slice spacing · "
         f"{float(sp[1]):.2f} mm pixels</span>{T.status_badge(row['status'])}</div>")
    html("<div class='ra-device-note'>" + T.note(
        "<b>Small screen.</b> This size is for checking the worklist and decisions only. Diagnostic interpretation "
        "requires a validated diagnostic display.", "warn") + "</div>")
    if blind and not ai_withheld:
        html(T.stepper(3 if (imp or reviewed) else 1, {1, 2} if (imp or reviewed) else set()))
    elif blind:
        html(T.note("The AI output is withheld for this case, so this is an independent read.", "info"))

    vpath = store.viewer_path(case_id)
    if not vpath.exists():
        html(T.note(f"Image data for {T.esc(case_id)} is missing in this run.", "warn"))
        return
    arrs = _load_viewer(str(vpath), vpath.stat().st_mtime)
    left, right = cols("read", [1.6, 1.0], gap="large")
    m, c, q = record["measurements"], record["confidence"], record["quality"]
    qc = U.qc_status(record)
    with left:
        _viewer(arrs, hide, nz, sp)
    with right:
        if need_impression:
            _impression_panel()
        else:
            if blind and imp is not None:
                ai_txt = ("AI suggestion withheld" if ai_withheld else
                          ("Liver tumour suspected" if decision["ai_label"] else "No liver tumour suspected"))
                html(T.comparison(imp["value"], imp.get("at_local"), None if ai_withheld else decision["ai_label"],
                                  ai_txt))
            _evidence_panel(m, c, q, qc, arrs)
        _decision_panel(m, c, qc, imp, need_impression)


def _viewer(arrs: dict, hide: bool, nz: int, sp) -> None:
    nums = [int(z) + 1 for z in arrs["z"]]            # 1-based slice numbers of the original volume
    slice_key = f"slice_{case_id}"
    start = nums[U.neutral_slice_index(arrs) if hide else U.key_slice_index(arrs, record)]
    hu = cfg["preprocess"]["hu_window"]
    for k, v in VIEW_DEFAULTS.items():      # defaults via session state only (no widget default + API clash)
        ss.setdefault(k, v)
    if ss.get(slice_key) not in nums:
        ss[slice_key] = start
    with st.container(key="viewer"):
        v1, v3 = cols("viewtools", [3, 1], vertical_alignment="bottom")
        with v1:
            opts = list(WINDOWS) + ["custom"]
            win = st.segmented_control("Window / level", opts, key="window", width="stretch",
                                       format_func=lambda k: f"{WINDOWS[k][0]}" if k in WINDOWS else "Custom") or "liver"
        v3.button("Reset view", key="reset_view", on_click=_reset_view, args=(slice_key, start),
                  icon=":material/restart_alt:", width="stretch")
        t1, t2, t3, v2 = cols("viewopts", [1, 1, 1, 1.2], vertical_alignment="center")
        show_liver = t1.toggle("AI liver", disabled=hide, key="ov_liver", help="AI liver outline")
        show_les = t2.toggle("AI lesions", disabled=hide, key="ov_lesions", help="AI lesion outline and fill")
        show_unc = t3.toggle("Uncertainty", disabled=hide, key="ov_unc", help="Voxel uncertainty map")
        with v2, st.popover("View options", icon=":material/tune:", width="stretch", help="Zoom, pan, overlay opacity, custom window"):
            st.slider("Zoom", 1.0, 3.0, step=0.25, key="zoom", format="%.2f×")
            st.slider("Pan left / right", -1.0, 1.0, step=0.1, key="pan_x")
            st.slider("Pan up / down", -1.0, 1.0, step=0.1, key="pan_y")
            st.slider("AI lesion fill opacity", 0, 80, step=5, key="op_lesion", format="%d%%", disabled=hide)
            st.slider("Uncertainty map opacity", 0, 90, step=5, key="op_unc", format="%d%%", disabled=hide)
            if win == "custom":
                st.slider("Window width (HU)", 40, 350, step=5, key="wl_w")
                st.slider("Window level (HU)", -50, 200, step=5, key="wl_l")
            st.caption(f"Viewer images store {hu[0]} to {hu[1]} HU; values outside show as black or white.")
        if win == "custom":
            w_, l_ = float(ss.get("wl_w", 150)), float(ss.get("wl_l", 70))
        else:
            w_, l_ = float(WINDOWS[win][1]), float(WINDOWS[win][2])
        sl = nums.index(ss[slice_key])
        kept = ss.get(f"kept_{reader_id}_{case_id}")
        img = U.render_slice(arrs, sl, show_liver=show_liver and not hide, show_lesions=show_les and not hide,
                             kept_lesions=kept, show_uncertainty=show_unc and not hide, show_reference=research,
                             upscale=max(1, 512 // arrs["ct"].shape[1]), contrast=U.window_to_u8(w_, l_, hu),
                             lesion_alpha=ss.get("op_lesion", 35) / 100.0, unc_alpha=ss.get("op_unc", 70) / 100.0,
                             zoom=float(ss.get("zoom", 1.0)), pan=(ss.get("pan_x", 0.0), ss.get("pan_y", 0.0)),
                             markers=True)
        z_abs = int(arrs["z"][sl])
        html(f"<div class='ra-vinfo'><span>Slice <b>{z_abs + 1}</b> / {nz}"
             f" · liver slab {int(arrs['z'][0]) + 1}-{int(arrs['z'][-1]) + 1}</span>"
             f"<span>W {w_:.0f} / L {l_:.0f} HU · zoom {float(ss.get('zoom', 1.0)):.2f}×</span></div>")
        st.image(img, width="stretch")
        s1, s2, s3 = cols("slicenav", [0.2, 1, 0.2], vertical_alignment="center")
        s1.button("Previous slice", key="slice_prev", on_click=_step_slice, args=(slice_key, -1, nums),
                  icon=":material/remove:", width="stretch", shortcut="K", help="Previous slice (K)")
        s2.select_slider("Slice", options=nums, key=slice_key, label_visibility="collapsed",
                         help="Drag, or use J / K. Arrow keys work after clicking the slider.")
        s3.button("Next slice", key="slice_next", on_click=_step_slice, args=(slice_key, 1, nums),
                  icon=":material/add:", width="stretch", shortcut="J", help="Next slice (J)")
        items = [("AI liver outline", T.OVERLAY["liver"], "line"), ("AI lesion", T.OVERLAY["lesion"], "fill"),
                 ("Uncertainty", T.OVERLAY["uncertainty"], "fill")]
        if research:
            items.append(("Reference (tumour solid, liver dotted)", T.OVERLAY["reference"], "dotted"))
        html(T.legend(items) + ("<div class='ra-vinfo'><span>AI overlays are hidden until you save your "
                                "impression.</span></div>" if hide else ""))
    st.caption("Radiological view: patient's right (R) on the left of the image, anterior (A) at the top. "
               "Keyboard: J / K next / previous slice, N / P next / previous case. Full screen: the expand icon on "
               "the image.")


def _impression_panel() -> None:
    with st.container(key="impression"):
        html(T.panel_title("1", "Independent read", "Record your own impression before any AI output is shown."))
        imp = st.segmented_control("Your impression", ["present", "absent"], key=f"imp_{case_id}",
                                   format_func={"present": "Tumour present", "absent": "No liver tumour"}.get,
                                   width="stretch")
        st.button("Save impression and reveal AI evidence", type="primary", disabled=imp is None, key="record_imp",
                  on_click=_record_impression, args=((reader_id, case_id), imp), icon=":material/lock_open:",
                  width="stretch")
        st.caption("Hidden until then: the AI prediction, segmentation overlays, uncertainty, the agreement between "
                   "the two AI tools and the case's AI priority.")


def _evidence_panel(m: dict, c: dict, q: dict, qc: dict, arrs: dict) -> None:
    with st.container(key="evidence"):
        html(T.panel_title("2", "AI evidence"))
        if not q["passed"]:
            html(T.withheld_card(f"AI output withheld: {qc['label'].lower()}",
                                 f"<p><b>What failed:</b> {T.esc(qc['detail'])}</p>"
                                 f"<p><b>What it means:</b> {T.esc(qc['consequence'])}</p>"))
        elif decision["decision"] != "show":
            html(T.withheld_card("AI suggestion withheld for this reader",
                                 f"<p>Your decision policy ({T.esc(pol_ver)}) withholds the suggestion because "
                                 f"{T.esc(U.humanize(decision['reason']).lower())}. Read this case independently.</p>"))
        else:
            ag = U.agreement(c.get("segmenter_probability"), c.get("vlm_probability"))
            positive = bool(decision["ai_label"])
            html(T.ai_summary(
                headline="AI suggestion: liver tumour suspected" if positive else "AI suggestion: no liver tumour suspected",
                positive=positive, p=c["tumour_probability"], threshold=policy.threshold,
                unc_phrase=U.uncertainty_phrase(c["uncertainty"]), unc_definition=U.UNCERTAINTY_DEFINITION,
                unc_level=c["uncertainty_level"], agreement_kind=ag["kind"], agreement_text=ag["text"], qc_ok=True,
                reasons=str(row["reasons"]).split("; ") if row["reasons"] else []))
        if q["passed"]:
            html(T.measures([("Liver volume", f"{m.get('liver_volume_ml', 0):.0f}", "mL"),
                             ("AI tumour volume", f"{m.get('predicted_tumour_volume_ml', 0):.1f}", "mL"),
                             ("AI lesions", f"{m.get('n_lesions', 0)}", ""),
                             ("Largest diameter", f"{m.get('largest_lesion_diameter_mm', 0):.0f}", "mm")]))
            if m.get("lesions"):
                les = pd.DataFrame(m["lesions"])
                j1, j2 = cols("decide", 2, vertical_alignment="bottom")
                pick = j1.selectbox("Go to lesion", list(les["id"]), key=f"lesjump_{case_id}",
                                    format_func=lambda i: f"Lesion {i} · {float(les.set_index('id').loc[i, 'volume_ml']):.1f} mL")
                ks = int(les.set_index("id").loc[pick, "key_slice"])
                j2.button("Show lesion slice", key="go_lesion", on_click=_jump,
                          args=(f"slice_{case_id}", int(arrs["z"][int(np.argmin(np.abs(arrs["z"] - ks)))]) + 1),
                          icon=":material/my_location:", width="stretch")
                st.dataframe(les[["id", "volume_ml", "max_axial_diameter_mm", "mean_hu", "mean_probability",
                                  "slice_range"]], hide_index=True, height=min(220, 40 + 36 * len(les)), column_config={
                    "id": "#", "volume_ml": st.column_config.NumberColumn("Volume (mL)", format="%.2f"),
                    "max_axial_diameter_mm": st.column_config.NumberColumn("Longest axial diameter (mm)", format="%.0f"),
                    "mean_hu": st.column_config.NumberColumn("Mean HU", format="%.0f"),
                    "mean_probability": st.column_config.NumberColumn("Mean voxel probability", format="%.2f"),
                    "slice_range": "Slices"})
        v = record.get("vlm") or {}
        with st.expander("Second-reader model (image and text)", icon=":material/neurology:"):
            if v.get("status") != "ok":
                st.caption(f"No second read for this case: {v.get('reason', v.get('status', 'not run'))}.")
            else:
                st.markdown(f"The second-reader model looked at {v.get('n_images')} axial slices through the predicted "
                            f"liver. Its patient-level probability of a liver tumour: **{U.pct(c.get('vlm_probability'))}"
                            f"** (calibrated; raw {U.pct(v['p_yes'])}).")
                if "vlm_rgb" in arrs:
                    st.image([im for im in arrs["vlm_rgb"]], width=96,
                             caption=[f"slice {int(z) + 1}" for z in arrs["vlm_z"]])
                    st.caption("Model input format: red = wide window, green = soft tissue, blue = narrow window.")
                if v.get("text"):
                    html(T.note("<b>Unverified model text</b>, quoted as written. It is never used for measurements or "
                                "for the suggestion.", "info"))
                    st.text(v["text"])
                for note_ in (v.get("guard") or {}).get("notes", []):
                    html(T.note("<b>Check:</b> " + T.esc(note_), "warn"))
        with st.expander("Uncertainty and calibration", icon=":material/insights:"):
            uc = c.get("uncertainty_components", {})
            st.markdown(f"- {U.UNCERTAINTY_DEFINITION}\n"
                        f"- This case: total {c['uncertainty']:.2f}; disagreement between the model's stochastic passes "
                        f"(epistemic) {uc.get('mutual_information', 0.0):.2f}; expected noise (aleatoric) "
                        f"{uc.get('expected_entropy', 0.0):.2f}.\n"
                        f"- Calibration: {c.get('calibrator', '-')}. Raw segmentation score {c['tumour_probability_raw']:.2f}; "
                        f"calibrated segmentation probability {U.pct(c.get('segmenter_probability'))}.")
        with st.expander("Models and versions", icon=":material/settings:"):
            model = record.get("model", {})
            st.markdown(f"- Model set **{mv['id']}** · decision policy **{pol_ver}** ({policy.summary()})\n"
                        f"- Segmentation model: {seg_name} ({model.get('backend', '-')} v{model.get('version', '-')}, "
                        f"{model.get('passes', '-')} passes)\n"
                        f"- Second-reader model: {(v.get('model') or {}).get('model_id', vlm_name)} · prompt "
                        f"{PROMPT_VERSION}\n- Session {ss.session_id}")
            if policy.notes:
                st.caption(" · ".join(policy.notes))


def _audit(c: dict, imp: dict | None) -> dict:
    ai_shown = decision["decision"] == "show"
    rec = ({"text": ("Liver tumour suspected" if decision["ai_label"] else "No liver tumour suspected")
                    + f" ({U.pct(c['tumour_probability'])}, threshold {U.pct(policy.threshold)})",
            "probability": c["tumour_probability"], "threshold": policy.threshold, "uncertainty": c["uncertainty"],
            "agreement": U.agreement(c.get("segmenter_probability"), c.get("vlm_probability"))["kind"],
            "category": row["category"]} if ai_shown else
           {"text": "AI suggestion withheld (" + ("QC failed" if not case["qc_passed"] else "reader policy") + ")",
            "category": row["category"]})
    return {"session_id": ss.session_id, "policy_version": pol_ver, "model_version": mv["id"], "recommendation": rec,
            "impression_at": (imp or {}).get("at"), "revealed_at": (imp or {}).get("at")}


def _decision_panel(m: dict, c: dict, qc: dict, imp: dict | None, need_impression: bool) -> None:
    prev = reviews.get(case_id)
    if reader_id in profiles:
        _simulation_panel()
        return
    if need_impression:
        return
    nxt = _next_unread(case_id)
    with st.container(key="decision"):
        if prev and not ss.get(f"revise_{case_id}"):
            act = U.STATUS_TEXT.get(prev["action"], prev["action"])
            fin = ("deferred for a second opinion" if prev.get("final_label") is None
                   else "tumour present" if prev["final_label"] else "no liver tumour")
            html(T.panel_title("3", "Decision saved"))
            html(T.note(f"<b>{T.esc(act)}.</b> Final finding: <b>{fin}</b> "
                        f"<span class='ra-muted'>· {T.esc(str(prev['timestamp'])[:16].replace('T', ' '))} UTC</span>"
                        + (f"<br><span class='ra-muted'>AI recommendation on screen: "
                           f"{T.esc((prev.get('recommendation') or {}).get('text', '-'))}</span>"
                           if prev.get("recommendation") else ""), "neutral"))
            if research:
                st.caption("Outcome vs reference: " + U.outcome_label(prev, case["gt_present"]))
            b1, b2 = cols("after", 2)
            b1.button("Next unread case", type="primary", key="after_next", disabled=nxt is None or nxt == case_id,
                      on_click=_open, args=(nxt,), icon=":material/arrow_forward:", width="stretch")
            b2.button("Open report", key="after_report", on_click=_go, args=("report",),
                      icon=":material/description:", width="stretch")
            st.button("Revise decision", key="revise", type="tertiary", icon=":material/edit:",
                      on_click=lambda: ss.__setitem__(f"revise_{case_id}", True))
            return
        ai_shown = decision["decision"] == "show"
        html(T.panel_title("3", "Reconcile and decide",
                           "Record your final finding. You decide; the AI suggestion is one piece of evidence."))
        fin = st.segmented_control("Final finding: liver tumour", list(FINDING), key=f"final_{case_id}",
                                   format_func=FINDING.get, width="stretch",
                                   help="Defer = no final finding yet; a second opinion is requested.")
        keep, vol, modified = None, None, False
        if ai_shown and fin == "present" and m.get("lesions"):
            with st.expander("Modify the AI lesions (optional)", icon=":material/edit:"):
                les_ids = [les["id"] for les in m["lesions"]]
                keep = st.multiselect("Lesions to keep", les_ids, default=les_ids, key=f"keep_{case_id}")
                ss[f"kept_{reader_id}_{case_id}"] = keep
                vol = sum(les["volume_ml"] for les in m["lesions"] if les["id"] in keep)
                modified = set(keep) != set(les_ids)
                st.caption(f"Tumour volume after your edit: {vol:.2f} mL (from the original-resolution lesion volumes).")
        note_ = st.text_input("Note (optional)", key=f"note_{case_id}")
        if fin is None:
            action, final = None, None
        elif fin == "defer":
            action, final = "defer", None
        elif not ai_shown:
            action, final = "manual", int(fin == "present")
        else:
            final = int(fin == "present")
            action = ("edit" if modified else "accept") if final == int(decision["ai_label"]) else "reject"
        if action == "defer":
            st.caption("Deferred: no final finding is recorded; the case is flagged for a second opinion.")
        if action in ("accept", "edit", "reject"):
            st.caption("Your final finding " + ("agrees with" if action != "reject" else "differs from")
                       + " the AI suggestion.")
        edit = {"kept_lesions": keep, "edited_volume_ml": round(vol, 3) if vol is not None else None, "note": note_} \
            if (modified or note_) else None
        st.button("Save decision", type="primary", key="save_decision", disabled=action is None,
                  on_click=_save, args=(action, final, (imp or {}).get("value"), _audit(c, imp), edit),
                  icon=":material/save:", width="stretch")


def _simulation_panel() -> None:
    prof = profiles[reader_id]
    with st.container(key="simulation"):
        html(T.panel_title(None, f"Research simulation · {prof.name}",
                           "A simulated read is not a clinical action. Batch simulation is in Reader adaptation."))
        if st.button("Simulate this read", key=f"sim_{case_id}", icon=":material/play_arrow:", width="stretch"):
            ev = U.simulate_and_log(store, prof, case, policy, salt=len(events_by_reader.get(reader_id, [])),
                                    audit={"session_id": ss.session_id, "policy_version": pol_ver,
                                           "model_version": mv["id"]})
            ss.last_sim = ev
            st.rerun()
        if "last_sim" in ss:
            ev = ss.pop("last_sim")
            txt = (f"Impression <b>{'tumour' if ev['initial_read'] else 'no tumour'}</b> → action "
                   f"<b>{T.esc(ev['action'])}</b> → final <b>{'tumour' if ev['final_label'] else 'no tumour'}</b>"
                   + (" (switched to the AI)" if ev.get("switched") else ""))
            html(T.note(txt, "neutral"))
            if research:
                st.caption("Outcome vs reference: " + U.outcome_label(ev, ev["gt_present"]))


# =========================================================================== Report (part of Read case)
def view_report() -> None:
    rv = reviews.get(case_id)
    c1, c2 = cols("reportbar", [4, 1.2], vertical_alignment="bottom")
    c1.markdown(T.page_title(f"Report draft · {case_id}", "A fixed template filled with the measured values. No language "
                                                         "model writes clinical text."), unsafe_allow_html=True)
    c2.button("Back to the case", key="report_back", on_click=_go, args=("read",), icon=":material/arrow_back:",
              width="stretch")
    if _hide_ai(case_id):
        html(T.note("The draft contains AI findings, so it is available after you save your independent impression "
                    "(blind first read).", "info"))
        return
    if rv:
        act = U.STATUS_TEXT.get(rv["action"], rv["action"])
        html(T.note(f"Reviewed by <b>{T.esc(reader_name)}</b>: {T.esc(act)}.", "neutral"))
    else:
        html(T.note("Draft - no decision saved yet.", "info"))
    review = {**rv, "reader_name": reader_name} if rv else None
    text = draft_report(record, review=review, policy={**decision, "personalised": policy.personalised})
    with st.container(key="report_doc"):
        st.code(text, language=None, wrap_lines=True)
    d1, d2 = cols("downloads", 2)
    d1.download_button("Download report (.txt)", text, file_name=f"{case_id}_report.txt", icon=":material/download:",
                       width="stretch")
    d2.download_button("Download case record (.json)", json.dumps(record, indent=2, default=str),
                       file_name=f"{case_id}.json", mime="application/json", icon=":material/data_object:",
                       width="stretch")


# =========================================================================== Research and quality area
def _chart_style(ax, title: str) -> None:
    ax.set_title(title, loc="left", color=T.NAVY, fontsize=12, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(T.BORDER_STRONG)
    ax.tick_params(colors=T.SLATE, labelsize=10)
    ax.xaxis.label.set_color(T.INK)
    ax.yaxis.label.set_color(T.INK)
    ax.grid(color="#E4EAF0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.figure.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")


def _policy_log_path() -> Path:
    return store.p("feedback") / "policy_changes.jsonl"


def _apply_policy() -> None:
    old = active[pkey]["policy"]
    new = U.current_policy(policy_name, reader_id, U.all_events(store, reader_options), cases, cfg)
    entry = {"timestamp": now_iso(), "session_id": ss.session_id, "reader_id": reader_id, "policy": policy_name,
             "from": U.policy_version(old), "to": U.policy_version(new), "threshold_from": old.threshold,
             "threshold_to": new.threshold, "withhold_from": old.defer_uncertainty, "withhold_to": new.defer_uncertainty,
             "n_decisions": new.n_feedback}
    p = _policy_log_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    active[pkey] = {"policy": new, "since": now_local()}


def _withhold_text(pol) -> str:
    return "never" if pol.defer_uncertainty is None else f"uncertainty ≥ {pol.defer_uncertainty:.2f}"


def view_adapt() -> None:
    evs = events_by_reader.get(reader_id, [])
    _research_header("Reader adaptation", f"How the decision policy for {reader_name} is learned from logged decisions.")
    html(T.note("The policy used for reading is <b>frozen for the session</b>. A proposed update is applied only when "
                "you press <i>Apply</i>, takes effect from the next case and is logged. Adaptation was evaluated "
                "offline with simulated readers (Model performance); it is not validated for clinical use.", "info"))
    p1, p2 = cols("policies", 2, gap="large")
    with p1:
        html(T.panel_title(None, "Active in this session") + T.measures([
            ("Version", pol_ver, ""), ("Suggestion threshold", U.pct(policy.threshold), ""),
            ("Withhold suggestion if", _withhold_text(policy), ""),
            ("Since", active[pkey]["since"], "")]))
        st.caption(("Personalised" if policy.personalised else "Universal") + f" · {policy.summary()}")
    with p2:
        html(T.panel_title(None, f"Proposed from {proposed.n_feedback} decisions with an impression") + T.measures([
            ("Version", U.policy_version(proposed), ""), ("Suggestion threshold", U.pct(proposed.threshold), ""),
            ("Withhold suggestion if", _withhold_text(proposed), "")]))
        if proposed.notes:
            st.caption(" · ".join(proposed.notes))
        same = U.policy_version(proposed) == pol_ver
        st.button("No change to apply" if same else "Apply from the next case", key="apply_policy",
                  type="secondary" if same else "primary", disabled=same, on_click=_apply_policy,
                  icon=":material/published_with_changes:", width="stretch")
    if reader_id in profiles:
        with st.container(key="simulation"):
            html(T.panel_title(None, f"Offline simulation · {reader_name}",
                               "Simulated reads of unread cases; inside this simulation the policy is refitted after "
                               "every read. The active policy above is not changed."))
            s1, s2 = cols("sim", 2, vertical_alignment="bottom")
            nmore = s1.number_input("Number of reads", 1, 40, 10, key="nmore")
            if s2.button(f"Simulate {int(nmore)} reads", key="simn", icon=":material/fast_forward:", width="stretch"):
                ev_list = list(evs)
                done = {e["case_id"] for e in ev_list}
                todo = [cid for cid in wl.case_id if cid not in done][: int(nmore)]
                for cid in todo:
                    pol = U.current_policy(policy_name, reader_id, {**events_by_reader, reader_id: ev_list}, cases, cfg)
                    ev_list.append(U.simulate_and_log(store, profiles[reader_id], cases[cid], pol, salt=len(ev_list),
                                                      audit={"session_id": ss.session_id,
                                                             "policy_version": U.policy_version(pol),
                                                             "model_version": mv["id"]}))
                ss.sim_done = len(todo)
                st.rerun()
            if "sim_done" in ss:
                html(T.note(f"Simulated {ss.pop('sim_done')} reads.", "neutral"))
    if evs:
        variant = policy_name if policy_name.startswith("reader") else "reader_threshold_defer"
        traj = U.policy_trajectory(evs, cases, cfg, variant)
        col1, col2 = cols("charts", 2, gap="large")
        with col1:
            fig, ax = plt.subplots(figsize=(5.8, 3.4))
            ax.step(traj.n_feedback, traj.threshold, where="post", color=T.NAVY, lw=2.2, label="Suggestion threshold")
            if traj.defer_uncertainty.notna().any():
                ax.step(traj.n_feedback, traj.defer_uncertainty, where="post", color=T.AMBER, lw=1.8, ls="--",
                        label="Withhold if uncertainty ≥")
            ax.axhline(cfg["policy"]["universal_threshold"], color=T.BORDER_STRONG, lw=1.2)
            ax.set_ylim(0, 1.02)
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            ax.set_xlabel("Decisions logged")
            _chart_style(ax, "Proposed policy as decisions accumulate")
            ax.legend(loc="lower right", frameon=False, fontsize=9)
            st.pyplot(fig, clear_figure=True)
        with col2:
            fig, ax = plt.subplots(figsize=(5.8, 3.4))
            cgrid = np.linspace(0.5, 1.0, 50)
            ax.plot(cgrid, proposed.trust(cgrid), color=T.NAVY, lw=2.2, label="Learned from decisions")
            dis = [e for e in evs if e.get("shown") and e.get("initial_read") is not None and e["ai_label"] is not None
                   and e.get("final_label") is not None and e["initial_read"] != e["ai_label"]]
            if dis:
                x = np.array([e["ai_confidence"] for e in dis])
                y = np.array([e["final_label"] == e["ai_label"] for e in dis], dtype=float)
                ax.scatter(x, y + np.random.default_rng(0).uniform(-0.03, 0.03, len(y)), s=40, color=T.TEAL,
                           edgecolor="#FFFFFF", linewidth=1.2, alpha=0.9, label="Disagreements (1 = followed the AI)")
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
        df["recommendation shown"] = [(e.get("recommendation") or {}).get("text", "") for e in evs]
        if research:
            df["reference"] = df["case_id"].map(lambda cid: int(cases[cid]["gt_present"]))
            df["outcome"] = [U.outcome_label(e, cases[e["case_id"]]["gt_present"]) for e in evs]
        ecols = [c for c in ["timestamp", "case_id", "action", "initial_read", "final_label", "recommendation shown",
                             "policy_version", "model_version", "session_id", "reference", "outcome", "source"]
                 if c in df.columns]
        with st.expander(f"All {len(evs)} logged decisions"):
            st.dataframe(df[ecols].iloc[::-1], hide_index=True, height=320)
    else:
        html(T.note("No decisions logged yet for this reader.", "neutral"))
    lp = _policy_log_path()
    if lp.exists():
        with st.expander("Policy change log"):
            st.dataframe(pd.DataFrame([json.loads(x) for x in lp.read_text().splitlines() if x.strip()]).iloc[::-1],
                         hide_index=True)


def view_perf() -> None:
    cal = R.get("calibration") or {}
    mc = R.get("model_card") or {}
    tools = cal.get("tools_test") or {}
    _research_header("Model performance", "Held-out LiTS17 test cases. Reader results come from simulated readers and "
                                          "are not evidence of clinical benefit.")
    t = []
    if cal.get("test"):
        a, b = cal["test"]["after"], cal["test"]["before"]
        t += [("Test AUROC", f"{a['auroc']:.2f}", ""), ("Calibration error (ECE)", f"{a['ece']:.3f}", f"raw {b['ece']:.3f}"),
              ("Brier score", f"{a['brier']:.3f}", f"raw {b['brier']:.3f}")]
    dice = (mc.get("val_dice_slicewise") or {}).get("tumour")
    if dice is not None:
        t.append(("Tumour Dice (validation)", f"{dice:.2f}", ""))
    if t:
        html(T.measures(t))
    if tools:
        names = {"segmenter": f"Segmentation model ({seg_name})", "medgemma": f"Second-reader model ({vlm_name})",
                 "fused": "Both combined"}
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
                  "over_reliance": "Over-reliance", "deferral_rate": "Withheld rate",
                  "actions_per_case": "Actions / case", "threshold": "Threshold"}
        tab = show[list(labels)].rename(columns=labels)
        for col in tab.columns[2:]:
            tab[col] = [("-" if pd.isna(x) else f"{x:.3f}".rstrip("0").rstrip(".")) for x in tab[col]]
        st.dataframe(tab, hide_index=True)
        st.caption("Harmful acceptance: an incorrect AI suggestion was accepted. Actions per case: accept 1, reject 2, "
                   "edit 3, independent read 4.")
    figs = [(n, title) for n, title in (
        ("ai_tools.png", "Segmentation model vs second-reader model"), ("calibration_reliability.png", "Calibration"),
        ("segmentation_quality.png", "Segmentation"), ("triage_lift.png", "Worklist triage"),
        ("risk_coverage.png", "Selective prediction"), ("policy_comparison.png", "Policies × readers"),
        ("benefit_by_reader.png", "Benefit by reader (RQ1)"), ("learning_curve.png", "Learning curve (RQ3)"))
        if (store.p("figures") / n).exists()]
    if figs:
        st.markdown("#### Figures")
        for i in range(0, len(figs), 2):
            rowc = cols(f"figs{i}", 2, gap="large")
            for col, (n, title) in zip(rowc, figs[i:i + 2]):
                with col:
                    html(f"<div class='ra-panel-t'>{T.esc(title)}</div>")
                    st.image(str(store.p("figures") / n), width="stretch")


def view_audit() -> None:
    _research_header(f"Audit trail · {case_id}", "Decisions with the recommendation on screen at the time, and every "
                                                 "tool the agent called.")
    if _hide_ai(case_id):
        html(T.note("This case is unread in a blind first read. Its audit trail contains AI output and opens after your "
                    "impression is saved.", "info"))
        return
    evs = [e for r_ in reader_options for e in FeedbackStore(store.p("feedback")).events(r_) if e["case_id"] == case_id]
    if evs:
        st.markdown("#### Decisions on this case")
        st.dataframe(pd.DataFrame([{"Time (UTC)": e["timestamp"], "Reader": reader_names.get(e["reader_id"], e["reader_id"]),
                                    "Action": U.STATUS_TEXT.get(e["action"], e["action"]),
                                    "Impression": e.get("initial_read"), "Final": e.get("final_label"),
                                    "Recommendation on screen": (e.get("recommendation") or {}).get("text", ""),
                                    "Policy": e.get("policy_version") or e.get("policy"),
                                    "Model set": e.get("model_version"), "Session": e.get("session_id")}
                                   for e in evs]), hide_index=True)
    tp = store.trace_path(case_id)
    if tp.exists():
        trace = json.loads(tp.read_text())
        st.markdown("#### Agent first read")
        html(T.measures([("Planner", str(trace["planner"]), ""), ("Status", str(trace["status"]), ""),
                         ("Guardrail overrides", str(trace.get("n_guardrail_overrides", 0)), "")]))
        st.dataframe(pd.DataFrame([{"Step": s_["step"], "Tool": s_["tool"], "Planned by": s_["planned_by"],
                                    "Reason": s_.get("reason", ""), "ms": s_.get("duration_ms"),
                                    "Result": json.dumps(s_.get("summary"), default=str)[:160],
                                    "Guardrail": s_.get("guardrail") or ""} for s_ in trace["steps"]]),
                     hide_index=True)
    from radagent.data.lits import locate

    ct_ok = Path(locate(str(store.manifest().set_index("case_id").loc[case_id, "ct_path"]), cfg)).exists()
    torch_ok = importlib.util.find_spec("torch") is not None
    can_recompute = ct_ok and (torch_ok or cfg["segmenter"]["backend"] == "plugin")
    with st.expander("Run the agent again on this case", icon=":material/replay:"):
        c1, c2 = cols("rerunrow", [1.4, 1], vertical_alignment="bottom")
        planner_choice = c1.segmented_control("Planner", ["rule", "llm"], default="rule", key="planner",
                                              format_func={"rule": "Rules", "llm": "Open LLM"}.get) or "rule"
        recompute = c1.toggle("Recompute from the CT file (runs both AI models now)", value=False,
                              disabled=not can_recompute, key="recompute",
                              help="Needs the LiTS CT on this machine and PyTorch (a GPU for the second-reader model). "
                                   + ("" if can_recompute else ("CT file not found here. " if not ct_ok else "")
                                      + ("PyTorch not installed. " if not torch_ok else "")))
        if c2.button("Run both models now" if recompute else "Re-run the agent", key="rerun_btn",
                     icon=":material/replay:", width="stretch"):
            from radagent.pipeline import run_single_case

            with st.spinner("Running the AI models (the first call loads them) ..." if recompute else "Agent running ..."):
                _, tr2 = run_single_case(cfg, case_id, planner_choice, reader_policy=policy, recompute=recompute)
            ss["rerun_trace"] = tr2
            if recompute:
                _load_run_cached.clear()
                st.rerun()
    if "rerun_trace" in ss and ss["rerun_trace"]["case_id"] == case_id:
        tr2 = ss["rerun_trace"]
        html(T.note(f"Re-run with the {T.esc(tr2['planner'])} planner: {T.esc(tr2['status'])}, "
                    f"{tr2.get('n_guardrail_overrides', 0)} guardrail override(s).", "neutral"))
        st.dataframe(pd.DataFrame([{"Step": s_["step"], "Tool": s_["tool"], "Planned by": s_["planned_by"],
                                    "Reason": s_.get("reason", ""), "Guardrail": s_.get("guardrail") or "",
                                    "Result": json.dumps(s_.get("summary"), default=str)[:160]}
                                   for s_ in tr2["steps"]]), hide_index=True)


def view_about() -> None:
    _research_header("About and intended use", "What the assistant does, what it is not for, and who decides.")
    c1, c2 = cols("about", 2, gap="large")
    with c1:
        st.markdown(f"""
#### Intended use (research)
- Organise a liver-CT worklist and pre-read each case: liver and tumour segmentation by **{seg_name}**,
  measurements at the original resolution, a calibrated patient-level probability and its uncertainty.
- A second opinion from **{vlm_name}**; cases where the two AI tools disagree are flagged.
- Study how individual readers use the AI and whether reader-specific policies help - offline, with logged
  and versioned changes only.
""")
    with c2:
        st.markdown("""
#### Not intended for
- Clinical diagnosis or patient management - a research prototype on public, de-identified data.
- Replacing a radiologist's read: **the radiologist decides every finding**.
- Measurements from the language model: all sizes come from the segmentation; model text is unverified.
""")
    st.markdown("""
#### Supported devices
- **Diagnostic interpretation:** not validated on any display. A future clinical evaluation must use validated
  diagnostic displays in the reading room.
- **Desktop and laptop screens:** the full research workflow.
- **Tablets and phones:** worklist, status and saved decisions only; the case view shows a warning on small screens.
""")
    st.markdown("#### How a case is pre-read")
    html(T.flow([("CT volume", "LiTS NIfTI, voxel spacing", "data"), ("Segmentation model", seg_name, "ai"),
                 ("Measure", "volumes, diameters", "step"), ("Quality control", "operational checks", "step"),
                 ("Second-reader model", vlm_name, "ai"), ("Calibrated probability", "+ uncertainty", "step"),
                 ("Triage", "AI priority (hidden in blind read)", "step"), ("Independent read", "your impression", "human"),
                 ("Reconcile and decide", "accept / modify / reject / defer", "human"), ("Report draft", "fixed template", "step")]))
    st.caption(f"Run directory: {store.root}")


{"worklist": view_worklist, "read": view_read, "report": view_report, "adapt": view_adapt, "perf": view_perf,
 "audit": view_audit, "about": view_about}[ss.view]()
