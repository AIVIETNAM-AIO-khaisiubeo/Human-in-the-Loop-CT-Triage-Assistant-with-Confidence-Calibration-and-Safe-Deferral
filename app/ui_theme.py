"""Visual design of the reading interface: colour tokens, CSS and small HTML building blocks.

Design rationale and the radiologist journey behind it: ``docs/UX_DESIGN.md``.
Everything here is plain Python returning strings (no Streamlit import), so it can be unit-tested.
All dynamic text is HTML-escaped.
"""
from __future__ import annotations

import html
import math

# --------------------------------------------------------------------------- tokens
NAVY = "#0B2545"
NAVY_2 = "#13315C"
INK = "#16324F"
MUTED = "#4A5D72"
TEAL = "#0F766E"
TEAL_DARK = "#0B5F59"
TEAL_SOFT = "#8FD3CB"
TEAL_TINT = "#E3F3F1"
SURFACE = "#F3F7F9"
BORDER = "#D3DDE6"
BORDER_STRONG = "#8A9BAD"
VIEWER_BG = "#0A1628"

# viewer overlay colours (RGB), chosen to be distinct on grey CT and for common colour-vision deficiencies
OVERLAY = {
    "liver": (64, 201, 186),      # soft teal outline
    "lesion": (255, 170, 51),     # amber
    "selected": (255, 230, 120),  # light yellow
    "reference": (245, 247, 250), # near-white
    "uncertainty": (199, 125, 255),  # violet
}

# triage category -> (css modifier, label, svg icon); shape + word + colour, never colour alone
_ICONS = {
    "triangle": '<path d="M8 1.6 15.2 14.4H.8z"/>',
    "diamond": '<path d="M8 .8 15.2 8 8 15.2.8 8z"/>',
    "half": '<circle cx="8" cy="8" r="6.2" fill="none" stroke="currentColor" stroke-width="2"/>'
            '<path d="M8 1.8a6.2 6.2 0 0 1 0 12.4z"/>',
    "ring": '<circle cx="8" cy="8" r="5.6" fill="none" stroke="currentColor" stroke-width="2.4"/>',
    "person": '<circle cx="8" cy="4.8" r="3"/><path d="M2.4 15c0-3.3 2.5-5.6 5.6-5.6s5.6 2.3 5.6 5.6z"/>',
    "square": '<rect x="1.6" y="1.6" width="12.8" height="12.8" rx="2.2" fill="none" stroke="currentColor" '
              'stroke-width="2"/><rect x="7" y="4" width="2" height="5.4" rx=".6"/><rect x="7" y="10.9" '
              'width="2" height="2" rx=".6"/>',
    "check": '<path d="M6.3 12.6 2 8.3l1.6-1.6 2.7 2.7 6.1-6.1L14 4.9z"/>',
    "alert": '<path d="M8 1.2 15.4 14.6H.6z"/><rect x="7.1" y="5.6" width="1.8" height="4.6" fill="#fff"/>'
             '<rect x="7.1" y="11.3" width="1.8" height="1.8" fill="#fff"/>',
    "info": '<circle cx="8" cy="8" r="7"/><rect x="7.1" y="7" width="1.8" height="5" fill="#fff"/>'
            '<rect x="7.1" y="3.8" width="1.8" height="1.8" fill="#fff"/>',
    "dot": '<circle cx="8" cy="8" r="4"/>',
}

CATEGORY_STYLE = {
    "URGENT": ("urgent", "Urgent", "triangle"),
    "SUSPICIOUS": ("suspicious", "Suspicious", "diamond"),
    "UNCERTAIN": ("uncertain", "Uncertain", "half"),
    "LIKELY NORMAL": ("normal", "Likely normal", "ring"),
    "DEFERRED": ("deferred", "Independent read", "person"),
    "QC FAIL": ("qc", "QC failed", "square"),
}

STATUS_STYLE = {
    "pending": ("pending", "Not yet read"),
    "accept": ("done", "Accepted"),
    "edit": ("done", "Edited"),
    "reject": ("done", "Rejected"),
    "manual": ("done", "Read independently"),
}


def esc(x) -> str:
    return html.escape(str(x), quote=True)


def icon(name: str, size: int = 16, label: str | None = None) -> str:
    aria = f'role="img" aria-label="{esc(label)}"' if label else 'aria-hidden="true"'
    return (f'<svg class="ra-ico" width="{size}" height="{size}" viewBox="0 0 16 16" fill="currentColor" '
            f'{aria} focusable="false">{_ICONS[name]}</svg>')


# --------------------------------------------------------------------------- building blocks
def badge(category: str, large: bool = False) -> str:
    mod, label, ico = CATEGORY_STYLE.get(category, ("qc", category.title(), "dot"))
    return (f'<span class="ra-badge ra-{mod}{" ra-badge-lg" if large else ""}">{icon(ico)}'
            f'<span>{esc(label)}</span></span>')


def status_badge(status: str) -> str:
    mod, label = STATUS_STYLE.get(status, ("pending", status))
    ico = "check" if mod == "done" else "dot"
    return f'<span class="ra-status ra-status-{mod}">{icon(ico, 14)}<span>{esc(label)}</span></span>'


def chip(text: str, kind: str = "plain", title: str | None = None) -> str:
    t = f' title="{esc(title)}"' if title else ""
    return f'<span class="ra-chip ra-chip-{kind}"{t}>{text}</span>'


def header(reader_name: str | None, policy_text: str | None, research_label: str, run_name: str) -> str:
    logo = ('<svg class="ra-logo" viewBox="0 0 40 40" aria-hidden="true" focusable="false">'
            '<rect width="40" height="40" rx="10" fill="#13315C"/>'
            '<path d="M9 21c0-6.6 5.6-11 12.6-11 5.4 0 9.4 2.6 9.4 7.4 0 6.4-6.4 12.6-13.8 12.6C12.4 30 9 26.4 9 21z"'
            ' fill="none" stroke="#8FD3CB" stroke-width="2.4"/>'
            '<circle cx="23.5" cy="18.5" r="3.2" fill="#8FD3CB"/></svg>')
    return f"""
<header class="ra-top" aria-label="Application header">
  <div class="ra-brand">{logo}
    <div><div class="ra-title">ReaderAdapt · Liver CT</div>
    <div class="ra-sub">AI first-read assistant for radiologists</div></div>
  </div>
  <div class="ra-top-meta">
    {chip(f'Reading as <b>{esc(reader_name)}</b>', 'reader', 'Change in Session settings') if reader_name else ''}
    {chip(esc(policy_text), 'policy', 'Decision policy for this reader') if policy_text else ''}
    {chip(icon('info', 14) + ' ' + esc(research_label), 'research')}
  </div>
  <div class="ra-top-note ra-mobile-only">Session settings (reader, policy, worklist): open the menu at the top left.</div>
  <span class="ra-sr">Run {esc(run_name)}</span>
</header>"""


def kpi_strip(items: list[tuple[str, str, str, str]]) -> str:
    """items: (label, value, hint, modifier) -> responsive grid of tiles."""
    tiles = "".join(
        f'<div class="ra-kpi ra-kpi-{esc(mod)}" role="listitem"><div class="ra-kpi-label">{esc(label)}</div>'
        f'<div class="ra-kpi-value">{value}</div><div class="ra-kpi-hint">{esc(hint)}</div></div>'
        for label, value, hint, mod in items)
    return f'<div class="ra-kpis" role="list">{tiles}</div>'


def stepper(current: int, done: set[int]) -> str:
    names = ["Your impression", "AI suggestion", "Your decision", "Report"]
    items = []
    for i, n in enumerate(names, start=1):
        state = "done" if i in done else ("current" if i == current else "todo")
        mark = icon("check", 14) if state == "done" else str(i)
        cur = ' aria-current="step"' if state == "current" else ""
        items.append(f'<li class="ra-step ra-step-{state}"{cur}><span class="ra-step-n">{mark}</span>'
                     f'<span class="ra-step-t">{esc(n)}</span></li>')
    return f'<ol class="ra-steps" aria-label="Reading steps">{"".join(items)}</ol>'


def meter(p: float, threshold: float | None = None, label: str = "Tumour probability") -> str:
    p = 0.0 if p is None or (isinstance(p, float) and math.isnan(p)) else float(p)
    pct = max(0.0, min(100.0, 100.0 * p))
    thr = ""
    if threshold is not None:
        thr = (f'<span class="ra-meter-thr" style="left:{100 * threshold:.1f}%" '
               f'title="suggestion threshold {threshold:.2f}"></span>')
    return (f'<div class="ra-meter" role="img" aria-label="{esc(label)} {p:.2f}'
            + (f', threshold {threshold:.2f}' if threshold is not None else "") + '">'
            f'<div class="ra-meter-fill" style="width:{pct:.1f}%"></div>{thr}</div>')


def confidence_words(uncertainty_level: str) -> str:
    return {"low": "The AI is confident about this case.",
            "medium": "The AI is moderately confident - check the key slices.",
            "high": "The AI is uncertain - please read this case carefully."}.get(uncertainty_level,
                                                                                 "Confidence not available.")


def ai_summary(*, headline: str, positive: bool, p: float, threshold: float, unc_level: str, unc: float,
               seg_p: float | None, vlm_p: float | None, disagree: bool | None, reasons: list[str],
               provenance: str) -> str:
    tone = "pos" if positive else "neg"
    agree = ""
    if vlm_p is not None:
        if disagree:
            agree = (f'<div class="ra-agree ra-agree-no">{icon("alert", 16)}<span><b>The two AI tools disagree.</b> '
                     f'Segmenter {seg_p:.2f} vs MedGemma {vlm_p:.2f} - this case was moved up the worklist.</span></div>')
        else:
            agree = (f'<div class="ra-agree ra-agree-yes">{icon("check", 16)}<span><b>Both AI tools agree.</b> '
                     f'Segmenter {seg_p:.2f} · MedGemma {vlm_p:.2f}</span></div>')
    why = "".join(f"<li>{esc(r)}</li>" for r in reasons if r)
    return f"""
<section class="ra-ai ra-ai-{tone}" aria-label="AI suggestion">
  <div class="ra-ai-kicker">AI suggestion</div>
  <div class="ra-ai-head">{esc(headline)}</div>
  <div class="ra-ai-prob"><span>Calibrated probability <b>{p:.2f}</b></span>
     <span class="ra-ai-thr">suggestion threshold {threshold:.2f}</span></div>
  {meter(p, threshold)}
  <div class="ra-ai-conf ra-unc-{esc(unc_level)}">{esc(confidence_words(unc_level))}
     <span class="ra-muted">Uncertainty {unc:.2f} ({esc(unc_level)}).</span></div>
  {agree}
  {f'<div class="ra-why"><div class="ra-why-t">Why this priority</div><ul>{why}</ul></div>' if why else ''}
  <div class="ra-prov">{icon('info', 14)} {esc(provenance)}</div>
</section>"""


def withheld_card(reason: str, policy_text: str) -> str:
    return f"""
<section class="ra-ai ra-ai-withheld" aria-label="AI suggestion withheld">
  <div class="ra-ai-kicker">AI suggestion withheld</div>
  <div class="ra-ai-head">Please read this case independently</div>
  <p>Reason: {esc(reason)}.</p>
  <div class="ra-prov">{icon('info', 14)} Policy: {esc(policy_text)}</div>
</section>"""


def tiles(items: list[tuple[str, str, str]]) -> str:
    """Measurement tiles: (label, value, unit)."""
    t = "".join(f'<div class="ra-tile"><div class="ra-tile-l">{esc(label)}</div>'
                f'<div class="ra-tile-v">{esc(value)}<span>{esc(unit)}</span></div></div>' for label, value, unit in items)
    return f'<div class="ra-tiles">{t}</div>'


def legend(items: list[tuple[str, tuple[int, int, int], str]]) -> str:
    """(label, rgb, style) with style 'line' | 'fill' | 'dotted'."""
    out = []
    for label, rgb, kind in items:
        col = "rgb({},{},{})".format(*rgb)
        out.append(f'<span class="ra-leg"><i class="ra-leg-{kind}" style="--c:{col}"></i>{esc(label)}</span>')
    return f'<div class="ra-legend" aria-label="Overlay legend">{"".join(out)}</div>'


def flow(steps: list[tuple[str, str, str]]) -> str:
    """Responsive pipeline diagram: (title, detail, kind) with kind in {'data', 'ai', 'human', 'step'}."""
    li = "".join(f'<li class="ra-flow-{esc(k)}"><b>{esc(t)}</b><span>{esc(d)}</span></li>' for t, d, k in steps)
    return f'<ol class="ra-flow" aria-label="How a case flows through the assistant">{li}</ol>'


def note(text: str, kind: str = "info") -> str:
    ico = {"info": "info", "warn": "alert", "ok": "check"}[kind]
    return f'<div class="ra-note ra-note-{kind}" role="note">{icon(ico, 16)}<div>{text}</div></div>'


def section_title(title: str, subtitle: str | None = None) -> str:
    sub = f'<p class="ra-sec-sub">{esc(subtitle)}</p>' if subtitle else ""
    return f'<div class="ra-sec"><h2 class="ra-sec-t">{esc(title)}</h2>{sub}</div>'


def case_card_main(case_id: str, suggestion: str, reasons: str) -> str:
    return (f'<div class="ra-case-id">{esc(case_id)}</div><div class="ra-case-sug">{esc(suggestion)}</div>'
            + (f'<div class="ra-case-why">{esc(reasons)}</div>' if reasons else ""))


def case_card_ai(p: float, threshold: float, unc_level: str, vlm_p: float | None, agree: str) -> str:
    v = ""
    if vlm_p is not None and not (isinstance(vlm_p, float) and math.isnan(vlm_p)):
        v = (f'<div class="ra-case-vlm">MedGemma {vlm_p:.2f} · '
             + (f'<span class="ra-disagree">{icon("alert", 13)} disagrees</span>' if agree == "disagree"
                else "agrees") + "</div>")
    return (f'<div class="ra-case-p"><span>Probability <b>{p:.2f}</b></span><span class="ra-muted">'
            f'{esc(unc_level)} uncertainty</span></div>{meter(p, threshold)}{v}')


# --------------------------------------------------------------------------- CSS
CSS = f"""
:root {{
  --ra-navy:{NAVY}; --ra-navy-2:{NAVY_2}; --ra-ink:{INK}; --ra-muted:{MUTED};
  --ra-teal:{TEAL}; --ra-teal-dark:{TEAL_DARK}; --ra-teal-soft:{TEAL_SOFT}; --ra-teal-tint:{TEAL_TINT};
  --ra-surface:{SURFACE}; --ra-border:{BORDER}; --ra-border-strong:{BORDER_STRONG}; --ra-viewer:{VIEWER_BG};
  --ra-radius:14px;
}}
/* ---------- page */
[data-testid="stApp"], [data-testid="stAppViewContainer"] {{ background:#FFFFFF; color:var(--ra-ink); }}
[data-testid="stMainBlockContainer"], .block-container {{ max-width:1440px; padding:3.4rem 2rem 4rem; }}
[data-testid="stMarkdownContainer"] {{ font-size:1.0625rem; line-height:1.55; }}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{ font-size:1.0625rem; }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{ font-size:0.9rem; color:var(--ra-muted); }}
h1, h2, h3, h4 {{ color:var(--ra-navy) !important; letter-spacing:-0.01em; }}
[data-testid="stMarkdownContainer"] h3 {{ font-size:1.2rem; font-weight:700; padding:0.4rem 0 0.2rem; }}
[data-testid="stMarkdownContainer"] h4 {{ font-size:1.08rem; font-weight:700; padding:0.3rem 0 0.1rem; }}
header[data-testid="stHeader"] {{ background:rgba(255,255,255,0.94); border-bottom:1px solid var(--ra-border); }}
a {{ color:var(--ra-teal-dark); }}
.ra-muted {{ color:var(--ra-muted); }}
.ra-sr {{ position:absolute; width:1px; height:1px; overflow:hidden; clip:rect(0 0 0 0); white-space:nowrap; }}
.ra-ico {{ flex:none; vertical-align:-2px; }}
.ra-mobile-only {{ display:none; }}

/* ---------- sidebar */
[data-testid="stSidebar"] {{ background:var(--ra-surface); border-right:1px solid var(--ra-border); }}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{ font-size:1rem; }}
.ra-side-t {{ font-size:1.1rem; font-weight:700; color:var(--ra-navy); margin:0.2rem 0 0.1rem; }}
.ra-side-s {{ font-size:0.92rem; color:var(--ra-muted); margin-bottom:0.4rem; }}

/* ---------- header band */
.ra-top {{ background:linear-gradient(120deg, var(--ra-navy) 0%, var(--ra-navy-2) 100%); color:#fff;
  border-radius:var(--ra-radius); padding:16px 22px; display:flex; flex-wrap:wrap; gap:12px 24px;
  align-items:center; justify-content:space-between; margin-bottom:14px; }}
.ra-brand {{ display:flex; gap:14px; align-items:center; }}
.ra-logo {{ width:44px; height:44px; flex:none; }}
.ra-title {{ font-size:1.5rem; font-weight:700; letter-spacing:-0.01em; line-height:1.2; color:#fff; }}
.ra-sub {{ font-size:0.95rem; color:var(--ra-teal-soft); }}
.ra-top-meta {{ display:flex; flex-wrap:wrap; gap:8px; }}
.ra-top-note {{ flex-basis:100%; font-size:0.9rem; color:#D9E6F2; }}
.ra-chip {{ display:inline-flex; align-items:center; gap:6px; border-radius:999px; padding:6px 12px;
  font-size:0.92rem; line-height:1.2; border:1px solid rgba(255,255,255,0.28); color:#fff; }}
.ra-chip-reader b {{ color:#fff; }}
.ra-chip-policy {{ color:#D9E6F2; }}
.ra-chip-research {{ border-color:#FDE7B0; color:#FDE7B0; font-weight:600; }}

/* ---------- status strip */
.ra-kpis {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(124px, 1fr)); gap:12px; margin:4px 0 14px; }}
.ra-kpi {{ border:1px solid var(--ra-border); border-radius:12px; padding:12px 14px; background:#fff;
  border-top:4px solid var(--ra-teal); }}
.ra-kpi-label {{ font-size:0.92rem; color:var(--ra-muted); font-weight:600; }}
.ra-kpi-value {{ font-size:1.75rem; font-weight:700; color:var(--ra-navy); line-height:1.2;
  font-variant-numeric:tabular-nums; }}
.ra-kpi-value span {{ font-size:1rem; color:var(--ra-muted); font-weight:600; margin-left:2px; }}
.ra-kpi-hint {{ font-size:0.88rem; color:var(--ra-muted); }}
.ra-kpi-urgent {{ border-top-color:#B42318; }} .ra-kpi-disagree {{ border-top-color:#B54708; }}
.ra-kpi-deferred {{ border-top-color:#1D4E89; }} .ra-kpi-done {{ border-top-color:#067647; }}

/* ---------- navigation (segmented control) */
.st-key-nav {{ margin:2px 0 10px; }}
.st-key-nav [data-testid="stWidgetLabel"] {{ display:none; }}
.st-key-nav [role="radiogroup"], .st-key-nav [data-testid="stButtonGroup"] {{ flex-wrap:wrap; gap:6px; }}
.st-key-nav button {{ min-height:50px; padding:0 18px; font-size:1.02rem; font-weight:600; border-radius:10px !important;
  color:var(--ra-navy); background:#fff; border:1.5px solid var(--ra-border) !important; }}
.st-key-nav [data-testid="stButtonGroup"] button[aria-checked="true"] {{
  background:var(--ra-navy) !important; color:#fff !important; border-color:var(--ra-navy) !important; }}

/* binary clinical answers (impression, final finding, edit) span the card: two big, equal targets */
[class*="st-key-imp_"], [class*="st-key-fin_"], [class*="st-key-pres_"] {{ width:100% !important; }}
[class*="st-key-imp_"] [role="radiogroup"], [class*="st-key-fin_"] [role="radiogroup"],
[class*="st-key-pres_"] [role="radiogroup"] {{ display:flex; width:100%; }}
[class*="st-key-imp_"] [role="radiogroup"] button, [class*="st-key-fin_"] [role="radiogroup"] button,
[class*="st-key-pres_"] [role="radiogroup"] button {{ flex:1 1 0; min-height:56px; font-size:1.05rem; }}

/* ---------- section titles */
.ra-sec {{ margin:6px 0 8px; }}
.ra-sec-t {{ font-size:1.45rem !important; font-weight:700; margin:0 !important; padding:0 !important; }}
.ra-sec-sub {{ color:var(--ra-muted); margin:2px 0 0; font-size:1rem; }}

/* ---------- controls: large touch targets + visible focus */
[data-testid="stButton"] button, [data-testid="stDownloadButton"] button, [data-testid="stPopover"] button,
[data-testid="stFormSubmitButton"] button {{
  min-height:48px; padding:0.55rem 1.1rem; border-radius:10px; font-size:1rem; font-weight:600; width:100%; }}
[data-testid="stBaseButton-primary"] {{ background:var(--ra-teal); border-color:var(--ra-teal); color:#fff; }}
[data-testid="stBaseButton-primary"]:hover {{ background:var(--ra-teal-dark); border-color:var(--ra-teal-dark); color:#fff; }}
[data-testid="stBaseButton-secondary"] {{ background:#fff; color:var(--ra-navy); border:1.5px solid var(--ra-border-strong); }}
[data-testid="stBaseButton-secondary"]:hover {{ background:var(--ra-teal-tint); color:var(--ra-teal-dark); border-color:var(--ra-teal); }}
[data-testid="stButtonGroup"] button {{ min-height:48px; font-size:1rem; padding:0 18px; color:var(--ra-navy); }}
[data-testid="stButtonGroup"] button[aria-checked="true"] {{ background:var(--ra-teal-tint); color:var(--ra-teal-dark);
  border-color:var(--ra-teal); font-weight:700; }}
div[data-baseweb="select"] > div, [data-testid="stSelectbox"] [role="group"], [data-testid="stMultiSelect"] [role="group"],
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input {{ min-height:48px; font-size:1rem; }}
[data-testid="stTooltipIcon"] button {{ min-width:28px; min-height:28px; display:inline-flex; align-items:center;
  justify-content:center; }}
[data-testid="stSidebarCollapseButton"] button, [data-testid="stExpandSidebarButton"],
[data-testid="stMainMenu"] button {{ min-width:40px; min-height:40px; }}
[data-testid="stSlider"] [role="slider"] {{ width:26px; height:26px; }}
[data-testid="stCheckbox"] label {{ min-height:44px; display:flex; align-items:center; }}
[data-testid="stRadio"] label {{ min-height:40px; align-items:center; }}
[data-testid="stExpander"] details {{ border-radius:12px; border:1px solid var(--ra-border); }}
[data-testid="stExpander"] summary {{ min-height:52px; font-weight:600; font-size:1.02rem; color:var(--ra-navy); }}
button:focus-visible, [role="tab"]:focus-visible, [role="slider"]:focus-visible, input:focus-visible,
summary:focus-visible, a:focus-visible, [tabindex]:focus-visible {{
  outline:3px solid var(--ra-teal) !important; outline-offset:2px !important; }}
@media (prefers-reduced-motion: reduce) {{ *, *::before, *::after {{ animation:none !important; transition:none !important; scroll-behavior:auto !important; }} }}

/* ---------- cards (bordered containers) */
[data-testid="stVerticalBlockBorderWrapper"], [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"][style*="border"] {{
  border-radius:var(--ra-radius) !important; }}
.ra-card-t {{ font-size:1.15rem; font-weight:700; color:var(--ra-navy); margin:0 0 4px; }}
.ra-card-s {{ color:var(--ra-muted); font-size:0.98rem; margin:0 0 6px; }}

/* ---------- badges */
.ra-badge {{ display:inline-flex; align-items:center; gap:6px; border-radius:8px; padding:4px 10px;
  font-weight:700; font-size:0.92rem; line-height:1.25; border:1px solid currentColor; }}
.ra-badge-lg {{ font-size:1rem; padding:6px 12px; }}
.ra-urgent {{ color:#B42318; background:#FEF3F2; }} .ra-suspicious {{ color:#B54708; background:#FFF6E5; }}
.ra-uncertain {{ color:#5B3CC4; background:#F3F1FF; }} .ra-normal {{ color:#067647; background:#ECF8F1; }}
.ra-deferred {{ color:#1D4E89; background:#EAF1FA; }} .ra-qc {{ color:#344054; background:#EEF1F4; }}
.ra-status {{ display:inline-flex; align-items:center; gap:6px; font-size:0.92rem; font-weight:600; }}
.ra-status-pending {{ color:var(--ra-muted); }} .ra-status-done {{ color:#067647; }}
.ra-rank {{ font-size:1.5rem; font-weight:700; color:var(--ra-navy); font-variant-numeric:tabular-nums; line-height:1.1; }}
.ra-rank span {{ font-size:0.85rem; color:var(--ra-muted); font-weight:600; display:block; }}

/* ---------- worklist cards */
.ra-case-id {{ font-size:1.15rem; font-weight:700; color:var(--ra-navy); }}
.ra-case-sug {{ font-size:1rem; color:var(--ra-ink); }}
.ra-case-why {{ font-size:0.92rem; color:var(--ra-muted); margin-top:2px; }}
.ra-case-p {{ display:flex; justify-content:space-between; gap:8px; font-size:0.95rem; }}
.ra-case-vlm {{ font-size:0.92rem; color:var(--ra-muted); margin-top:4px; }}
.ra-disagree {{ color:#B54708; font-weight:700; display:inline-flex; gap:4px; align-items:center; }}
.st-key-wl_first [data-testid="stVerticalBlockBorderWrapper"], .ra-first {{ border-color:var(--ra-teal) !important; }}

/* ---------- meter */
.ra-meter {{ position:relative; height:10px; border-radius:999px; background:#E4EBF1; margin:6px 0 4px; overflow:visible; }}
.ra-meter-fill {{ height:100%; border-radius:999px; background:var(--ra-teal); }}
.ra-meter-thr {{ position:absolute; top:-4px; width:2px; height:18px; background:var(--ra-navy); border-radius:1px; }}

/* ---------- stepper */
ol.ra-steps {{ list-style:none; display:flex !important; flex-direction:row !important; flex-wrap:wrap; gap:8px;
  padding:0 !important; margin:4px 0 12px !important; }}
.ra-steps .ra-step {{ margin:0 !important; padding:7px 18px 7px 8px !important; }}
.ra-step {{ display:flex; align-items:center; gap:8px; border-radius:999px;
  border:1.5px solid var(--ra-border); color:var(--ra-muted); font-weight:600; font-size:0.98rem; background:#fff; }}
.ra-step-n {{ width:28px; height:28px; border-radius:50%; display:inline-flex; align-items:center; justify-content:center;
  background:#E4EBF1; color:var(--ra-navy); font-weight:700; }}
.ra-step-current {{ border-color:var(--ra-teal); color:var(--ra-navy); background:var(--ra-teal-tint); }}
.ra-step-current .ra-step-n {{ background:var(--ra-teal); color:#fff; }}
.ra-step-done {{ color:var(--ra-ink); }}
.ra-step-done .ra-step-n {{ background:#067647; color:#fff; }}

/* ---------- viewer */
.st-key-viewer [data-testid="stImage"], .st-key-viewer [data-testid="stImageContainer"] {{ background:var(--ra-viewer);
  border-radius:12px; padding:8px; }}
.st-key-viewer img {{ border-radius:6px; max-width:100%; height:auto; image-rendering:auto; }}
.st-key-viewer [data-testid="stImageCaption"], .st-key-viewer [data-testid="stImageCaption"] p {{ color:#C9D6E3 !important;
  font-size:0.92rem; padding:4px 4px 0; }}
.ra-legend {{ display:flex; flex-wrap:wrap; gap:6px 18px; font-size:0.95rem; color:var(--ra-ink); margin:6px 0 2px; }}
.ra-leg {{ display:inline-flex; align-items:center; gap:8px; }}
.ra-leg i {{ display:inline-block; width:22px; height:12px; border-radius:3px; background:#2B3A4A; }}
.ra-leg-line {{ box-shadow:inset 0 0 0 2.5px var(--c); }}
.ra-leg-fill {{ background:var(--c) !important; }}
.ra-leg-dotted {{ background:repeating-linear-gradient(90deg, var(--c) 0 4px, #2B3A4A 4px 7px) !important; }}

/* ---------- AI suggestion */
.ra-ai {{ border:1px solid var(--ra-border); border-left:6px solid var(--ra-teal); border-radius:var(--ra-radius);
  padding:14px 18px; background:#fff; margin-bottom:10px; }}
.ra-ai-pos {{ border-left-color:#B54708; }} .ra-ai-neg {{ border-left-color:#067647; }}
.ra-ai-withheld {{ border-left-color:#1D4E89; background:#F7FAFD; }}
.ra-ai-kicker {{ text-transform:uppercase; letter-spacing:0.06em; font-size:0.8rem; font-weight:700; color:var(--ra-muted); }}
.ra-ai-head {{ font-size:1.45rem; font-weight:700; color:var(--ra-navy); line-height:1.25; margin:2px 0 6px; }}
.ra-ai-prob {{ display:flex; justify-content:space-between; flex-wrap:wrap; gap:6px; font-size:1rem; }}
.ra-ai-thr {{ color:var(--ra-muted); font-size:0.92rem; }}
.ra-ai-conf {{ margin-top:6px; font-size:1rem; }}
.ra-unc-high {{ color:#5B3CC4; font-weight:600; }}
.ra-agree {{ display:flex; gap:8px; align-items:flex-start; border-radius:10px; padding:8px 12px; margin-top:10px; font-size:0.98rem; }}
.ra-agree .ra-ico {{ margin-top:3px; }}
.ra-agree-yes {{ background:#ECF8F1; color:#05603A; }}
.ra-agree-no {{ background:#FFF6E5; color:#93370D; }}
.ra-why {{ margin-top:10px; }}
.ra-why-t {{ font-weight:700; font-size:0.95rem; color:var(--ra-navy); }}
.ra-why ul {{ margin:2px 0 0 1.1rem; padding:0; }} .ra-why li {{ font-size:0.95rem; }}
.ra-prov {{ margin-top:10px; font-size:0.88rem; color:var(--ra-muted); display:flex; gap:6px; align-items:flex-start; }}
.ra-prov .ra-ico {{ margin-top:3px; }}

/* ---------- flow diagram */
ol.ra-flow {{ list-style:none; display:flex !important; flex-direction:row; flex-wrap:wrap; gap:10px 28px;
  padding:0 !important; margin:8px 0 14px !important; }}
.ra-flow li {{ position:relative; margin:0 !important; padding:10px 14px !important; border-radius:12px; min-width:140px;
  border:1.5px solid var(--ra-border-strong); background:#fff; display:flex; flex-direction:column; gap:2px; }}
.ra-flow li b {{ color:var(--ra-navy); font-size:1rem; }}
.ra-flow li span {{ color:var(--ra-muted); font-size:0.88rem; }}
.ra-flow li:not(:last-child)::after {{ content:"→"; position:absolute; right:-22px; top:50%; transform:translateY(-50%);
  color:var(--ra-muted); font-weight:700; }}
.ra-flow-data {{ background:var(--ra-surface) !important; }}
.ra-flow-ai {{ border-color:var(--ra-teal) !important; }}
.ra-flow-human {{ background:var(--ra-teal-tint) !important; border-color:var(--ra-teal) !important; }}

/* ---------- tiles, notes */
.ra-tiles {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(120px, 1fr)); gap:10px; margin:8px 0; }}
.ra-tile {{ background:var(--ra-surface); border-radius:12px; padding:10px 12px; }}
.ra-tile-l {{ font-size:0.9rem; color:var(--ra-muted); font-weight:600; }}
.ra-tile-v {{ font-size:1.45rem; font-weight:700; color:var(--ra-navy); font-variant-numeric:tabular-nums; }}
.ra-tile-v span {{ font-size:0.95rem; color:var(--ra-muted); margin-left:4px; font-weight:600; }}
.ra-note {{ display:flex; gap:10px; align-items:flex-start; border-radius:12px; padding:10px 14px; margin:6px 0 10px;
  font-size:1rem; }}
.ra-note .ra-ico {{ margin-top:4px; }}
.ra-note-info {{ background:#EAF1FA; color:#1D3B63; }} .ra-note-warn {{ background:#FFF6E5; color:#7A2E0E; }}
.ra-note-ok {{ background:#ECF8F1; color:#05603A; }}

/* ---------- report */
.st-key-report_doc [data-testid="stCode"] pre, .st-key-report_doc pre {{ background:#FFFFFF !important;
  border:1px solid var(--ra-border); border-radius:12px; white-space:pre-wrap !important; word-break:break-word; }}
.st-key-report_doc code {{ font-size:0.95rem !important; color:var(--ra-ink) !important; line-height:1.55; }}

/* ---------- responsive */
@media (max-width: 900px) {{
  /* previous / next become arrow buttons next to the case picker; their text stays for screen readers */
  .st-key-casenav [data-testid="stHorizontalBlock"] {{ flex-wrap:nowrap !important; gap:8px !important; }}
  .st-key-casenav [data-testid="stColumn"] {{ min-width:0 !important; width:auto !important; flex:0 0 56px !important; }}
  .st-key-casenav [data-testid="stColumn"]:nth-child(2) {{ flex:1 1 auto !important; }}
  .st-key-prev_case [data-testid="stMarkdownContainer"], .st-key-next_case [data-testid="stMarkdownContainer"] {{
    position:absolute; width:1px; height:1px; overflow:hidden; clip:rect(0 0 0 0); white-space:nowrap; }}
  .st-key-prev_case button, .st-key-next_case button {{ padding:0 !important; }}
}}
@media (max-width: 640px) {{
  [data-testid="stMainBlockContainer"], .block-container {{ padding:3.2rem 0.85rem 4rem; }}
  .ra-top {{ padding:14px 16px; }}
  .ra-title {{ font-size:1.25rem; }}
  .ra-mobile-only {{ display:block; }}
  .ra-kpis {{ grid-template-columns:repeat(auto-fit, minmax(104px, 1fr)); gap:6px; }}
  .ra-kpi {{ padding:8px 10px; border-top-width:3px; }}
  .ra-kpi-label {{ font-size:0.82rem; line-height:1.2; min-height:2.05em; }}
  .ra-kpi-value {{ font-size:1.35rem; }}
  .ra-kpi-hint {{ display:none; }}
  .ra-tiles {{ grid-template-columns:repeat(auto-fit, minmax(96px, 1fr)); gap:6px; }}
  ol.ra-flow {{ flex-direction:column; gap:22px; }}
  .ra-flow li:not(:last-child)::after {{ content:"↓"; right:auto; left:24px; top:auto; bottom:-21px; transform:none; }}
  .ra-tile-v {{ font-size:1.2rem; }}
  .ra-ai-head {{ font-size:1.25rem; }}
  .ra-sec-t {{ font-size:1.25rem !important; }}
  .st-key-nav button {{ padding:0 12px; font-size:0.98rem; flex:1 1 auto; }}
  .ra-step {{ padding:6px 10px 6px 6px; font-size:0.92rem; }}
}}
"""


# minimum column width (px) per reflowing row created with streamlit_app.cols(name, ...)
REFLOW = {"read": 440, "overlays": 136, "decide": 170, "after": 170, "sim": 170, "reportbar": 240,
          "downloads": 220, "charts": 360, "figs": 380, "about": 320, "onboard": 320, "rerunrow": 260,
          "wlfilter": 240}


def reflow_css() -> str:
    out = []
    for name, px in REFLOW.items():
        sel = f'[class*="st-key-rx_{name}"]' if name == "figs" else f".st-key-rx_{name}"
        hb = (f'{sel} > [data-testid="stHorizontalBlock"], {sel} > * > [data-testid="stHorizontalBlock"]')
        col = ", ".join(f'{h.strip()} > [data-testid="stColumn"]' for h in hb.split(","))
        out.append(f"{hb} {{ flex-wrap:wrap !important; }}")
        out.append(f"{col} {{ flex:1 1 {px}px !important; min-width:min({px}px, 100%) !important; width:auto !important; }}")
    # worklist cards: four blocks that wrap as the card gets narrow
    out.append('[class*="st-key-wl_"] [data-testid="stHorizontalBlock"] { flex-wrap:wrap !important; }')
    out.append('[class*="st-key-wl_"] [data-testid="stColumn"] { flex:1 1 180px !important; '
               'min-width:min(180px, 100%) !important; width:auto !important; }')
    out.append('[class*="st-key-wl_"] [data-testid="stColumn"]:nth-child(2) { flex-grow:2 !important; }')
    out.append('.st-key-rx_read > * > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child, '
               '.st-key-rx_read > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child '
               '{ flex-grow:1.2 !important; }')
    return "\n".join(out)


def style_tag() -> str:
    return f"<style>{CSS}\n{reflow_css()}</style>"
