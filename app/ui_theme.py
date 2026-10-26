"""Visual design of the reading interface: semantic colour tokens, CSS and small HTML components.

Design rationale, the radiologist journey and the design-review response: ``docs/UX_DESIGN.md``.
Everything here is plain Python returning strings (no Streamlit import), so it can be unit-tested.
All dynamic text is HTML-escaped.

Colour has one meaning each (never decoration):
navy = application structure · teal = active controls and primary actions (never "the AI is right") ·
red = a validated critical rule (urgent) · amber = caution / review needed · slate = neutral and technical.
"""
from __future__ import annotations

import html
import math

# --------------------------------------------------------------------------- semantic tokens
NAVY = "#102D52"       # structure: header, navigation, headings
NAVY_2 = "#163A66"
INK = "#1B2B40"        # body text
SLATE = "#526579"      # neutral / technical / secondary text
TEAL = "#0F766E"       # active controls and primary actions only
TEAL_DARK = "#0B5F59"
TEAL_TINT = "#E3F3F1"
TEAL_SOFT = "#8FD3CB"  # accent on navy
RED = "#B42318"        # validated critical rule (urgent)
RED_BG = "#FEF3F2"
AMBER = "#B54708"      # caution, review needed, QC issue, disagreement
AMBER_BG = "#FFF6E5"
SLATE_BG = "#F1F4F7"
SURFACE = "#F5F7FA"
BORDER = "#D6DEE6"
BORDER_STRONG = "#8A9BAD"
VIEWER_BG = "#0A1628"
# legacy aliases used elsewhere in the app
MUTED = SLATE

# viewer overlays (RGB): mutually distinguishable and visible on grey CT (checked in tests/test_app.py)
OVERLAY = {
    "liver": (64, 201, 186),      # teal outline
    "lesion": (255, 170, 51),     # amber outline + adjustable fill
    "selected": (255, 230, 120),
    "reference": (245, 247, 250),
    "uncertainty": (199, 125, 255),  # violet, adjustable opacity
}

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
    "lock": '<rect x="2.5" y="7" width="11" height="8" rx="1.6"/><path d="M5 7V5a3 3 0 0 1 6 0v2" fill="none" '
            'stroke="currentColor" stroke-width="1.8"/>',
    "dot": '<circle cx="8" cy="8" r="3.4"/>',
    "clock": '<circle cx="8" cy="8" r="6.4" fill="none" stroke="currentColor" stroke-width="1.8"/>'
             '<path d="M7.2 4h1.6v4.2l2.6 1.6-.8 1.3-3.4-2z"/>',
}

# AI category (an AI output, never a diagnosis) -> (css modifier, label, icon)
CATEGORY_STYLE = {
    "URGENT": ("urgent", "Urgent (rule)", "triangle"),
    "SUSPICIOUS": ("caution", "AI: tumour suspected", "diamond"),
    "UNCERTAIN": ("caution-outline", "AI: uncertain", "half"),
    "LIKELY NORMAL": ("neutral", "AI: no tumour suspected", "ring"),
    "DEFERRED": ("neutral-outline", "AI withheld (policy)", "person"),
    "QC FAIL": ("neutral-outline", "AI withheld (QC)", "square"),
}

STATUS_STYLE = {
    "pending": ("unread", "Unread", "dot"),
    "impression": ("progress", "Impression recorded", "clock"),
    "accept": ("done", "Decided · agreed with AI", "check"),
    "edit": ("done", "Decided · AI lesions modified", "check"),
    "reject": ("done", "Decided · differs from AI", "check"),
    "manual": ("done", "Decided · independent read", "check"),
    "defer": ("caution", "Deferred · second opinion", "clock"),
}


def esc(x) -> str:
    return html.escape(str(x), quote=True)


def icon(name: str, size: int = 16, label: str | None = None) -> str:
    aria = f'role="img" aria-label="{esc(label)}"' if label else 'aria-hidden="true"'
    return (f'<svg class="ra-ico" width="{size}" height="{size}" viewBox="0 0 16 16" fill="currentColor" '
            f'{aria} focusable="false">{_ICONS[name]}</svg>')


def pct(p) -> str:
    return "-" if p is None or (isinstance(p, float) and math.isnan(p)) else f"{100 * float(p):.0f}%"


# --------------------------------------------------------------------------- components
def badge(category: str) -> str:
    mod, label, ico = CATEGORY_STYLE.get(category, ("neutral", category.title(), "dot"))
    return f'<span class="ra-badge ra-b-{mod}">{icon(ico, 14)}<span>{esc(label)}</span></span>'


def status_badge(status: str) -> str:
    mod, label, ico = STATUS_STYLE.get(status, ("unread", status, "dot"))
    return f'<span class="ra-status ra-s-{mod}">{icon(ico, 14)}<span>{esc(label)}</span></span>'


def qc_badge(kind: str, label: str, detail: str = "") -> str:
    t = f' title="{esc(detail)}"' if detail else ""
    if kind == "ok":
        return f'<span class="ra-qc ra-qc-ok"{t}>{icon("check", 14)}<span>{esc(label)}</span></span>'
    return f'<span class="ra-qc ra-qc-fail"{t}>{icon("square", 14)}<span>{esc(label)}</span></span>'


def hidden_finding() -> str:
    return (f'<span class="ra-hidden">{icon("lock", 14)}<span>Hidden until your impression</span></span>')


def chip(text: str, kind: str = "plain", title: str | None = None) -> str:
    t = f' title="{esc(title)}"' if title else ""
    return f'<span class="ra-chip ra-chip-{kind}"{t}>{text}</span>'


def header(reader_name: str | None, session_id: str | None, model_id: str | None, mode_label: str | None) -> str:
    """Slim navy session bar: identity of the app, the reader, the session and the model set."""
    logo = ('<svg class="ra-logo" viewBox="0 0 40 40" aria-hidden="true" focusable="false">'
            '<rect width="40" height="40" rx="9" fill="#163A66"/>'
            '<path d="M9 21c0-6.6 5.6-11 12.6-11 5.4 0 9.4 2.6 9.4 7.4 0 6.4-6.4 12.6-13.8 12.6C12.4 30 9 26.4 9 21z"'
            ' fill="none" stroke="#8FD3CB" stroke-width="2.4"/><circle cx="23.5" cy="18.5" r="3.2" fill="#8FD3CB"/></svg>')
    meta = []
    if reader_name:
        meta.append(chip(f"Reader <b>{esc(reader_name)}</b>", "meta"))
    if mode_label:
        meta.append(chip(esc(mode_label), "meta"))
    if session_id:
        meta.append(chip(f"Session {esc(session_id)}", "meta", "Reading-session identifier (recorded with every decision)"))
    if model_id:
        meta.append(chip(f"Model set {esc(model_id)}", "meta", "Segmentation model, second-reader model and "
                                                                "calibration used in this session"))
    return f"""
<header class="ra-top" aria-label="Session">
  <div class="ra-brand">{logo}<span class="ra-title">ReaderAdapt · Liver CT</span>
    {chip(icon('info', 13) + ' Research prototype · not for clinical use', 'research')}</div>
  <div class="ra-top-meta">{''.join(meta)}</div>
</header>"""


def status_line(items: list[tuple[str, str, str]], note_html: str = "") -> str:
    """Slim queue summary: (label, value, modifier) separated by middots."""
    parts = "".join(f'<span class="ra-sl ra-sl-{esc(m)}"><span class="ra-sl-l">{esc(lbl)}</span> '
                    f'<b>{esc(v)}</b></span>' for lbl, v, m in items)
    return f'<div class="ra-statusline" role="status">{parts}{note_html}</div>'


def stepper(current: int, done: set[int]) -> str:
    names = ["Independent read", "Reveal AI evidence", "Reconcile and decide"]
    items = []
    for i, n in enumerate(names, start=1):
        state = "done" if i in done else ("current" if i == current else "todo")
        mark = icon("check", 12) if state == "done" else str(i)
        cur = ' aria-current="step"' if state == "current" else ""
        items.append(f'<li class="ra-step ra-step-{state}"{cur}><span class="ra-step-n">{mark}</span>'
                     f'<span class="ra-step-t">{esc(n)}</span></li>')
    return f'<ol class="ra-steps" aria-label="Reading steps">{"".join(items)}</ol>'


def meter(p: float, threshold: float | None = None, label: str = "Tumour probability") -> str:
    p = 0.0 if p is None or (isinstance(p, float) and math.isnan(p)) else float(p)
    pc = max(0.0, min(100.0, 100.0 * p))
    thr = ""
    if threshold is not None:
        thr = (f'<span class="ra-meter-thr" style="left:{100 * threshold:.1f}%" '
               f'title="suggestion threshold {pct(threshold)}"></span>')
    return (f'<div class="ra-meter" role="img" aria-label="{esc(label)} {pct(p)}'
            + (f', threshold {pct(threshold)}' if threshold is not None else "") + '">'
            f'<div class="ra-meter-fill" style="width:{pc:.1f}%"></div>{thr}</div>')


def comparison(impression: int | None, impression_time: str | None, ai_label: int | None,
               ai_text: str) -> str:
    """Your independent impression and the AI suggestion side by side, after the reveal."""
    if impression is None:
        mine = '<span class="ra-muted">Not recorded</span>'
    else:
        mine = f"<b>{'Tumour present' if impression else 'No liver tumour'}</b>"
        if impression_time:
            mine += f' <span class="ra-muted">· {esc(impression_time)}</span>'
    rel = ""
    if impression is not None and ai_label is not None:
        rel = ('<span class="ra-rel ra-rel-same">Same as your impression</span>' if impression == ai_label
               else '<span class="ra-rel ra-rel-diff">Differs from your impression</span>')
    return (f'<div class="ra-compare"><div><div class="ra-k">Your impression</div>{mine}</div>'
            f'<div><div class="ra-k">AI suggestion</div><b>{esc(ai_text)}</b></div>{rel}</div>')


def ai_summary(*, headline: str, positive: bool, p: float, threshold: float, unc_phrase: str,
               unc_definition: str, unc_level: str, agreement_kind: str, agreement_text: str,
               qc_ok: bool, reasons: list[str]) -> str:
    agree_cls = {"disagree": "ra-ag-caution", "differ": "ra-ag-caution", "agree": "ra-ag-neutral"}.get(
        agreement_kind, "ra-ag-neutral")
    agree_ico = "alert" if agreement_kind in ("disagree", "differ") else "info"
    why = "".join(f"<li>{esc(r)}</li>" for r in reasons if r)
    return f"""
<section class="ra-ai" aria-label="AI summary">
  <div class="ra-ai-head">{esc(headline)}</div>
  <div class="ra-ai-target">Patient-level probability that this CT shows at least one liver tumour (calibrated):
    <b>{pct(p)}</b> <span class="ra-muted">· suggestion threshold {pct(threshold)}</span></div>
  {meter(p, threshold, "Patient-level tumour probability")}
  <div class="ra-ai-unc ra-unc-{esc(unc_level)}" title="{esc(unc_definition)}">{esc(unc_phrase)}
    <span class="ra-muted">· 0 = certain, 1 = a 50/50 call; high ≥ 0.85</span></div>
  <div class="ra-ag {agree_cls}">{icon(agree_ico, 15)}<span>{esc(agreement_text)}</span></div>
  <div class="ra-valid">{icon('check' if qc_ok else 'alert', 14)}
    {'AI output valid: automatic quality checks passed.' if qc_ok else 'AI output not valid: quality checks failed.'}</div>
  {f'<details class="ra-why"><summary>Why this case is ranked here</summary><ul>{why}</ul></details>' if why else ''}
</section>"""


def withheld_card(title: str, body_html: str) -> str:
    return (f'<section class="ra-ai ra-ai-withheld" aria-label="AI suggestion withheld">'
            f'<div class="ra-ai-head">{esc(title)}</div><div class="ra-ai-body">{body_html}</div></section>')


def measures(items: list[tuple[str, str, str]]) -> str:
    """One compact measurement group (label, value, unit) instead of separate tiles."""
    cells = "".join(f'<div><dt>{esc(lbl)}</dt><dd>{esc(v)}<span>{esc(u)}</span></dd></div>' for lbl, v, u in items)
    return f'<dl class="ra-measures">{cells}</dl>'


def legend(items: list[tuple[str, tuple[int, int, int], str]]) -> str:
    out = []
    for label, rgb, kind in items:
        col = "rgb({},{},{})".format(*rgb)
        out.append(f'<span class="ra-leg"><i class="ra-leg-{kind}" style="--c:{col}"></i>{esc(label)}</span>')
    return f'<div class="ra-legend" aria-label="Overlay legend">{"".join(out)}</div>'


def flow(steps: list[tuple[str, str, str]]) -> str:
    li = "".join(f'<li class="ra-flow-{esc(k)}"><b>{esc(t)}</b><span>{esc(d)}</span></li>' for t, d, k in steps)
    return f'<ol class="ra-flow" aria-label="How a case flows through the assistant">{li}</ol>'


def note(text: str, kind: str = "info") -> str:
    ico = {"info": "info", "warn": "alert", "neutral": "check", "critical": "alert"}[kind]
    return f'<div class="ra-note ra-note-{kind}" role="note">{icon(ico, 16)}<div>{text}</div></div>'


def page_title(title: str, subtitle: str | None = None) -> str:
    sub = f'<p class="ra-sec-sub">{esc(subtitle)}</p>' if subtitle else ""
    return f'<div class="ra-sec"><h1 class="ra-sec-t">{esc(title)}</h1>{sub}</div>'


section_title = page_title


def panel_title(n: str | None, title: str, sub: str | None = None) -> str:
    num = f'<span class="ra-pn">{esc(n)}</span>' if n else ""
    s = f'<div class="ra-panel-s">{esc(sub)}</div>' if sub else ""
    return f'<div class="ra-panel-t">{num}{esc(title)}</div>{s}'


def cell(label: str, content_html: str) -> str:
    """Worklist cell; the label is visible only when the row wraps on a narrow screen."""
    return f'<div class="ra-cell"><span class="ra-cell-l">{esc(label)}</span>{content_html}</div>'


# --------------------------------------------------------------------------- CSS
CSS = f"""
:root {{
  --ra-navy:{NAVY}; --ra-navy-2:{NAVY_2}; --ra-ink:{INK}; --ra-slate:{SLATE};
  --ra-teal:{TEAL}; --ra-teal-dark:{TEAL_DARK}; --ra-teal-tint:{TEAL_TINT}; --ra-teal-soft:{TEAL_SOFT};
  --ra-red:{RED}; --ra-red-bg:{RED_BG}; --ra-amber:{AMBER}; --ra-amber-bg:{AMBER_BG}; --ra-slate-bg:{SLATE_BG};
  --ra-surface:{SURFACE}; --ra-border:{BORDER}; --ra-border-strong:{BORDER_STRONG}; --ra-viewer:{VIEWER_BG};
  --ra-radius:10px;
}}
/* ---------- page + type scale: body 16, labels 14-15, secondary 13-14, case id 19, headings 22 */
[data-testid="stApp"], [data-testid="stAppViewContainer"] {{ background:#FFFFFF; color:var(--ra-ink); }}
[data-testid="stMainBlockContainer"], .block-container {{ max-width:1600px; padding:3.1rem 1.5rem 3rem; }}
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {{ font-size:1rem; line-height:1.5; }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{ font-size:0.875rem; color:var(--ra-slate); }}
[data-testid="stWidgetLabel"] p {{ font-size:0.9375rem !important; font-weight:500; color:var(--ra-ink); }}
h1, h2, h3, h4 {{ color:var(--ra-navy) !important; letter-spacing:-0.005em; }}
[data-testid="stMarkdownContainer"] h4 {{ font-size:1.0625rem; font-weight:700; padding:0.2rem 0 0.1rem; }}
header[data-testid="stHeader"] {{ background:rgba(255,255,255,0.96); border-bottom:1px solid var(--ra-border); }}
a {{ color:var(--ra-teal-dark); }}
.ra-muted {{ color:var(--ra-slate); font-weight:400; }}
.ra-ico {{ flex:none; vertical-align:-2px; }}
.ra-device-note {{ display:none; }}

/* ---------- sidebar */
[data-testid="stSidebar"] {{ background:var(--ra-surface); border-right:1px solid var(--ra-border); }}
.ra-side-t {{ font-size:0.9375rem; font-weight:700; color:var(--ra-navy); margin:0.3rem 0 0.1rem;
  text-transform:uppercase; letter-spacing:0.04em; }}
.ra-side-s {{ font-size:0.875rem; color:var(--ra-slate); margin-bottom:0.3rem; }}
.ra-kv {{ font-size:0.875rem; color:var(--ra-ink); line-height:1.6; }}
.ra-kv b {{ font-weight:600; }}

/* ---------- slim session header */
.ra-top {{ background:var(--ra-navy); color:#fff; border-radius:var(--ra-radius); padding:8px 14px;
  display:flex; flex-wrap:wrap; gap:6px 16px; align-items:center; justify-content:space-between; margin-bottom:8px; }}
.ra-brand {{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; }}
.ra-logo {{ width:30px; height:30px; flex:none; }}
.ra-title {{ font-size:1.375rem; font-weight:700; color:#fff; letter-spacing:-0.01em; }}
.ra-top-meta {{ display:flex; flex-wrap:wrap; gap:6px; }}
.ra-chip {{ display:inline-flex; align-items:center; gap:5px; border-radius:999px; padding:3px 10px;
  font-size:0.8125rem; line-height:1.3; border:1px solid rgba(255,255,255,0.3); color:#DCE6F2; }}
.ra-chip b {{ color:#fff; font-weight:600; }}
.ra-chip-research {{ border-color:#FDE7B0; color:#FDE7B0; font-weight:600; }}

/* ---------- primary navigation: two modes */
.st-key-nav {{ margin:0 0 6px; }}
.st-key-nav [data-testid="stWidgetLabel"] {{ display:none; }}
.st-key-nav button {{ min-height:44px; padding:0 18px; font-size:1rem; font-weight:600; color:var(--ra-navy); }}
.st-key-nav [data-testid="stButtonGroup"] button[aria-checked="true"] {{
  background:var(--ra-navy) !important; color:#fff !important; border-color:var(--ra-navy) !important; }}

/* ---------- headings */
.ra-sec {{ margin:4px 0 6px; }}
.ra-sec-t {{ font-size:1.375rem !important; font-weight:700; margin:0 !important; padding:0 !important; }}
.ra-sec-sub {{ color:var(--ra-slate); margin:2px 0 0; font-size:0.9375rem; }}
.ra-crumb {{ font-size:0.875rem; color:var(--ra-slate); margin-bottom:2px; }}

/* ---------- status line (slim queue summary) */
.ra-statusline {{ display:flex; flex-wrap:wrap; gap:4px 0; align-items:center; font-size:0.9375rem;
  border-top:1px solid var(--ra-border); border-bottom:1px solid var(--ra-border); padding:7px 2px; margin:4px 0 8px; }}
.ra-sl {{ display:inline-flex; gap:5px; align-items:baseline; padding-right:14px; margin-right:14px;
  border-right:1px solid var(--ra-border); }}
.ra-sl:last-of-type {{ border-right:none; }}
.ra-sl-l {{ color:var(--ra-slate); }}
.ra-sl b {{ color:var(--ra-navy); font-variant-numeric:tabular-nums; }}
.ra-sl-urgent b {{ color:var(--ra-red); }} .ra-sl-caution b {{ color:var(--ra-amber); }}
.ra-sl-note {{ color:var(--ra-slate); font-size:0.875rem; display:inline-flex; gap:5px; align-items:center; }}

/* ---------- controls: touch targets + focus */
[data-testid="stButton"] button, [data-testid="stDownloadButton"] button, [data-testid="stPopover"] button,
[data-testid="stFormSubmitButton"] button {{ min-height:44px; padding:0.45rem 1rem; border-radius:8px;
  font-size:0.9375rem; font-weight:600; width:100%; }}
[data-testid="stBaseButton-primary"] {{ background:var(--ra-teal); border-color:var(--ra-teal); color:#fff; }}
[data-testid="stBaseButton-primary"]:hover {{ background:var(--ra-teal-dark); border-color:var(--ra-teal-dark); color:#fff; }}
[data-testid="stBaseButton-secondary"] {{ background:#fff; color:var(--ra-navy); border:1.5px solid var(--ra-border-strong); }}
[data-testid="stBaseButton-secondary"]:hover {{ background:var(--ra-teal-tint); color:var(--ra-teal-dark); border-color:var(--ra-teal); }}
[data-testid="stBaseButton-tertiary"] {{ color:var(--ra-navy); justify-content:flex-start; }}
[data-testid="stButtonGroup"] button {{ min-height:44px; font-size:0.9375rem; padding:0 16px; color:var(--ra-navy); }}
[data-testid="stButtonGroup"] button[aria-checked="true"] {{ background:var(--ra-teal-tint); color:var(--ra-teal-dark);
  border-color:var(--ra-teal); }}
[class*="st-key-imp_"], [class*="st-key-final_"] {{ width:100% !important; }}
[class*="st-key-imp_"] [role="radiogroup"], [class*="st-key-final_"] [role="radiogroup"] {{ display:flex; width:100%; }}
[class*="st-key-imp_"] [role="radiogroup"] button, [class*="st-key-final_"] [role="radiogroup"] button {{
  flex:1 1 0; min-height:52px; font-size:1rem; }}
div[data-baseweb="select"] > div, [data-testid="stSelectbox"] [role="group"], [data-testid="stMultiSelect"] [role="group"],
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input {{ min-height:44px; font-size:0.9375rem; }}
[data-testid="stTooltipIcon"] button {{ min-width:28px; min-height:28px; display:inline-flex; align-items:center; justify-content:center; }}
[data-testid="stSidebarCollapseButton"] button, [data-testid="stExpandSidebarButton"], [data-testid="stMainMenu"] button {{
  min-width:40px; min-height:40px; }}
[data-testid="stSlider"] [role="slider"] {{ width:24px; height:24px; }}
[data-testid="stCheckbox"] label {{ min-height:40px; display:flex; align-items:center; }}
[data-testid="stExpander"] details {{ border-radius:8px; border:1px solid var(--ra-border); }}
[data-testid="stExpander"] summary {{ min-height:44px; font-weight:600; font-size:0.9375rem; color:var(--ra-navy); }}
button:focus-visible, [role="slider"]:focus-visible, input:focus-visible, summary:focus-visible, a:focus-visible,
[tabindex]:focus-visible {{ outline:3px solid var(--ra-teal) !important; outline-offset:2px !important; }}
@media (prefers-reduced-motion: reduce) {{ *, *::before, *::after {{ animation:none !important; transition:none !important; }} }}

/* ---------- worklist: dense rows, consistent height, separators instead of cards */
.st-key-rowhead {{ border-bottom:2px solid var(--ra-border-strong); padding:4px 6px; }}
.ra-th {{ font-size:0.875rem; font-weight:600; color:var(--ra-slate); }}
[class*="st-key-row_"] {{ border-bottom:1px solid var(--ra-border); padding:4px 6px; gap:0 !important; }}
[class*="st-key-row_"]:hover {{ background:#FAFBFC; }}
[class*="st-key-row_"] [data-testid="stHorizontalBlock"], .st-key-rowhead [data-testid="stHorizontalBlock"] {{
  flex-wrap:wrap !important; align-items:center; gap:2px 14px !important; }}
[class*="st-key-row_"] [data-testid="stColumn"], .st-key-rowhead [data-testid="stColumn"] {{ width:auto !important;
  min-width:0 !important; }}
[class*="st-key-row_"] [data-testid="stColumn"]:nth-child(1), .st-key-rowhead [data-testid="stColumn"]:nth-child(1) {{ flex:0 0 40px !important; }}
[class*="st-key-row_"] [data-testid="stColumn"]:nth-child(2), .st-key-rowhead [data-testid="stColumn"]:nth-child(2) {{ flex:1 1 110px !important; }}
[class*="st-key-row_"] [data-testid="stColumn"]:nth-child(3), .st-key-rowhead [data-testid="stColumn"]:nth-child(3) {{ flex:1 1 190px !important; }}
[class*="st-key-row_"] [data-testid="stColumn"]:nth-child(4), .st-key-rowhead [data-testid="stColumn"]:nth-child(4) {{ flex:2 1 250px !important; }}
[class*="st-key-row_"] [data-testid="stColumn"]:nth-child(5), .st-key-rowhead [data-testid="stColumn"]:nth-child(5) {{ flex:1 1 200px !important; }}
[class*="st-key-row_"] [data-testid="stColumn"]:nth-child(6), .st-key-rowhead [data-testid="stColumn"]:nth-child(6) {{ flex:0 0 116px !important; }}
[class*="st-key-row_"] [data-testid="stMarkdownContainer"], .st-key-rowhead [data-testid="stMarkdownContainer"] {{
  margin-bottom:0 !important; }}
.ra-pos {{ font-size:0.9375rem; font-weight:600; color:var(--ra-slate); font-variant-numeric:tabular-nums; }}
.ra-case {{ font-size:1.1875rem; font-weight:700; color:var(--ra-navy); }}
.ra-cell {{ font-size:0.9375rem; line-height:1.35; }}
.ra-cell-l {{ display:none; }}
.ra-cell-d {{ font-size:0.8125rem; color:var(--ra-slate); margin-top:2px; }}
.ra-p {{ font-variant-numeric:tabular-nums; font-weight:600; color:var(--ra-navy); margin-left:6px; }}
.ra-flag {{ color:var(--ra-amber); font-size:0.8125rem; font-weight:600; margin-left:6px; white-space:nowrap; }}

/* ---------- badges and status (shape + word + colour) */
.ra-badge, .ra-status, .ra-qc, .ra-hidden {{ display:inline-flex; align-items:center; gap:5px; font-size:0.875rem;
  line-height:1.25; border-radius:6px; }}
.ra-badge {{ padding:2px 8px; font-weight:600; border:1px solid currentColor; }}
.ra-b-urgent {{ color:var(--ra-red); background:var(--ra-red-bg); }}
.ra-b-caution {{ color:var(--ra-amber); background:var(--ra-amber-bg); }}
.ra-b-caution-outline {{ color:var(--ra-amber); background:#fff; }}
.ra-b-neutral {{ color:var(--ra-slate); background:var(--ra-slate-bg); }}
.ra-b-neutral-outline {{ color:var(--ra-slate); background:#fff; }}
.ra-status {{ font-weight:500; color:var(--ra-ink); }}
.ra-s-unread {{ color:var(--ra-slate); }} .ra-s-progress {{ color:var(--ra-navy); }}
.ra-s-done {{ color:var(--ra-navy); }} .ra-s-caution {{ color:var(--ra-amber); }}
.ra-qc-ok {{ color:var(--ra-slate); }}
.ra-qc-fail {{ color:var(--ra-amber); font-weight:600; }}
.ra-hidden {{ color:var(--ra-slate); font-style:normal; }}

/* ---------- meter */
.ra-meter {{ position:relative; height:8px; border-radius:999px; background:#E4EAF0; margin:6px 0 4px; }}
.ra-meter-fill {{ height:100%; border-radius:999px; background:var(--ra-navy-2); }}
.ra-meter-thr {{ position:absolute; top:-4px; width:2px; height:16px; background:var(--ra-ink); }}

/* ---------- case bar + stepper */
.ra-casehead {{ display:flex; flex-wrap:wrap; gap:4px 12px; align-items:baseline; margin:2px 0 4px; }}
.ra-casehead .ra-case {{ font-size:1.1875rem; }}
.ra-meta {{ font-size:0.875rem; color:var(--ra-slate); }}
ol.ra-steps {{ list-style:none; display:flex !important; flex-direction:row !important; flex-wrap:wrap; gap:6px 18px;
  padding:0 !important; margin:2px 0 8px !important; }}
.ra-steps .ra-step {{ margin:0 !important; padding:0 !important; display:flex; align-items:center; gap:7px;
  font-size:0.875rem; color:var(--ra-slate); font-weight:500; }}
.ra-step-n {{ width:22px; height:22px; border-radius:50%; display:inline-flex; align-items:center; justify-content:center;
  border:1.5px solid var(--ra-border-strong); font-size:0.75rem; font-weight:700; }}
.ra-step-current {{ color:var(--ra-navy) !important; font-weight:700 !important; }}
.ra-step-current .ra-step-n {{ background:var(--ra-navy); border-color:var(--ra-navy); color:#fff; }}
.ra-step-done .ra-step-n {{ background:var(--ra-slate-bg); border-color:var(--ra-slate); color:var(--ra-navy); }}

/* ---------- viewer: the one strong boundary on the page */
.st-key-viewer {{ background:var(--ra-viewer); border-radius:var(--ra-radius); padding:10px 12px 14px; }}
.st-key-viewer [data-testid="stMarkdownContainer"] {{ margin-bottom:0 !important; }}
.st-key-viewer [data-testid="stWidgetLabel"] p, .st-key-viewer [data-testid="stCheckbox"] p,
.st-key-viewer [data-testid="stCaptionContainer"] p {{ color:#D7E1EC !important; }}
.st-key-viewer button p {{ color:inherit !important; }}
.st-key-viewer [data-testid="stPopover"] button {{ background:#132540; color:#E6EDF5; border:1.5px solid #3B5373; }}
.st-key-viewer [data-testid="stPopover"] button:hover {{ border-color:{TEAL_SOFT}; color:#fff; }}
.st-key-viewer [data-testid="stImage"], .st-key-viewer [data-testid="stImageContainer"] {{ background:#000; border-radius:6px; }}
.st-key-viewer img {{ border-radius:4px; max-width:100%; height:auto; }}
.st-key-viewer [data-testid="stBaseButton-secondary"] {{ background:#132540; color:#E6EDF5; border-color:#3B5373; }}
.st-key-viewer [data-testid="stBaseButton-secondary"]:hover {{ background:#1B3254; color:#fff; border-color:{TEAL_SOFT}; }}
.st-key-viewer [data-testid="stButtonGroup"] button {{ background:#132540; color:#E6EDF5; border-color:#3B5373; }}
.st-key-viewer [data-testid="stButtonGroup"] button[aria-checked="true"] {{ background:#E6EDF5; color:var(--ra-navy);
  border-color:#E6EDF5; }}
.st-key-viewer [data-testid="stSlider"] [role="slider"] {{ background:#E6EDF5; }}
/* slice step buttons: icon (and shortcut hint) only; the label stays available to screen readers */
.st-key-slice_prev [data-testid="stMarkdownContainer"], .st-key-slice_next [data-testid="stMarkdownContainer"] {{
  position:absolute; width:1px; height:1px; overflow:hidden; clip:rect(0 0 0 0); white-space:nowrap; }}
.st-key-slice_prev button, .st-key-slice_next button {{ padding:0 6px !important; }}
.ra-vinfo {{ display:flex; flex-wrap:wrap; justify-content:space-between; gap:4px 12px; font-size:0.875rem;
  color:#D7E1EC; font-variant-numeric:tabular-nums; }}
.ra-vinfo b {{ color:#fff; }}
.ra-legend {{ display:flex; flex-wrap:wrap; gap:4px 16px; font-size:0.875rem; color:#D7E1EC; margin:4px 0 0; }}
.ra-leg {{ display:inline-flex; align-items:center; gap:7px; }}
.ra-leg i {{ display:inline-block; width:20px; height:11px; border-radius:2px; background:#2B3A4A; }}
.ra-leg-line {{ box-shadow:inset 0 0 0 2.5px var(--c); }}
.ra-leg-fill {{ background:var(--c) !important; }}
.ra-leg-dotted {{ background:repeating-linear-gradient(90deg, var(--c) 0 4px, #2B3A4A 4px 7px) !important; }}

/* ---------- evidence + decision panels: one surface each, separators inside */
.st-key-evidence, .st-key-decision, .st-key-impression, .st-key-simulation {{ border:1px solid var(--ra-border);
  border-radius:var(--ra-radius); padding:12px 14px; background:#fff; }}
.st-key-decision {{ border:2px solid var(--ra-navy); }}
.st-key-impression {{ border:2px solid var(--ra-navy); }}
.ra-panel-t {{ font-size:1.0625rem; font-weight:700; color:var(--ra-navy); display:flex; align-items:center; gap:8px; }}
.ra-pn {{ width:24px; height:24px; border-radius:50%; background:var(--ra-navy); color:#fff; font-size:0.8125rem;
  display:inline-flex; align-items:center; justify-content:center; }}
.ra-panel-s {{ font-size:0.9375rem; color:var(--ra-slate); margin:2px 0 4px; }}
.ra-compare {{ display:flex; flex-wrap:wrap; gap:8px 22px; align-items:center; background:var(--ra-slate-bg);
  border-radius:8px; padding:8px 12px; margin:4px 0 8px; font-size:0.9375rem; }}
.ra-k {{ font-size:0.8125rem; color:var(--ra-slate); }}
.ra-rel {{ font-size:0.875rem; font-weight:600; border-radius:6px; padding:2px 8px; }}
.ra-rel-same {{ color:var(--ra-navy); background:#fff; border:1px solid var(--ra-border-strong); }}
.ra-rel-diff {{ color:var(--ra-amber); background:#fff; border:1px solid var(--ra-amber); }}
.ra-ai {{ margin:2px 0 6px; }}
.ra-ai-head {{ font-size:1.1875rem; font-weight:700; color:var(--ra-navy); line-height:1.3; }}
.ra-ai-target {{ font-size:0.9375rem; margin-top:4px; }}
.ra-ai-target b {{ font-variant-numeric:tabular-nums; color:var(--ra-navy); }}
.ra-ai-unc {{ font-size:0.9375rem; margin-top:4px; }}
.ra-unc-high {{ color:var(--ra-amber); font-weight:600; }}
.ra-ag {{ display:flex; gap:7px; align-items:flex-start; border-radius:8px; padding:6px 10px; margin-top:8px; font-size:0.9375rem; }}
.ra-ag .ra-ico {{ margin-top:3px; }}
.ra-ag-neutral {{ background:var(--ra-slate-bg); color:var(--ra-ink); }}
.ra-ag-caution {{ background:var(--ra-amber-bg); color:#7A2E0E; }}
.ra-valid {{ font-size:0.875rem; color:var(--ra-slate); margin-top:6px; display:flex; gap:6px; align-items:center; }}
.ra-why {{ margin-top:6px; font-size:0.9375rem; }}
.ra-why summary {{ cursor:pointer; color:var(--ra-navy); font-weight:600; min-height:32px; display:flex; align-items:center; }}
.ra-why ul {{ margin:2px 0 0 1.1rem; padding:0; }}
.ra-ai-withheld .ra-ai-body {{ font-size:0.9375rem; margin-top:4px; }}
dl.ra-measures {{ display:flex; flex-wrap:wrap; gap:4px 0; margin:6px 0; padding:6px 0; border-top:1px solid var(--ra-border);
  border-bottom:1px solid var(--ra-border); }}
dl.ra-measures > div {{ padding:0 16px 0 0; margin-right:16px; border-right:1px solid var(--ra-border); }}
dl.ra-measures > div:last-child {{ border-right:none; }}
dl.ra-measures dt {{ font-size:0.8125rem; color:var(--ra-slate); }}
dl.ra-measures dd {{ margin:0; font-size:1.0625rem; font-weight:700; color:var(--ra-navy); font-variant-numeric:tabular-nums; }}
dl.ra-measures dd span {{ font-size:0.875rem; font-weight:500; color:var(--ra-slate); margin-left:3px; }}

/* ---------- notes */
.ra-note {{ display:flex; gap:9px; align-items:flex-start; border-radius:8px; padding:8px 12px; margin:4px 0 8px; font-size:0.9375rem; }}
.ra-note .ra-ico {{ margin-top:3px; }}
.ra-note-info {{ background:#EEF3F9; color:#1B3A5E; }}
.ra-note-neutral {{ background:var(--ra-slate-bg); color:var(--ra-ink); }}
.ra-note-warn {{ background:var(--ra-amber-bg); color:#7A2E0E; }}
.ra-note-critical {{ background:var(--ra-red-bg); color:#7A1A12; }}

/* ---------- report */
.st-key-report_doc pre {{ background:#FFFFFF !important; border:1px solid var(--ra-border); border-radius:8px;
  white-space:pre-wrap !important; word-break:break-word; }}
.st-key-report_doc code {{ font-size:0.875rem !important; color:var(--ra-ink) !important; line-height:1.55; }}

/* ---------- flow diagram (About) */
ol.ra-flow {{ list-style:none; display:flex !important; flex-direction:row; flex-wrap:wrap; gap:10px 26px;
  padding:0 !important; margin:8px 0 14px !important; }}
.ra-flow li {{ position:relative; margin:0 !important; padding:8px 12px !important; border-radius:8px; min-width:130px;
  border:1px solid var(--ra-border-strong); background:#fff; display:flex; flex-direction:column; gap:2px; }}
.ra-flow li b {{ color:var(--ra-navy); font-size:0.9375rem; }}
.ra-flow li span {{ color:var(--ra-slate); font-size:0.8125rem; }}
.ra-flow li:not(:last-child)::after {{ content:"→"; position:absolute; right:-20px; top:50%; transform:translateY(-50%);
  color:var(--ra-slate); font-weight:700; }}
.ra-flow-ai {{ border-color:var(--ra-navy) !important; }}
.ra-flow-human {{ border:2px solid var(--ra-navy) !important; }}

/* ---------- responsive */
@media (max-width: 900px) {{
  .ra-device-note {{ display:block; }}
  .st-key-casenav [data-testid="stHorizontalBlock"] {{ flex-wrap:nowrap !important; gap:8px !important; }}
  .st-key-casenav [data-testid="stColumn"] {{ min-width:0 !important; width:auto !important; flex:0 0 52px !important; }}
  .st-key-casenav [data-testid="stColumn"]:nth-child(2) {{ flex:1 1 auto !important; }}
  .st-key-prev_case [data-testid="stMarkdownContainer"], .st-key-next_case [data-testid="stMarkdownContainer"] {{
    position:absolute; width:1px; height:1px; overflow:hidden; clip:rect(0 0 0 0); white-space:nowrap; }}
  .st-key-prev_case button, .st-key-next_case button {{ padding:0 !important; }}
}}
@media (max-width: 640px) {{
  [data-testid="stMainBlockContainer"], .block-container {{ padding:3rem 0.75rem 3rem; }}
  .ra-top-meta {{ display:none; }}
  .ra-title {{ font-size:1.125rem; }}
  .st-key-rowhead {{ display:none; }}
  .ra-cell-l {{ display:inline; font-size:0.75rem; color:var(--ra-slate); text-transform:uppercase; letter-spacing:0.04em;
    margin-right:6px; }}
  [class*="st-key-row_"] {{ padding:8px 4px; }}
  .ra-sl {{ padding-right:10px; margin-right:10px; }}
  .st-key-nav {{ width:100% !important; }}
  .st-key-nav [data-testid="stButtonGroup"], .st-key-nav [role="radiogroup"] {{ display:flex; width:100% !important; }}
  .st-key-nav button {{ flex:1 1 0; }}
  ol.ra-flow {{ flex-direction:column; gap:20px; }}
  .ra-flow li:not(:last-child)::after {{ content:"↓"; right:auto; left:20px; top:auto; bottom:-19px; transform:none; }}
}}
"""

# rows created with streamlit_app.cols(name, ...) reflow by the space they have: (min px per column, grow)
REFLOW = {"overlays": [140], "decide": [160], "after": [160], "reportbar": [240], "downloads": [220],
          "charts": [360], "figs": [380], "about": [320], "onboard": [320], "rerunrow": [260],
          "toolbar": [220, 150, 170, 150], "viewtools": [330, 130], "viewopts": [118, 104, 138, 150], "sim": [170], "policies": [260],
          "read": [560, 360], "slicenav": [52, 200, 52]}
REFLOW_GROW = {"read": [1.6, 1.0], "toolbar": [2, 1.4, 1, 0.8], "viewtools": [3, 1], "slicenav": [0, 1, 0]}


def reflow_css() -> str:
    out = []
    for name, bases in REFLOW.items():
        sel = f'[class*="st-key-rx_{name}"]' if name == "figs" else f".st-key-rx_{name}"
        hbs = [f'{sel} > [data-testid="stHorizontalBlock"]', f'{sel} > * > [data-testid="stHorizontalBlock"]']
        out.append(", ".join(hbs) + " { flex-wrap:wrap !important; }")
        grows = REFLOW_GROW.get(name, [])
        for i, px in enumerate(bases, start=1):
            nth = f":nth-child({i})" if len(bases) > 1 else ""
            g = grows[i - 1] if i - 1 < len(grows) else 1
            col = ", ".join(f'{h} > [data-testid="stColumn"]{nth}' for h in hbs)
            out.append(f"{col} {{ flex:{g} 1 {px}px !important; min-width:min({px}px, 100%) !important; "
                       f"width:auto !important; }}")
    return "\n".join(out)


def style_tag(extra: str = "") -> str:
    return f"<style>{CSS}\n{reflow_css()}\n{extra}</style>"
