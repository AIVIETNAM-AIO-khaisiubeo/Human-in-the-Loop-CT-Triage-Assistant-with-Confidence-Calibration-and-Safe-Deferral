# Reading interface - UX design notes

The website is the radiologist's side of ReaderAdapt-Liver. These notes record who it is for, the
reading journey it supports, the design rules derived from it, and how the external design review
(round 2) was addressed. They are the reference for future UI changes.

> Research prototype. Nothing here claims clinical readiness; section 6 lists what would have to be
> shown before any clinical use.

## 1. Context of use

| | |
|---|---|
| **Users** | General radiologists and residents in a regional hospital; a QA / research lead who reviews performance and adaptation. |
| **Devices** | Reading-room workstation (large screen, dimmed room) for interpretation; office desktop; tablet or phone only for checking the worklist and saved decisions. |
| **Environment** | Interruptions, end-of-shift fatigue, shared workstations, hospital networks that may block external fonts and CDNs (the bundled Source Sans font is used). |
| **Stakes** | The AI suggests; the radiologist decides. Anchoring on AI output, over-trust and under-trust are use errors with patient impact. |

## 2. Two working modes and a separate research area

| Mode | Goal | What is on screen |
|---|---|---|
| **Worklist** | Find the next case and see whether it is ready to read. | Slim session bar, one-line queue summary, a toolbar (status filter, case search, order, details), a dense table with consistent row heights: position, case, *Image / AI QC* (operational), *AI finding* (hidden in blind mode), status, one **Open** action. |
| **Read case** | Read the images, examine the evidence, decide, document. | Case bar (previous / case picker / next), a three-step indicator, the CT viewer as the main workspace, then *AI evidence* and *Reconcile and decide* as two separate panels. No queue statistics. |
| **Research and quality** (sidebar) | Governance, not reading. | Reader adaptation, model performance, audit trail of the current case, intended use and supported devices. |

## 3. The reading journey

| # | Stage | Interface |
|---|---|---|
| 0 | **Arrive** | Slim navy bar: app, *research prototype*, reader, reading mode, session id, model-set id. |
| 1 | **Plan** | Worklist. In *blind first read* (default) cases are in **arrival order** and no AI output is shown for unread cases: no prediction, no probability, no disagreement flag, no AI-derived priority or urgency count. Operational QC status is shown, and a warning appears only when unread cases need action. |
| 2 | **Independent read** | The viewer opens with every AI overlay off and disabled; the start slice is the middle of the slab, not the AI's key slice. Two large answers (*Tumour present* / *No liver tumour*); **Save impression and reveal AI evidence**. |
| 3 | **Reveal AI evidence** | Your impression and the AI suggestion side by side (*same as / differs from your impression*). AI summary: target and unit stated (*patient-level probability that this CT shows at least one liver tumour, calibrated*), threshold, uncertainty with its scale, class *and* score agreement of the two AI tools, validity (QC), reasons on demand. Measurements as one group, lesion table with *show lesion slice*. Second-reader text, uncertainty/calibration and model versions are collapsed. |
| 4 | **Reconcile and decide** | A separate panel: final finding **Tumour / No tumour / Defer** (second opinion), optional lesion edit and note, one **Save decision**. The action (agreed / modified / differs / independent / deferred) is derived; the AI suggestion is not a pre-selected answer. |
| 5 | **Report** | Fixed-template draft with the decision and the recommendation that was on screen; copy / download. Closed in blind mode until the impression is saved. |
| 6 | **Govern** | Research area: the active (frozen) policy versus the proposed one, explicit *Apply*, policy-change log, decision log with recommendation shown, agent trace. |

## 4. Design rules

**Colour has one meaning each** (contrast checked in `tests/test_app.py`):

| Token | Value | Meaning | Contrast |
|---|---|---|---|
| Navy | `#102D52` | application structure: header, navigation, headings | 13.8 : 1 on white |
| Teal | `#0F766E` | active controls and primary actions only - never "the AI is right" | 5.5 : 1 with white text |
| Red | `#B42318` | a validated critical rule (urgent: probability ≥ 85 % and lesion ≥ 5 mL or ≥ 20 mm) | 6.1 : 1 on its tint |
| Amber | `#B54708` | caution or review needed: QC issue, uncertain output, tool disagreement, deferred | 5.1 : 1 on its tint |
| Slate | `#526579` | neutral and technical information, metadata | 6.0 : 1 on white |

No green is used: a positive colour next to an AI output would read as clinical approval. Category badges
say *AI:* (e.g. *AI: tumour suspected*) and pair a distinct shape with the word. The CT is shown on a
near-black panel; overlays (teal liver outline, amber lesions with adjustable fill, violet uncertainty) are
mutually distinguishable and visible on dark and grey backgrounds.

**Type** (body 16 px): app title and page heading 22 px, case identifier 19 px, labels and table text
14-15 px, secondary metadata 13-14 px. Bold is reserved for the case identifier, findings and actions;
probabilities are percentages, volumes in mL, numbers in tabular figures.

**Components**: one strong boundary (the viewer), one evidence panel, one decision panel; separators
instead of nested cards; measurements as a single group; disclosure for details.

**Touch and keyboard**: controls at least 44 px high (the two clinical answers 52 px); visible 3 px focus
ring; reduced-motion support. Shortcuts: **J / K** next / previous slice, **N / P** next / previous case -
never on decision buttons, and inactive while typing.

**Responsive**: rows reflow by the space they actually have; on narrow screens the worklist rows become
compact cards with inline labels, the session bar keeps only the title and research label, and the case
view shows a *small screen - not for diagnostic interpretation* notice.

## 5. Design review (round 2) - what changed

| Review point | Response |
|---|---|
| **P0** AI cues leaked into the "independent impression first" worklist | *Blind first read* mode: no AI prediction, probability, disagreement or AI priority for unread cases; arrival order; urgency and disagreement counts removed from the summary; case picker shows status only; report and audit trail closed until the impression is saved. AI evidence unlocks only after a saved impression and is shown next to it. |
| **P0** Adaptation must not change behaviour silently | The decision policy is frozen per session (versioned, shown in the sidebar); a proposed update is applied only with *Apply* and logged to `feedback/policy_changes.jsonl`; batch simulation runs offline and never changes the active policy. "Reader adaptation" moved to the research area. |
| **P0** Audit what was shown | Each decision stores session id, policy version, model-set version, the recommendation text and values on screen, and the impression time. The report quotes the recommendation shown. |
| **P0** Ambiguous terms | *Urgent* is defined by its rule; agreement distinguishes class from score (*both flag a tumour, but their scores differ: 73 % vs 91 %*); uncertainty states its scale and thresholds; QC failures say what failed (image vs segmentation) and what it means, and are kept apart from clinical urgency; probability names its target. |
| **P1** Reading view crowded | KPI cards removed; two-mode navigation; the CT viewer takes ~60 % of the width with orientation markers, *slice n / N*, window/level presets plus custom W/L in HU, zoom and pan, overlay visibility and opacity, reset, step buttons and shortcuts. Evidence and decision are separate panels; model names moved to *Models and versions*. |
| **P1** Worklist too much dashboard | Dense table by default with consistent rows, one toolbar, a one-line summary, priority reason in *Details*. |
| **P2** Visual system, typography, boxes | Semantic palette above; restrained type scale; fewer containers. |
| **P2** Mobile | Compact cards, slim header, two-item navigation, explicit supported-device policy (About). |

## 6. Before any clinical use

The interface is a research instrument. A clinical evaluation would need, at minimum: a usability-engineering
process in the spirit of IEC 62366-1 and current human-factors guidance for medical devices; formative and
summative studies with representative radiologists on realistic cases and the intended diagnostic displays,
measuring whether users recognise failed QC, distinguish probability from uncertainty, read independently
before seeing AI output, understand the consequences of each decision, recover from mistakes and complete the
workflow under interruption; validation of the viewer on diagnostic displays; and offline validation of any
adaptive behaviour before it is enabled.
