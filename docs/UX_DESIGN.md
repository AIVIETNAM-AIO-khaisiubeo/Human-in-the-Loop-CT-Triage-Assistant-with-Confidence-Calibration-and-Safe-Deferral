# Reading interface - UX design notes

The website is the radiologist's side of ReaderAdapt-Liver. These notes record who it is for, the
reading journey it supports, and the design rules derived from that journey. They are the reference
for future UI changes.

## 1. Context of use

| | |
|---|---|
| **Users** | General radiologists and residents in a regional hospital (not necessarily abdominal sub-specialists); a QA / research lead who reviews performance. |
| **Devices** | Reading-room workstation (large screen, dimmed room); office desktop; tablet or phone on the ward and on call. |
| **Environment** | Frequent interruptions, end-of-shift fatigue, shared workstations, hospital networks that may block external fonts and CDNs. |
| **Stakes** | The AI makes suggestions; the radiologist is responsible for every finding. Over-trust (accepting a wrong suggestion) and under-trust (ignoring a right one) are both harms. |

## 2. Radiologist journey

| # | Stage | What the radiologist needs to know or do | Interface answer |
|---|---|---|---|
| 0 | **Arrive** | Who am I reading as? Is this a clinical or research tool? How much work is waiting? | Navy header with reader identity and a calm, permanent "research prototype - not for clinical use" label; a status strip: *to read, urgent, AI tools disagree, deferred to you, reviewed*. Model performance is not shown here - it belongs to QA. |
| 1 | **Plan** (worklist) | Which case first, and *why* is it ranked there? | Case cards in priority order with a category badge (shape + word + colour), the AI suggestion, the reason for the priority, the AI tools' agreement and the review status; one large **Open** button per case. A compact table remains for desktop power users. |
| 2 | **Independent read** | Look at the CT without being anchored by the AI. | Sequential protocol: the viewer opens with AI overlays hidden; a clear step indicator (*1 Your impression → 2 AI suggestion → 3 Decision → 4 Report*); two large answer buttons. |
| 3 | **Understand the AI** | What does it suggest, how sure is it, do its tools agree, what did it measure, and can I trust this? | Plain-language headline ("Liver tumour likely"), the calibrated probability and an uncertainty level in words, an explicit agreement line for the segmenter and MedGemma, the measurements as large tiles, quality-control warnings, the lesion table. Technical details and MedGemma's unverified text are one tap away, not in the way. Provenance (which models, calibrated on which split) is always stated. |
| 4 | **Decide** | Accept, edit or reject with no accidental taps, then move on. | Three large, well-separated buttons (Accept is the only filled one); edit opens a short form; a saved-decision confirmation with **Next case** and **View report**. |
| 5 | **Report** | Check and take the structured draft. | Document-style panel with copy and download, the review status and who decided. |
| 6 | **Reflect** | How is the assistant adapting to me, and why? | "How the assistant adapts to you": current threshold and deferral rule in words, number of reads, why it is or is not personalised yet, the trust curve. |
| 7 | **Govern** (QA lead) | Is the AI calibrated? What did the agent do on this case? | *Performance* and *Audit trail* views, separated from the reading flow. |

Moving between cases never requires the sidebar: **Previous / Next** sit above the viewer and every
worklist card opens its case directly. The sidebar holds session settings that change rarely (reader,
decision policy, worklist, protocol, research mode).

## 3. Design rules

**Colour.** Deep navy for structure and headings, soft teal for actions and highlights, a white page.
Every text/background pair meets WCAG 2.1 AA (most exceed 6:1):

| Token | Value | Use | Contrast |
|---|---|---|---|
| navy | `#0B2545` | headings, header band | 15.4 : 1 on white |
| ink | `#16324F` | body text | 13.1 : 1 |
| muted | `#4A5D72` | secondary text | 6.8 : 1 |
| teal | `#0F766E` | primary buttons, focus ring, links | 5.5 : 1 with white |
| teal tint | `#E3F3F1` | selected / highlight backgrounds | teal-dark text 6.6 : 1 |
| soft teal | `#8FD3CB` | accents on navy | 9.1 : 1 on navy |

Triage categories have their own muted colours (urgent red `#B42318`, suspicious amber `#B54708`,
uncertain violet `#5B3CC4`, likely normal green `#067647`, deferred blue `#1D4E89`, QC fail slate
`#344054`), always paired with a glyph and a word, so colour is never the only cue. The CT itself is
shown on a near-black panel: radiologists read images on dark surroundings, and a white frame around
a CT causes glare in a dimmed reading room.

**Type.** The bundled Source Sans font (no external font requests). Body 17 px, line height 1.55;
page title 26-28 px, section titles 20-21 px, card titles 18 px; secondary text never below 14 px;
tabular figures for numbers.

**Touch and keyboard.** Buttons, segmented controls, inputs and toggles are at least 48 px high with
at least 8 px between targets; the slider thumb is enlarged. Every interactive element shows a 3 px teal
focus ring. Motion is disabled for users who ask for reduced motion.

**Responsive.** Layouts are built from columns that stack below 640 px; the status strip reflows from
five tiles to two per row; the worklist becomes a single column of cards with full-width buttons; images
scale to the screen width.

**Trust.** Say what the AI is and is not: the research label is permanent but not alarming; the AI's
probability is called *calibrated* only because it is; uncertainty and tool disagreement are shown next
to the suggestion, never hidden; model text is labelled *unverified*; the radiologist's decision is
always the last step.
