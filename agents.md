# AGENTS.md — Biscuit Quality Testing System ("Acknowledge" team, AI Open 2026)

This file is the shared source of truth for the team (and for any AI coding
assistant helping out). Read this before touching code. Update it when the
project state changes — stale docs are worse than no docs.

## 1. What this project actually is

We are building a prototype for **Case 3: Data Analysis — SPL Biscuit Quality
Testing** for the AI Open 2026 semifinals. The proposal promises a
**Machine Learning-based Decision Support System** for biscuit certification
auditors, with these promised parts:

1. Classification model — predicts pass/fail from lab test parameters
2. Clustering model — groups samples to reveal failure patterns
3. Interactive Dashboard — visualizes quality metrics + root causes
4. Predictive Simulator — user types in hypothetical lab values, gets a
   prediction before mass production
5. Executive Summary generator — turns model output into a plain-language
   report

**Deadline: prototype demo video due 17 July 2026.**

## 2. Current state (as of this doc's creation)

Two repos exist:

- `biscuit-classification` — Colab notebook (`lombah.ipynb`), trains a
  MobileNetV2 image classifier on biscuit **photos** (OK / NOK). Training-only,
  not wired to any app.
- `biscuit-classification-web` (this repo) — the actual product:
  - `backend/app.py` — Flask API. Loads the trained `.keras` model from
    GitHub Releases (tag: `models`). One route that matters: `POST /predict`,
    takes an uploaded **image**, returns class + confidence.
  - `frontend/index.html` — single-page upload UI, calls `/predict`.

**What's built = image-based visual inspection only.**
**What's promised in the proposal = tabular lab-parameter analysis
(moisture, fat, protein, ash, microbial count, heavy metals) + clustering +
simulator + dashboard + executive summary.**

Those are two different data modalities. The image classifier is not wasted
work — the proposal's own attachment shows a "Visual Inspection of Cookie"
card — but it is only **one module** of the full promised system, not the
whole thing. The tabular side does not exist yet.

## 3. Build plan to close the gap (target: demo-able by July 17)

**Decision (confirmed with team lead, no committee dataset was provided for
this case as of the deadline): we are NOT training a tabular ML model.**
Reasons:
- No real or committee-provided lab dataset exists — a model "trained" on
  fabricated data would report a meaningless accuracy number.
- A rule-based engine encoding real SNI 2973:2011 thresholds is more
  honest, fully explainable, and appropriate for a certification/audit
  context (auditors need to point to *which exact standard* was violated —
  a black-box model can't do that as convincingly).
- This is a **showcase-quality prototype**, not a full production app. Goal
  is a clean, working, honestly-scoped demo — not maximum technical
  complexity. We can scale to real ML later if the team continues past the
  competition.

**Confirmed SNI 2973:2011 thresholds (verified via research, not guessed):**

| Parameter | Rule | Confidence |
|---|---|---|
| Moisture | fail if > 5% | High |
| Protein | fail if < 5% | High |
| Ash | fail if > 1% | Medium |
| Fat, microbial count, heavy metals | placeholder thresholds, clearly labeled as such in UI/code | Low — real BSN document not freely accessible; swap in real numbers if committee data appears later |

Build order, each step a separate branch/PR:

1. **Rule engine (backend)** — a config-driven set of threshold checks
   (`if moisture > 5: fail("too wet")`, etc.) exposed via a new
   `POST /predict-lab` route. Input: JSON of lab parameters. Output:
   overall pass/fail + list of which parameter(s) failed and why.
2. **Failure-category tagging** — group failed parameters into categories
   (moisture-related, protein-related, contamination-related) so the
   dashboard has something structured to visualize. This satisfies the
   proposal's "clustering/pattern mapping" promise without needing real ML
   — it's rule-based grouping, and we say so honestly in the video.
3. **Predictive Simulator UI** — a form (number inputs, one per lab
   parameter) calling `/predict-lab`, showing pass/fail + reasons instantly.
4. **Dashboard** — simple visualization of failure categories (e.g. a small
   bar/donut chart of which parameter type is failing most across sample
   submissions).
5. **Executive Summary** — template-based text generator built from the
   rule engine's output (e.g. "Sample failed due to excess moisture
   (6.2% > 5% limit), consistent with high-moisture failure pattern...").
6. **UI/UX enhancement pass** — improve the existing prototype's visual
   design based on Figma reference:
   https://www.figma.com/site/AAmxUTRZiU28wxjGpX2xIG/Untitled
   (Claude needs edit access or exported screenshots to read this file —
   ask whoever owns it to share, or export frames as images.)
7. Keep the image classifier as an additional "Visual Inspection" tab in the
   UI, not the centerpiece.

**Explicitly out of scope for the demo:** training any model on tabular
data, real-time continuous learning, real lab instrument integration,
multi-tenant SaaS auth/billing. These are fine to describe as "future work"
in the video, but are not being built now.

## 4. Tech stack

- Backend: Flask + flask-cors, TensorFlow/Keras (image model), scikit-learn
  (tabular models — to be added)
- Frontend: currently plain HTML/CSS/JS (`frontend/index.html`). Team can
  decide whether to keep it plain or move to a framework — no framework
  chosen yet, discuss before switching mid-sprint.
- Model storage: large model files go in **GitHub Releases**, never
  committed directly to git history.

## 5. How to run locally

```bash
cd backend
pip install -r requirements.txt
python app.py          # runs on http://localhost:5000
```

Open `frontend/index.html` directly in a browser (it points at
`http://localhost:5000/predict` by default — update `API_URL` in the
`<script>` block if your backend runs elsewhere).

## 6. Team conventions

- No fixed FE/BE split right now — everyone works across both frontend and
  backend. Roles listed in the proposal are placeholders for the paperwork,
  not a hard division of labor.
- Before starting new work, check this file's Section 3 for what's already
  claimed/in-progress. Add your name + what you're working on below to avoid
  collisions:

| Task | Owner | Status |
|---|---|---|
| Rule engine + `/predict-lab` route | — | not started |
| Failure-category tagging logic | — | not started |
| Predictive Simulator UI | — | not started |
| Dashboard | — | not started |
| Executive Summary generator | — | not started |
| UI/UX polish pass (Figma reference) | — | not started |
| Demo video script/recording | — | not started |

- Commit messages: short, present tense (`add clustering route`, not
  `Added clustering route` or `stuff`).
- Don't commit `.keras`/`.h5`/large model files — use Releases like the
  existing model.

## 7. Open questions for the team

- ~~Does anyone have access to real lab test data?~~ **Resolved: no
  committee dataset provided — going rule-based, see Section 3.**
- Frontend: stay plain HTML/JS, or move to a framework for the dashboard
  (charts are easier with something like Chart.js even in plain JS — may not
  need a framework switch at all)?
- Who has edit access to the Figma design file, and can they export the
  relevant frames as PNGs so the whole team (and AI assistants helping out)
  can actually see the intended design?
- Who is recording/scripting the demo video, and by what internal date
  (leave buffer before July 17)?