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

In priority order — each step should be a separate branch/PR:

1. **Synthetic tabular dataset** — generate realistic lab data grounded in
   real SNI 2973:2011 biscuit quality thresholds (moisture %, fat %,
   protein %, ash, microbial count, heavy metal ppm → pass/fail label).
   State clearly in the video that this is synthesized for demo purposes.
2. **Classification model (tabular)** — simple, explainable model (logistic
   regression or random forest, not a CNN) trained on the dataset above.
3. **Clustering model** — KMeans on the same features, to group failure
   types (e.g. "high-moisture cluster" vs "contamination cluster").
4. **New Flask routes** — `POST /predict-lab` (classification) and
   `POST /cluster` (clustering), alongside the existing `/predict` (image)
   route. Do not break `/predict`.
5. **Predictive Simulator UI** — a form (sliders/number inputs) for each lab
   parameter, calling `/predict-lab`.
6. **Dashboard** — charts showing cluster distribution and which parameter
   is driving failures most.
7. **Executive Summary** — rule-based / template text generator from model
   output. No need for an LLM call here — deterministic and explainable is
   better for a judged demo.
8. Keep the image classifier as an additional "Visual Inspection" tab in the
   UI, not the centerpiece.

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
| Synthetic dataset | — | not started |
| Tabular classification model | — | not started |
| Clustering model | — | not started |
| `/predict-lab` + `/cluster` routes | — | not started |
| Predictive Simulator UI | — | not started |
| Dashboard | — | not started |
| Executive Summary generator | — | not started |
| Demo video script/recording | — | not started |

- Commit messages: short, present tense (`add clustering route`, not
  `Added clustering route` or `stuff`).
- Don't commit `.keras`/`.h5`/large model files — use Releases like the
  existing model.

## 7. Open questions for the team

- Does anyone have access to real (even partial/anonymized) lab test data,
  or are we fully committed to synthetic data for the demo?
- Frontend: stay plain HTML/JS, or move to a framework for the dashboard
  (charts are easier with something like Chart.js even in plain JS — may not
  need a framework switch at all)?
- Who is recording/scripting the demo video, and by what internal date
  (leave buffer before July 17)?