# 🐻 Bear Panel — Synthetic Consumer Research Platform

A bear-themed **synthetic consumer research platform** that runs locally on
**Ollama** via **CrewAI**. Each bear is one synthetic survey respondent with a
full persona. What began as a playful demo is now a serious tool: you can
calibrate the whole panel to a **real survey dataset**, mix real / custom /
AI-generated respondents, and validate every synthetic study against ground
truth — while keeping the bears.

## Run

```bash
# 1. Ollama with a local model
ollama pull llama3.1
ollama serve

# 2. Backend
pip install fastapi uvicorn crewai pydantic python-multipart openpyxl
python local_swarm.py            # serves http://127.0.0.1:8000

# 3. Frontend
# open index.html in a browser (double-click, or serve the folder)
```

`python-multipart` is required for dataset upload; `openpyxl` is required only
for `.xlsx` ground-truth files (CSV and JSON work without it).

## The five workspaces (top tabs)

### 🏫 Run panel
Enter a survey, target audience, and panel size, choose the **source mix**, and
watch the run flow through four phases: **Generating → Screening Room →
Survey Hall → Insights**. Click any bear to read its persona; click the
Insights Bear for the aggregated report.

The **Screening Room** is a decorated, animated research facility: a
"Bear Screening Room" banner, entry/exit doors, screening desks with
clipboards, a waiting area with chairs and a table, a coffee station, wall
posters, and plants. Bears **walk around, stop to chat, look about the room,
and drift toward the screening desks**. Qualified bears briefly **celebrate**;
disqualified bears **walk out through the glowing exit door**. The animation is
lightweight (simple sprites + slot layout) and scales to large panels.

### 🎨 Character Studio
Hand-build a synthetic respondent: name, age, gender, location, occupation,
education, income, family status, interests, preferred brands, sustainability
attitude, tech adoption, and a free-text personality description. Save it to the
library, mark it a reusable **template**, or **lock** it so it joins every
future study.

### 📇 Persona Library
A central repository of saved personas. Search, filter by tag, lock/unlock,
delete, or push a persona into your next run as a custom character.

### 📊 Ground Truth
Upload a real survey dataset (**CSV / XLSX / JSON**). The platform auto-detects
demographic columns, models each column's distribution, and shows the detected
mapping plus distribution/response bars. This one dataset drives calibration,
audience matching, and validation everywhere else.

### 📈 Dashboard
Researcher metrics for the latest run: total / qualified / disqualified bears,
average age, gender & province & source composition, token usage, an estimated
hosted-model cost, runtime, and the ground-truth similarity score. Includes the
full **validation** block (below).

## Ground-truth calibration (the core feature)

When a dataset is loaded and **Calibrate** is on:

* **Persona calibration** — synthetic bears sample their demographics from the
  real distributions, so the panel statistically resembles your respondents.
* **Response calibration** — a compact summary of observed answer patterns is
  injected into each respondent's prompt, nudging answers toward real-world
  frequencies without overriding the persona.
* **Audience matching** — the **historical** slice of the panel is built from
  real dataset rows; those bears voice their *actual* recorded answers.
* **Validation dashboard** — compares synthetic vs real distributions across
  Age, Gender, Geography, Sustainability, and Tech adoption, with a per-dimension
  similarity score (1 − total-variation distance) and an overall figure.
* **Insight confidence score (0–100)** — combines ground-truth sample size,
  panel size, response similarity, and demographic coverage into a
  High / Moderate / Low rating shown on every study.

## Hybrid panel mode

The panel mix sliders set the proportion of three sources (proportions are
normalized, so they need not sum to 100):

| Source | Colour | Meaning |
|---|---|---|
| **Synthetic** | green | AI-generated bears (calibrated when ground truth is on) |
| **Custom** | amber | your Character Studio creations + locked personas |
| **Historical** | blue | real respondent profiles sampled from the dataset |

Locked library personas always join, regardless of the mix.

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/assign` | Start a run. Body: `survey`, `panel_size`, `audience`, `source_mix`, `custom_characters`, `calibrated` |
| `GET` | `/status` | Live state of every bear + `_run`, `_performance`, `_validation`, `_dashboard`, `_ground_truth`, `_interactions` |
| `GET` | `/roster` | Compact roster |
| `POST`/`GET`/`DELETE` | `/ground_truth` | Upload / inspect / clear the calibration dataset |
| `GET` | `/validation` | Latest validation report |
| `GET` | `/dashboard` | Latest dashboard metrics |
| `GET`/`POST` | `/personas` | List (with `?q=` / `?tag=`) or save a persona |
| `PUT`/`DELETE` | `/personas/{id}` | Update or delete a persona |
| `POST` | `/personas/{id}/lock` | Toggle "always participates" |
| `POST` | `/preview_persona` | Normalize a Character Studio draft without saving |

## Design notes

* The roster is never hardcoded — every run builds a fresh set of bears sized by
  `panel_size` and the source mix.
* `panel_size` is capped (`MAX_PANEL_SIZE`, default 500) and answers run in
  parallel (`MAX_PARALLEL_AGENTS`, default 6) to stay friendly to local Ollama.
* Similarity uses distribution overlap (1 − total-variation distance), which is
  robust and easy to read. The cost estimate is a "what would this have cost on
  a hosted model?" figure — local Ollama itself is free.
* Config via env vars: `OLLAMA_MODEL`, `OLLAMA_BASE_URL`, `MAX_PANEL_SIZE`,
  `MAX_PARALLEL_AGENTS`, `COST_PER_1M_PROMPT`, `COST_PER_1M_COMPLETION`.
