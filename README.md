# Wayfare — Neural Network Travel Destination Recommender

A complete, submittable NNDL project: real dataset → cleaned/transformed data →
from-scratch neural network (trained via backprop, no black-box library) →
evaluated ranking engine → a real full-stack web app (Flask API + frontend) →
plus dependency-free standalone demos → written report.

## Folder structure

```
Wayfare_Project_Final/
├── data/
│   ├── raw/                    original, unmodified uploaded files
│   └── processed/              cleaned + transformed data (see data/processed/README.md)
│
├── ml/                         the neural network, trained from scratch
│   ├── prepare_data.py         raw CSV -> cleaned/encoded arrays
│   ├── train.py                NumPy MLP + embeddings, backprop, Adam
│   └── evaluate.py             ranking eval vs popularity/random baselines
│
├── model/                      trained weights, shared reference copy
│   ├── trained_params.npz
│   ├── model_export.json       same weights as JSON (used by standalone demos)
│   └── dest_names.json
│
├── app/                        ⭐ THE FULL-STACK APP — frontend + backend connected
│   ├── backend/                 Flask API that loads the trained model and serves it
│   │   ├── app.py                routes: /api/health, /api/questions, /api/recommend
│   │   ├── inference.py          model loading + the forward pass (readable, ~100 lines)
│   │   └── requirements.txt
│   ├── frontend/                 plain HTML/CSS/JS that calls the backend over HTTP
│   │   ├── index.html
│   │   ├── style.css
│   │   └── app.js                fetch()'s /api/questions and POSTs to /api/recommend
│   └── model/                    the backend's own copy of the weights it loads at startup
│       ├── trained_params.npz
│       └── dest_names.json
│
├── standalone_demo/            no backend needed — model weights embedded directly in HTML
│   ├── wayfare_demo.html        minimal single-purpose demo
│   └── wayfare_showcase.html    full presentation: hero, architecture diagram, live demo,
│                                 evaluation chart, data-derived destination cards
│
└── report/
    └── Wayfare_Project_Report.docx
```

## Two different ways this ships, and why both exist

1. **`app/` — the real full-stack version.** The frontend never sees the model weights.
   It calls a Flask REST API (`/api/recommend`), which loads `trained_params.npz` once at
   startup and runs the forward pass server-side, in NumPy. This is the architecture you'd
   actually deploy, and the one to point to if asked "where does the frontend talk to the
   backend."
2. **`standalone_demo/` — zero-install fallback.** The exact same trained weights, exported
   to JSON and embedded directly in a single HTML file, with the identical forward pass
   re-implemented in vanilla JavaScript. No server, no install — just open the file. Useful
   for a quick demo on a machine without Python set up, or to submit as a single artifact.

Both run the *same trained model* and will give identical rankings for the same answers —
they're two different serving strategies for one artifact, not two different models.

## Running the full-stack app

```bash
cd app/backend
pip install -r requirements.txt
python app.py
```
Then open **http://localhost:5000** — Flask serves the frontend directly, so there's
nothing else to start.

The frontend is now a small three-screen product rather than a single form:
- **Home** — a one-question-per-screen wizard (progress bar, back/next) instead of one
  long form.
- **Results dashboard** — the top 9 matches as a photo grid, ranked by match confidence,
  with quick tags for budget style and top draw.
- **Destination page** (`#/destination/<id>`) — its own routed screen with a photo
  gallery, "what this place is known for" facts pulled from `destinations_enriched.csv`,
  and an **estimated trip budget** built from your own survey answers (budget style, trip
  length, travel party) and converted to your local currency.

Destination photos come from Wikipedia's public REST API (no key needed) and currency
detection/conversion happens client-side via `ipapi.co` and `frankfurter.app` — the Flask
backend never touches either; it only computes the USD estimate. All three are free,
keyless public APIs, so nothing needs to be configured to try this locally.

API quick-check once it's running:
```bash
curl http://localhost:5000/api/health
curl -X POST http://localhost:5000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"answers": {"form_c": [2], "form_f": [0,5]}, "top_k": 5}'
curl http://localhost:5000/api/destination/430
curl -X POST http://localhost:5000/api/estimate \
  -H "Content-Type: application/json" \
  -d '{"answers": {"form_b": [1], "form_c": [2], "form_h": [1]}, "destination_id": 430}'
```

## Reproducing the ML pipeline from scratch

```bash
cd ml
python3 prepare_data.py    # reads ../data/raw, writes cleaned arrays here
python3 train.py           # trains the network, writes trained_params.npz
python3 evaluate.py        # prints HitRate@10 / NDCG@10 / Precision@10 / Recall@10
```
Outputs are already pre-computed in `data/processed/` and `model/`, so you can run the app
or open the standalone demo without re-running this. If you do retrain, copy the new
`trained_params.npz` into both `model/` and `app/model/` (and regenerate
`model/model_export.json` — see the bottom of `ml/train.py`'s companion export step in the
report) so every part of the project stays in sync.

## Honesty note on "destination features" (budget / climate / etc.)

The raw Stravl dataset has **no ground-truth per-destination attributes** — there's no
official "this city's climate is X" column anywhere in what was released. What
`data/processed/destinations_enriched.csv` contains instead is a *derived* profile:
for each destination, the aggregated real survey answers of the users who actually swiped
"yes" on it (e.g. "68% of people who liked Tirana said they prefer a mid-range budget").
That's genuine, data-grounded signal — not an invented fact — but it describes *the people
who liked the place*, not an official attribute of the place. The neural network itself
never sees hand-labeled destination attributes at all: destination "characteristics" are a
16-dimensional embedding learned purely from co-occurrence patterns in the 851,133 real
swipes. Every destination *name* everywhere in this project — data, model, both UIs, report
— is real, taken from the official `destination_ids.txt` lookup table.
