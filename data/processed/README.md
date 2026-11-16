# Data cleaning & transformation — what was done, exactly

Source: `data/raw/Stravl_Travel_Preference_Data.csv` (80,301 rows, 38 columns) +
`data/raw/destination_ids.txt` (1,486 real destination names, one per line, 0-indexed).

## 1. Parsing
Several CSV columns (`form_a`, `form_b`, `form_c`, `form_f`, `form_g`, `form_h`, `form_i`,
`form_j`, `form_k`, `yes_swipes`, `no_swipes`, `maybe_swipes`) are stored as Python-list
literals inside strings, e.g. `"['0', '1']"`. These were parsed with `ast.literal_eval`
(never `eval`) into real lists, with empty/malformed cells treated as `[]`.

## 2. Dropped columns
`form_r`, `form_rr` were dropped — 78% and 84% missing respectively, too sparse to encode
reliably. `Model`, `Retrieval`, `DynaMatch`, `Rating_0..9`, `Rec_0..9` were inspected and
found to be placeholder columns (value `-1` in the overwhelming majority of rows, reserved
by Stravl for a different evaluation they never populated in this release) — dropped as
unusable.

## 3. Destination ID resolution
Every destination ID appearing in `yes_swipes` / `no_swipes` / `maybe_swipes` was
cross-referenced against `destination_ids.txt` (1,486 entries, IDs 0–1485) to attach a
real place name. All 1,452+ destinations actually referenced in swipe data resolved
successfully — no missing or out-of-range IDs.

## 4. Feature encoding (`user_feats.npy`)
Each of the 9 usable survey fields was multi-hot encoded (a field can have more than one
answer selected, so this is not strict one-hot):

| Field | Options | Encoded width |
|---|---|---|
| form_a | 4 | 4 |
| form_b | 4 | 4 |
| form_c | 4 | 4 |
| form_f | 8 | 8 |
| form_g | 8 | 8 |
| form_h | 3 | 3 |
| form_i | 3 | 3 |
| form_j | 3 | 3 |
| form_k | 3 | 3 |

Total per-user feature vector: **40 dimensions**, stored as `user_feats.npy`
(80,301 × 40, float32).

Human-readable question wording for each field is our best-effort interpretation of what
each anonymized code block represents (travel frequency, trip length, budget, interests,
trip style, companions, season, accommodation, climate) — the exact original Stravl survey
wording was not publicly available at build time. The model trains on the anonymized codes
either way, so this only affects how things are *labeled* for humans reading the CSVs and
using the demo, not what the network learns.

## 5. Interaction table (`interactions.npy`)
Every (user, destination) swipe was expanded into one row: `yes` → label 1, `no` → label 0,
`maybe` → label 1 with reduced confidence (treated as a weak positive). No swipes were
discarded and no synthetic negatives were generated — negatives are the real "no" swipes
users gave. Result: **851,133 labeled interactions**.

## 6. Train / test split
Interactions were shuffled and split 85% / 15% (`train_interactions.npy`,
`test_interactions.npy`) at the interaction level, seeded for reproducibility.

## 7. Human-readable exports
- `users_cleaned.csv` — same 80,301 users, with every form_x code list decoded into
  plain-English labels (semicolon-separated where multiple were selected) plus swipe counts.
- `interactions_cleaned.csv` — all 851,133 swipes with the real destination name attached
  and the label spelled out (`yes` / `no` / `maybe`), for anyone who wants to inspect or
  re-derive features without touching the raw list-encoded columns.
- `destinations_enriched.csv` — one row per destination with a **derived** profile:
  the most common survey answer (and its confidence, i.e. the share of fans who gave that
  answer) among users who swiped "yes" on it, for each of the 9 fields, plus its total
  yes-swipe count. 1,437 of 1,486 destinations had ≥5 "yes" swipes and got a profile;
  destinations below that threshold are left blank rather than guessed.
