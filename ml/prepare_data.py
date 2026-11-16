import pandas as pd, numpy as np, ast, json

CSV = '/mnt/user-data/uploads/Stravl_Travel_Preference_Data.csv'
DEST_FILE = '/mnt/user-data/uploads/destination_ids.txt'

def parselist(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return []
    return []

print("Loading CSV...")
df = pd.read_csv(CSV)

with open(DEST_FILE) as f:
    dest_names = f.read().splitlines()
N_DEST = len(dest_names)
print("Num destinations:", N_DEST)

FORM_CARD = {
    'form_a': 4, 'form_b': 4, 'form_c': 4,
    'form_f': 8, 'form_g': 8,
    'form_h': 3, 'form_i': 3, 'form_j': 3, 'form_k': 3,
}
FEAT_DIM = sum(FORM_CARD.values())
print("User feature dim:", FEAT_DIM)

def encode_user_row(row):
    vec = np.zeros(FEAT_DIM, dtype=np.float32)
    offset = 0
    for col, card in FORM_CARD.items():
        codes = parselist(row[col])
        for c in codes:
            try:
                idx = int(c)
                if 0 <= idx < card:
                    vec[offset + idx] = 1.0
            except Exception:
                pass
        offset += card
    return vec

print("Encoding user features...")
user_feats = np.zeros((len(df), FEAT_DIM), dtype=np.float32)
for i, row in enumerate(df.itertuples(index=False)):
    row = row._asdict() if hasattr(row, '_asdict') else None

# faster: vectorized per column
offset = 0
col_slices = {}
for col, card in FORM_CARD.items():
    col_slices[col] = (offset, card)
    offset += card

for col, (off, card) in col_slices.items():
    series = df[col].apply(parselist)
    for i, codes in enumerate(series):
        for c in codes:
            try:
                idx = int(c)
                if 0 <= idx < card:
                    user_feats[i, off+idx] = 1.0
            except Exception:
                pass
    print(f"  encoded {col}")

print("Parsing swipes...")
yes = df['yes_swipes'].apply(parselist)
no = df['no_swipes'].apply(parselist)
maybe = df['maybe_swipes'].apply(parselist)

rows = []  # (user_idx, dest_id, label)
for i in range(len(df)):
    for d in yes[i]:
        rows.append((i, int(d), 1))
    for d in no[i]:
        rows.append((i, int(d), 0))
    for d in maybe[i]:
        rows.append((i, int(d), 1))  # weak positive, will downweight

inter = np.array(rows, dtype=np.int64)
print("Total interactions:", inter.shape)

np.save('user_feats.npy', user_feats)
np.save('interactions.npy', inter)
with open('dest_names.json', 'w') as f:
    json.dump(dest_names, f)
with open('meta.json', 'w') as f:
    json.dump({'FEAT_DIM': FEAT_DIM, 'N_DEST': N_DEST, 'N_USERS': len(df),
               'FORM_CARD': FORM_CARD}, f)

print("Done. Saved user_feats.npy, interactions.npy, dest_names.json, meta.json")
