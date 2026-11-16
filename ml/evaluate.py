import numpy as np, json

user_feats = np.load('user_feats.npy')
meta = json.load(open('meta.json'))
FEAT_DIM = meta['FEAT_DIM']; N_DEST = meta['N_DEST']
train_i = np.load('train_interactions.npy')
test_i = np.load('test_interactions.npy')
params = dict(np.load('trained_params.npz'))
rng = np.random.default_rng(7)

def sigmoid(x): return 1.0/(1.0+np.exp(-x))

def forward(u_idx, d_idx):
    uf = user_feats[u_idx]
    de = params['E'][d_idx]
    x0 = np.concatenate([uf, de], axis=1)
    a1 = np.maximum(0, x0 @ params['W1'] + params['b1'])
    a2 = np.maximum(0, a1 @ params['W2'] + params['b2'])
    z3 = a2 @ params['W3'] + params['b3']
    return sigmoid(z3).flatten()

# popularity baseline: yes-rate per destination from TRAIN
pop_yes = np.zeros(N_DEST); pop_tot = np.zeros(N_DEST)
for u,d,y in train_i:
    pop_tot[d]+=1
    if y==1: pop_yes[d]+=1
pop_score = pop_yes / np.maximum(pop_tot,1)

# --- Leave-one-out style ranking eval: for each positive TEST interaction,
# rank it against 99 random destinations the user did NOT swipe (train or test) ---
seen = {}
for u,d,y in np.vstack([train_i, test_i]):
    seen.setdefault(u, set()).add(d)

test_pos = test_i[test_i[:,2]==1]
sample_n = 4000
sel = rng.choice(len(test_pos), size=min(sample_n, len(test_pos)), replace=False)

K = 10
def eval_model(score_fn, label):
    hits, ndcgs, precs, recs = [], [], [], []
    for idx in sel:
        u, true_d, _ = test_pos[idx]
        u = int(u); true_d = int(true_d)
        negs = []
        while len(negs) < 99:
            cand = rng.integers(0, N_DEST)
            if cand not in seen.get(u, set()) and cand != true_d:
                negs.append(cand)
        cand_items = np.array([true_d] + negs)
        scores = score_fn(u, cand_items)
        order = np.argsort(-scores)
        rank = np.where(order == 0)[0][0]  # position of true_d (index 0 in cand_items)
        hit = 1 if rank < K else 0
        hits.append(hit)
        ndcgs.append(1.0/np.log2(rank+2) if rank < K else 0.0)
        precs.append(hit / K)
        recs.append(hit / 1.0)
    print(f"{label}: HitRate@{K}={np.mean(hits):.4f}  NDCG@{K}={np.mean(ndcgs):.4f}  Precision@{K}={np.mean(precs):.4f}  Recall@{K}={np.mean(recs):.4f}")

def nn_score(u, cand_items):
    u_arr = np.full(len(cand_items), u)
    return forward(u_arr, cand_items)

def pop_score_fn(u, cand_items):
    return pop_score[cand_items]

eval_model(nn_score, "Neural Network (MLP + learned destination embeddings)")
eval_model(pop_score_fn, "Popularity baseline (global yes-rate)")

# random baseline
def rand_score_fn(u, cand_items):
    return rng.random(len(cand_items))
eval_model(rand_score_fn, "Random baseline")
