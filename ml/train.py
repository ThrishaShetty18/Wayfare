import numpy as np, json, time

np.random.seed(42)

user_feats = np.load('user_feats.npy')          # (N_USERS, 40)
inter = np.load('interactions.npy')              # (M, 3): user_idx, dest_id, label
meta = json.load(open('meta.json'))
FEAT_DIM = meta['FEAT_DIM']
N_DEST = meta['N_DEST']

print("Interactions:", inter.shape, "Users:", user_feats.shape)

# ---- train/test split (interaction-level, stratified by label) ----
rng = np.random.default_rng(42)
idx = rng.permutation(len(inter))
inter = inter[idx]
split = int(0.85 * len(inter))
train_i, test_i = inter[:split], inter[split:]
print("Train:", train_i.shape, "Test:", test_i.shape)

EMB_DIM = 16
H1, H2 = 64, 32
IN_DIM = FEAT_DIM + EMB_DIM

def init_w(fan_in, fan_out):
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float32)

params = {
    'E': (rng.normal(0, 0.05, size=(N_DEST, EMB_DIM))).astype(np.float32),
    'W1': init_w(IN_DIM, H1), 'b1': np.zeros(H1, dtype=np.float32),
    'W2': init_w(H1, H2), 'b2': np.zeros(H2, dtype=np.float32),
    'W3': init_w(H2, 1), 'b3': np.zeros(1, dtype=np.float32),
}

# Adam optimizer state
m = {k: np.zeros_like(v) for k, v in params.items()}
v = {k: np.zeros_like(v) for k, v in params.items()}
beta1, beta2, eps = 0.9, 0.999, 1e-8
lr = 0.003
t = 0

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def forward(u_idx, d_idx, training=True, drop_p=0.3):
    uf = user_feats[u_idx]                 # (B, FEAT_DIM)
    de = params['E'][d_idx]                # (B, EMB_DIM)
    x0 = np.concatenate([uf, de], axis=1)  # (B, IN_DIM)
    z1 = x0 @ params['W1'] + params['b1']
    a1 = np.maximum(0, z1)
    mask1 = None
    if training:
        mask1 = (rng.random(a1.shape) > drop_p).astype(np.float32) / (1 - drop_p)
        a1 = a1 * mask1
    z2 = a1 @ params['W2'] + params['b2']
    a2 = np.maximum(0, z2)
    mask2 = None
    if training:
        mask2 = (rng.random(a2.shape) > drop_p).astype(np.float32) / (1 - drop_p)
        a2 = a2 * mask2
    z3 = a2 @ params['W3'] + params['b3']
    p = sigmoid(z3).flatten()
    cache = (x0, uf, de, z1, a1, mask1, z2, a2, mask2, z3)
    return p, cache

def backward(cache, y, u_idx, d_idx):
    x0, uf, de, z1, a1, mask1, z2, a2, mask2, z3 = cache
    B = len(y)
    p = sigmoid(z3).flatten()
    dz3 = (p - y).reshape(-1, 1) / B          # BCE + sigmoid grad
    grads = {}
    grads['W3'] = a2.T @ dz3
    grads['b3'] = dz3.sum(axis=0)
    da2 = dz3 @ params['W3'].T
    if mask2 is not None: da2 = da2 * mask2
    dz2 = da2 * (z2 > 0)
    grads['W2'] = a1.T @ dz2
    grads['b2'] = dz2.sum(axis=0)
    da1 = dz2 @ params['W2'].T
    if mask1 is not None: da1 = da1 * mask1
    dz1 = da1 * (z1 > 0)
    grads['W1'] = x0.T @ dz1
    grads['b1'] = dz1.sum(axis=0)
    dx0 = dz1 @ params['W1'].T
    dde = dx0[:, FEAT_DIM:]
    gE = np.zeros_like(params['E'])
    np.add.at(gE, d_idx, dde)
    grads['E'] = gE
    return grads

def adam_step(grads):
    global t
    t += 1
    for k in params:
        m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
        v[k] = beta2 * v[k] + (1 - beta2) * (grads[k] ** 2)
        mhat = m[k] / (1 - beta1 ** t)
        vhat = v[k] / (1 - beta2 ** t)
        params[k] -= lr * mhat / (np.sqrt(vhat) + eps)

def bce_loss(p, y):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

BATCH = 2048
EPOCHS = 12
n_train = len(train_i)

print("Training...")
t0 = time.time()
for epoch in range(EPOCHS):
    perm = rng.permutation(n_train)
    ep_loss = 0.0
    nb = 0
    for start in range(0, n_train, BATCH):
        b_idx = perm[start:start+BATCH]
        batch = train_i[b_idx]
        u_idx, d_idx, y = batch[:, 0], batch[:, 1], batch[:, 2].astype(np.float32)
        p, cache = forward(u_idx, d_idx, training=True)
        loss = bce_loss(p, y)
        grads = backward(cache, y, u_idx, d_idx)
        adam_step(grads)
        ep_loss += loss
        nb += 1
    # quick test loss
    tu, td, ty = test_i[:20000,0], test_i[:20000,1], test_i[:20000,2].astype(np.float32)
    tp, _ = forward(tu, td, training=False)
    test_loss = bce_loss(tp, ty)
    test_acc = ((tp > 0.5).astype(np.float32) == ty).mean()
    print(f"Epoch {epoch+1}/{EPOCHS}  train_loss={ep_loss/nb:.4f}  test_loss={test_loss:.4f}  test_acc={test_acc:.4f}  ({time.time()-t0:.1f}s)")

np.savez('trained_params.npz', **params)
np.save('test_interactions.npy', test_i)
np.save('train_interactions.npy', train_i)
print("Saved trained_params.npz")
