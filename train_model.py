"""
Product recommendation system — Optimized Matrix Factorization
Features:
  [1] Early Stopping       — stop when validation stops improving to avoid overfit
  [2] Learning Rate Decay  — decay lr each epoch for smoother convergence
  [3] BPR Ranking Loss     — optimize ranking directly (in addition to MSE)
  [4] Popularity Penalty   — down-weight popular items to improve coverage
  [5] Score Normalization  — normalize scores to [0,1] before training
  [6] Grid Search          — find best hyperparameters on validation set
  [*] Denormalize metrics — evaluate on original score scale for interpretability
"""

import numpy as np
import pandas as pd
import random
import math
from collections import defaultdict

random.seed(42)
np.random.seed(42)

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 70)
print("LOAD DATASETS")
print("=" * 70)

df_products  = pd.read_csv('datasets/products.csv')
df_customers = pd.read_csv('datasets/customers.csv')
interaction  = pd.read_csv('datasets/user_item_interactions.csv')

print(f"  products       : {len(df_products)} rows")
print(f"  customers      : {len(df_customers)} rows")
print(f"  interactions   : {len(interaction)} rows")

# ============================================================
# [5] SCORE NORMALIZATION TO [0, 1]
# Keeps SGD stable when targets lie in a bounded range
# ============================================================
s_min = interaction['score'].min()
s_max = interaction['score'].max()
interaction['score_norm'] = (interaction['score'] - s_min) / (s_max - s_min)

print(f"\n[5] Score normalization: [{s_min:.3f}, {s_max:.3f}] -> [0.000, 1.000]")

# ============================================================
# INDEX MAPPING
# ============================================================
all_users = sorted(interaction['customer_id'].unique())
all_items = sorted(interaction['product_id'].unique())
user2idx  = {u: i for i, u in enumerate(all_users)}
item2idx  = {p: i for i, p in enumerate(all_items)}
idx2item  = {i: p for p, i in item2idx.items()}
n_u, n_i  = len(all_users), len(all_items)

records = [
    (user2idx[r.customer_id], item2idx[r.product_id], float(r.score_norm))
    for r in interaction.itertuples()
]

random.shuffle(records)
n       = len(records)
n_train = int(0.70 * n)
n_val   = int(0.15 * n)
train_data = records[:n_train]
val_data   = records[n_train:n_train + n_val]
test_data  = records[n_train + n_val:]

user_seen = defaultdict(set)
for u, i, _ in train_data:
    user_seen[u].add(i)

# ============================================================
# [4] POPULARITY PENALTY WEIGHTS
# Popular items get lower weight -> less dominance -> better coverage
# ============================================================
item_count = defaultdict(int)
for u, i, _ in train_data:
    item_count[i] += 1
max_count    = max(item_count.values()) if item_count else 1
item_weights = {
    i: 1.0 / (1.0 + math.log1p(item_count.get(i, 0) / max_count))
    for i in range(n_i)
}
print(f"[4] Popularity penalty: weight range [{min(item_weights.values()):.3f}, {max(item_weights.values()):.3f}]")


# ============================================================
# MODEL: Matrix Factorization with all optimizations
# ============================================================
class MatrixFactorization:
    """
    Matrix Factorization with:
    - Early Stopping [1]
    - Learning Rate Decay [2]
    - Popularity-weighted loss [4]
    - Score normalization support [5]
    """

    def __init__(self, n_users, n_items, n_factors=30,
                 lr=0.008, reg=0.015, n_epochs=60,
                 patience=8, lr_decay=0.97,
                 item_weights=None, verbose=True):
        self.n_users      = n_users
        self.n_items      = n_items
        self.n_factors    = n_factors
        self.lr           = lr
        self.lr_init      = lr
        self.reg          = reg
        self.n_epochs     = n_epochs
        self.patience     = patience       # [1] Early stopping patience
        self.lr_decay     = lr_decay       # [2] Multiply lr by this each epoch
        self.item_weights = item_weights   # [4] Per-item loss weights
        self.verbose      = verbose

        # He initialization — better than uniform random for MF
        scale = math.sqrt(2.0 / n_factors)
        self.U    = np.random.normal(0, scale, (n_users, n_factors))
        self.V    = np.random.normal(0, scale, (n_items, n_factors))
        self.b_u  = np.zeros(n_users)
        self.b_i  = np.zeros(n_items)
        self.b_g  = 0.0

        self.train_losses, self.val_losses   = [], []
        self.train_rmse,   self.val_rmse     = [], []
        self.best_U = self.best_V = None
        self.best_b_u = self.best_b_i = None
        self.best_val_loss = float('inf')
        self.best_epoch    = 0
        self.stopped_epoch = n_epochs

    def _pred(self, u, i):
        return self.b_g + self.b_u[u] + self.b_i[i] + np.dot(self.U[u], self.V[i])

    def predict_batch(self, pairs):
        return np.array([self._pred(u, i) for u, i in pairs])

    def _save_best(self, epoch):
        self.best_U   = self.U.copy()
        self.best_V   = self.V.copy()
        self.best_b_u = self.b_u.copy()
        self.best_b_i = self.b_i.copy()
        self.best_b_g = self.b_g
        self.best_epoch = epoch

    def _restore_best(self):
        self.U   = self.best_U
        self.V   = self.best_V
        self.b_u = self.best_b_u
        self.b_i = self.best_b_i
        self.b_g = self.best_b_g

    def fit(self, train_data, val_data, bpr_weight=0.3):
        """
        bpr_weight: fraction of BPR loss blended into SGD update.
        BPR = Bayesian Personalized Ranking: for each (u, i_pos),
        sample random i_neg not bought by u -> optimize pred(u,i_pos) > pred(u,i_neg).
        """
        self.b_g = float(np.mean([r for *_, r in train_data]))

        no_improve = 0
        t_pairs = [(u, i) for u, i, _ in train_data]
        v_pairs = [(u, i) for u, i, _ in val_data]
        t_act   = np.array([r for _, _, r in train_data])
        v_act   = np.array([r for _, _, r in val_data])

        # Build per-user positive item sets for BPR sampling [3]
        user_pos = defaultdict(list)
        for u, i, _ in train_data:
            user_pos[u].append(i)

        for epoch in range(self.n_epochs):
            random.shuffle(train_data)

            for u, i, r in train_data:
                w    = self.item_weights[i] if self.item_weights else 1.0
                pred = self._pred(u, i)
                err  = r - pred

                # MSE gradient
                du_mse = w * err * self.V[i] - self.reg * self.U[u]
                dv_mse = w * err * self.U[u] - self.reg * self.V[i]
                dbu    = w * err - self.reg * self.b_u[u]
                dbi    = w * err - self.reg * self.b_i[i]

                # [3] BPR gradient (ranking loss)
                du_bpr = np.zeros(self.n_factors)
                dv_pos = np.zeros(self.n_factors)
                dv_neg = np.zeros(self.n_factors)
                if bpr_weight > 0 and len(user_pos[u]) > 0:
                    # Sample negative item (not bought by u)
                    j = random.randint(0, self.n_items - 1)
                    attempts = 0
                    while j in user_seen.get(u, set()) and attempts < 5:
                        j = random.randint(0, self.n_items - 1)
                        attempts += 1
                    if j not in user_seen.get(u, set()):
                        x_uij = self._pred(u, i) - self._pred(u, j)
                        # sigmoid derivative: sigma'(x) = sigma(x)(1-sigma(x))
                        sigmoid = 1.0 / (1.0 + math.exp(-x_uij))
                        grad    = 1.0 - sigmoid   # gradient of log-sigmoid
                        du_bpr  = grad * (self.V[i] - self.V[j]) - self.reg * self.U[u]
                        dv_pos  = grad * self.U[u]  - self.reg * self.V[i]
                        dv_neg  = -grad * self.U[u] - self.reg * self.V[j]

                # Combine gradients
                a = 1.0 - bpr_weight
                b = bpr_weight
                self.b_g    += self.lr * err * w * 0.05
                self.b_u[u] += self.lr * dbu
                self.b_i[i] += self.lr * dbi
                self.U[u]   += self.lr * (a * du_mse + b * du_bpr)
                self.V[i]   += self.lr * (a * dv_mse + b * dv_pos)
                if bpr_weight > 0:
                    self.V[j]   += self.lr * b * dv_neg

            # [2] Learning Rate Decay
            self.lr *= self.lr_decay

            # Metrics
            tl = float(np.mean((t_act - self.predict_batch(t_pairs)) ** 2))
            vl = float(np.mean((v_act - self.predict_batch(v_pairs)) ** 2))
            self.train_losses.append(tl); self.val_losses.append(vl)
            self.train_rmse.append(math.sqrt(tl))
            self.val_rmse.append(math.sqrt(vl))

            # [1] Early Stopping
            if vl < self.best_val_loss - 1e-5:
                self.best_val_loss = vl
                self._save_best(epoch + 1)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    self.stopped_epoch = epoch + 1
                    if self.verbose:
                        print(f"  Early stop at epoch {epoch+1} "
                              f"(best={self.best_epoch}, val_rmse={math.sqrt(self.best_val_loss):.4f})")
                    break

            if self.verbose and (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1:>3} | "
                      f"train={math.sqrt(tl):.4f} | val={math.sqrt(vl):.4f} | "
                      f"lr={self.lr:.5f} | no_improve={no_improve}")

        # Restore best weights [1]
        if self.best_U is not None:
            self._restore_best()

        return self

    def recommend(self, u, n=10, exclude=None):
        scores = [(i, self._pred(u, i))
                  for i in range(self.n_items)
                  if not (exclude and i in exclude)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]


# ============================================================
# [6] GRID SEARCH — find best hyperparameters on validation set
# ============================================================
print("\n" + "=" * 70)
print("[6] GRID SEARCH HYPERPARAMETERS")
print("=" * 70)

param_grid = [
    {'n_factors': 20, 'lr': 0.008, 'reg': 0.015, 'bpr': 0.0},
    {'n_factors': 30, 'lr': 0.008, 'reg': 0.015, 'bpr': 0.0},
    {'n_factors': 30, 'lr': 0.008, 'reg': 0.015, 'bpr': 0.2},
    {'n_factors': 40, 'lr': 0.006, 'reg': 0.02,  'bpr': 0.2},
    {'n_factors': 40, 'lr': 0.006, 'reg': 0.02,  'bpr': 0.3},
    {'n_factors': 50, 'lr': 0.005, 'reg': 0.02,  'bpr': 0.3},
]

print(f"\n  {'Config':>46} | {'Val RMSE':>10} | {'Best Ep':>8}")
print("  " + "-" * 72)

best_config    = None
best_val_score = float('inf')

for cfg in param_grid:
    np.random.seed(42); random.seed(42)
    m = MatrixFactorization(
        n_u, n_i,
        n_factors=cfg['n_factors'],
        lr=cfg['lr'], reg=cfg['reg'],
        n_epochs=80, patience=10, lr_decay=0.97,
        item_weights=item_weights,
        verbose=False
    )
    m.fit(train_data, val_data, bpr_weight=cfg['bpr'])
    vr = min(m.val_rmse)
    label = f"factors={cfg['n_factors']}, lr={cfg['lr']}, reg={cfg['reg']}, bpr={cfg['bpr']}"
    marker = " < BEST" if vr < best_val_score else ""
    print(f"  {label:>46} | {vr:>10.4f} | {m.best_epoch:>8}{marker}")
    if vr < best_val_score:
        best_val_score = vr
        best_config    = cfg

print(f"\n  Best config: {best_config}")


# ============================================================
# TRAIN FINAL MODEL WITH BEST CONFIG
# ============================================================
print("\n" + "=" * 70)
print("TRAIN FINAL MODEL WITH BEST CONFIG")
print("=" * 70)
print(f"\n  n_factors={best_config['n_factors']}, lr={best_config['lr']}, reg={best_config['reg']}, bpr={best_config['bpr']}")
print(f"  + Early Stopping (patience=10)")
print(f"  + LR Decay (x0.97/epoch)")
print(f"  + BPR Ranking Loss (weight={best_config['bpr']}) [3]")
print(f"  + Popularity Penalty weights [4]")
print(f"  + Score Normalization [0,1] [5]\n")

np.random.seed(42); random.seed(42)
model = MatrixFactorization(
    n_u, n_i,
    n_factors=best_config['n_factors'],
    lr=best_config['lr'],
    reg=best_config['reg'],
    n_epochs=80, patience=10, lr_decay=0.97,
    item_weights=item_weights,
    verbose=True
)
model.fit(train_data, val_data, bpr_weight=best_config['bpr'])


# ============================================================
# EVALUATION — denormalize to original scale for interpretability
# ============================================================
print("\n" + "=" * 70)
print("EVALUATE MODEL ON TEST SET")
print("=" * 70)

t_pairs   = [(u, i) for u, i, _ in test_data]
# Predictions and actuals are on normalized scale [0,1]
t_pred_n  = model.predict_batch(t_pairs)
t_act_n   = np.array([r for _, _, r in test_data])

# Denormalize to original score scale for meaningful metrics
t_pred    = np.clip(t_pred_n, 0, 1) * (s_max - s_min) + s_min
t_act     = t_act_n * (s_max - s_min) + s_min

mse      = float(np.mean((t_act - t_pred) ** 2))
rmse     = math.sqrt(mse)
mae      = float(np.mean(np.abs(t_act - t_pred)))
ss_res   = float(np.sum((t_act - t_pred) ** 2))
ss_tot   = float(np.sum((t_act - t_act.mean()) ** 2))
r2       = 1 - ss_res / ss_tot if ss_tot > 0 else 0

print(f"\n Regression Metrics (original scale, after denormalize):")
print(f"   {'MSE':<10}: {mse:.4f}")
print(f"   {'RMSE':<10}: {rmse:.4f}")
print(f"   {'MAE':<10}: {mae:.4f}")
print(f"   {'R2':<10}: {r2:.4f}")
print(f"\n Validation RMSE (normalized [0,1]): {min(model.val_rmse):.4f} (best @ epoch {model.best_epoch})")

# Ranking metrics
def precision_at_k(rec, rel, k):
    return len(set(rec[:k]) & rel) / k

def recall_at_k(rec, rel, k):
    return len(set(rec[:k]) & rel) / len(rel) if rel else 0

def ndcg_at_k(rec, rel, k):
    dcg  = sum(1 / math.log2(r + 2) for r, it in enumerate(rec[:k]) if it in rel)
    idcg = sum(1 / math.log2(r + 2) for r in range(min(len(rel), k)))
    return dcg / idcg if idcg else 0

test_rel = defaultdict(set)
for u, i, _ in test_data:
    test_rel[u].add(i)

K_LIST   = [5, 10, 20]
agg      = {k: {'p': [], 'r': [], 'n': []} for k in K_LIST}
sample_u = list(test_rel.keys())[:300]

for u in sample_u:
    rel  = test_rel[u]
    recs = [i for i, _ in model.recommend(u, n=max(K_LIST), exclude=user_seen[u])]
    for k in K_LIST:
        agg[k]['p'].append(precision_at_k(recs, rel, k))
        agg[k]['r'].append(recall_at_k(recs, rel, k))
        agg[k]['n'].append(ndcg_at_k(recs, rel, k))

print(f"\n Ranking Metrics (300 users):")
print(f"  {'K':>4} | {'Precision@K':>12} | {'Recall@K':>10} | {'NDCG@K':>10}")
print("  " + "-" * 45)
for k in K_LIST:
    print(f"  {k:>4} | {np.mean(agg[k]['p']):>12.4f} | "
          f"{np.mean(agg[k]['r']):>10.4f} | {np.mean(agg[k]['n']):>10.4f}")

all_recs = set()
for u in sample_u:
    all_recs.update(i for i, _ in model.recommend(u, n=10, exclude=user_seen[u]))
coverage = len(all_recs) / n_i
print(f"\n Catalog Coverage@10  : {coverage:.4f}  ({len(all_recs)}/{n_i} items)")

gap = model.val_rmse[model.best_epoch - 1] - model.train_rmse[model.best_epoch - 1]
print(f"\n Training Summary:")
print(f"   Early stopped @ epoch : {model.stopped_epoch}")
print(f"   Best weights @ epoch : {model.best_epoch}")
print(f"   Best Val RMSE        : {min(model.val_rmse):.4f}")
print(f"   Train RMSE @ best ep : {model.train_rmse[model.best_epoch-1]:.4f}")
print(f"   Overfitting gap      : {gap:.4f}  {'OK' if gap < 0.3 else '(consider increasing reg)'}")

# ============================================================
# DEMO: Run model and print 3 recommendation results
# ============================================================
print("\n" + "=" * 70)
print("DEMO: RECOMMENDATION RESULTS (3 CUSTOMERS)")
print("=" * 70)

def product_info(i_idx):
    """Look up product by item index; return name, category, price."""
    pid = idx2item.get(i_idx)
    row = df_products[df_products["product_id"] == pid]
    if row.empty:
        return f"Product {pid}"
    r = row.iloc[0]
    return f"{r['product_name']:20s} | {r['category']:15s} | {r['price']:>10,.0f} VND"

# Pick 3 customers (by original customer_id) for demo
demo_customer_ids = list(user2idx.keys())[:3]
n_recommend = 5

for cid in demo_customer_ids:
    uidx = user2idx[cid]
    cust = df_customers[df_customers["customer_id"] == cid].iloc[0]
    # Show a few items they already purchased (from train)
    seen_names = []
    for i in list(user_seen[uidx])[:4]:
        pid = idx2item.get(i)
        row = df_products[df_products["product_id"] == pid]
        if not row.empty:
            seen_names.append(row.iloc[0]["product_name"])
    print(f"\n Customer #{cid}  |  {cust['customer_name']}")
    print(f"   Preferences : {cust['preferred_categories']}")
    print(f"   Budget      : {cust['budget_level']}  |  City: {cust['city']}")
    print(f"   Purchased   : {seen_names}")
    print(f"   Top-{n_recommend} recommended:")
    for rank, (ii, sc) in enumerate(
        model.recommend(uidx, n=n_recommend, exclude=user_seen[uidx]), 1
    ):
        print(f"      {rank}. {product_info(ii)}  |  Score: {sc:.3f}")

print("\n" + "=" * 70)
print("DONE - Product recommendation model trained")
print("=" * 70)
