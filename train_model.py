"""
Product recommendation system - load from CSV datasets
Run: python train_model.py
"""

import numpy as np
import pandas as pd
import random
import math
from collections import defaultdict

# ============================================================
# 1. LOAD DATASETS FROM CSV
# ============================================================
print("=" * 70)
print("STEP 1: LOAD DATASETS")
print("=" * 70)

df_products    = pd.read_csv('datasets/products.csv')
df_customers   = pd.read_csv('datasets/customers.csv')
df_orders      = pd.read_csv('datasets/orders.csv')
df_items       = pd.read_csv('datasets/order_items.csv')
interaction    = pd.read_csv('datasets/user_item_interactions.csv')

print(f"Loaded products.csv          : {len(df_products):>5} rows")
print(f"Loaded customers.csv         : {len(df_customers):>5} rows")
print(f"Loaded orders.csv            : {len(df_orders):>5} rows")
print(f"Loaded order_items.csv       : {len(df_items):>5} rows")
print(f"Loaded user_item_interactions: {len(interaction):>5} rows")

NUM_PRODUCTS  = len(df_products)
NUM_CUSTOMERS = len(df_customers)
sparsity = 1 - len(interaction) / (NUM_CUSTOMERS * NUM_PRODUCTS)

print(f"\nDataset statistics:")
print(f"   - Products          : {NUM_PRODUCTS}")
print(f"   - Customers         : {NUM_CUSTOMERS}")
print(f"   - Orders            : {len(df_orders)}")
print(f"   - Order items       : {len(df_items)}")
print(f"   - (user,item) pairs : {len(interaction)}")
print(f"   - Sparsity          : {sparsity:.4f} ({sparsity*100:.1f}%)")
print(f"   - Score range       : [{interaction['score'].min():.2f}, {interaction['score'].max():.2f}]")

# ============================================================
# 2. BUILD MODEL - Matrix Factorization (SGD from scratch)
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: BUILD MODEL - MATRIX FACTORIZATION (from scratch)")
print("=" * 70)

class MatrixFactorizationSGD:
    """
    Collaborative Filtering using Matrix Factorization + SGD.
    No external ML library is used.
    Objective: min ||R - (U @ V^T + b_u + b_i + b_global)||^2 + λ(||U||²+||V||²)
    """

    def __init__(self, n_users, n_items, n_factors=20,
                 lr=0.005, reg=0.02, n_epochs=30, verbose=True):
        self.n_users   = n_users
        self.n_items   = n_items
        self.n_factors = n_factors
        self.lr        = lr
        self.reg       = reg
        self.n_epochs  = n_epochs
        self.verbose   = verbose

        # Initialize weights randomly
        np.random.seed(42)
        self.U       = np.random.normal(0, 0.1, (n_users, n_factors))
        self.V       = np.random.normal(0, 0.1, (n_items, n_factors))
        self.b_u     = np.zeros(n_users)
        self.b_i     = np.zeros(n_items)
        self.b_global = 0.0

        self.train_losses, self.val_losses   = [], []
        self.train_rmse,   self.val_rmse     = [], []

    def _predict(self, u, i):
        return self.b_global + self.b_u[u] + self.b_i[i] + np.dot(self.U[u], self.V[i])

    def predict_batch(self, pairs):
        return np.array([self._predict(u, i) for u, i in pairs])

    def fit(self, train_data, val_data):
        self.b_global = float(np.mean([r for *_, r in train_data]))

        print(f"\n🔧 Model configuration:")
        print(f"   n_factors  : {self.n_factors}   (latent dimensions)")
        print(f"   lr         : {self.lr}   (learning rate)")
        print(f"   reg λ      : {self.reg}   (L2 regularization)")
        print(f"   n_epochs   : {self.n_epochs}")
        print(f"   Train/Val  : {len(train_data)} / {len(val_data)} interactions")

        header = f"\n{'Epoch':>6} | {'Train Loss':>11} | {'Val Loss':>10} | {'Train RMSE':>11} | {'Val RMSE':>10}"
        print(header)
        print("-" * 60)

        for epoch in range(self.n_epochs):
            random.shuffle(train_data)

            for u, i, r in train_data:
                pred = self._predict(u, i)
                err  = r - pred

                # Gradient descent
                self.b_global += self.lr * err * 0.05
                self.b_u[u]   += self.lr * (err - self.reg * self.b_u[u])
                self.b_i[i]   += self.lr * (err - self.reg * self.b_i[i])
                Uu = self.U[u].copy()
                self.U[u] += self.lr * (err * self.V[i] - self.reg * self.U[u])
                self.V[i] += self.lr * (err * Uu        - self.reg * self.V[i])

            # Compute metrics
            t_pairs = [(u,i) for u,i,_ in train_data]
            v_pairs = [(u,i) for u,i,_ in val_data]
            t_act   = np.array([r for _,_,r in train_data])
            v_act   = np.array([r for _,_,r in val_data])

            t_pred  = self.predict_batch(t_pairs)
            v_pred  = self.predict_batch(v_pairs)

            tl = float(np.mean((t_act - t_pred)**2))
            vl = float(np.mean((v_act - v_pred)**2))
            tr = math.sqrt(tl)
            vr = math.sqrt(vl)

            self.train_losses.append(tl); self.val_losses.append(vl)
            self.train_rmse.append(tr);   self.val_rmse.append(vr)

            if self.verbose and (epoch+1) % 5 == 0:
                print(f"{epoch+1:>6} | {tl:>11.4f} | {vl:>10.4f} | {tr:>11.4f} | {vr:>10.4f}")

        return self

    def recommend(self, u, n=10, exclude=None):
        scores = []
        for i in range(self.n_items):
            if exclude and i in exclude:
                continue
            scores.append((i, self._predict(u, i)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]


# ============================================================
# 3. PREPARE DATA & SPLIT
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: PREPARE & SPLIT DATA")
print("=" * 70)

all_users = sorted(interaction['customer_id'].unique())
all_items = sorted(interaction['product_id'].unique())
user2idx  = {u: i for i, u in enumerate(all_users)}
item2idx  = {p: i for i, p in enumerate(all_items)}
idx2item  = {i: p for p, i in item2idx.items()}

n_u = len(all_users)
n_i = len(all_items)
print(f"   Active users: {n_u} | Active items: {n_i}")

records = [
    (user2idx[r.customer_id], item2idx[r.product_id], r.score)
    for r in interaction.itertuples()
]

random.seed(42)
random.shuffle(records)
n       = len(records)
n_train = int(0.70 * n)
n_val   = int(0.15 * n)
train_data = records[:n_train]
val_data   = records[n_train:n_train+n_val]
test_data  = records[n_train+n_val:]
print(f"   Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

user_seen = defaultdict(set)
for u, i, _ in train_data:
    user_seen[u].add(i)

# ============================================================
# 4. TRAIN
# ============================================================
model = MatrixFactorizationSGD(
    n_users=n_u, n_items=n_i,
    n_factors=20, lr=0.005, reg=0.02, n_epochs=30
)
model.fit(train_data, val_data)

# ============================================================
# 5. EVALUATION
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: EVALUATE MODEL")
print("=" * 70)

# Regression metrics
t_pairs  = [(u,i) for u,i,_ in test_data]
t_act    = np.array([r for _,_,r in test_data])
t_pred   = model.predict_batch(t_pairs)
mse      = float(np.mean((t_act - t_pred)**2))
rmse     = math.sqrt(mse)
mae      = float(np.mean(np.abs(t_act - t_pred)))
ss_res   = float(np.sum((t_act - t_pred)**2))
ss_tot   = float(np.sum((t_act - t_act.mean())**2))
r2       = 1 - ss_res/ss_tot if ss_tot > 0 else 0

print(f"\n Regression Metrics (Test Set):")
print(f"   {'MSE':<10}: {mse:.4f}")
print(f"   {'RMSE':<10}: {rmse:.4f}")
print(f"   {'MAE':<10}: {mae:.4f}")
print(f"   {'R²':<10}: {r2:.4f}")

# Ranking metrics
def precision_at_k(rec, rel, k):
    return len(set(rec[:k]) & rel) / k

def recall_at_k(rec, rel, k):
    return len(set(rec[:k]) & rel) / len(rel) if rel else 0

def ndcg_at_k(rec, rel, k):
    dcg  = sum(1/math.log2(r+2) for r, it in enumerate(rec[:k]) if it in rel)
    idcg = sum(1/math.log2(r+2) for r in range(min(len(rel), k)))
    return dcg/idcg if idcg else 0

test_rel = defaultdict(set)
for u, i, _ in test_data:
    test_rel[u].add(i)

K_LIST = [5, 10, 20]
agg    = {k: {'p':[], 'r':[], 'n':[]} for k in K_LIST}
sample_users = list(test_rel.keys())[:300]

for u in sample_users:
    rel  = test_rel[u]
    recs = [i for i,_ in model.recommend(u, n=max(K_LIST), exclude=user_seen[u])]
    for k in K_LIST:
        agg[k]['p'].append(precision_at_k(recs, rel, k))
        agg[k]['r'].append(recall_at_k(recs, rel, k))
        agg[k]['n'].append(ndcg_at_k(recs, rel, k))

print(f"\n Ranking Metrics ({len(sample_users)} users):")
print(f"{'K':>4} | {'Precision@K':>12} | {'Recall@K':>10} | {'NDCG@K':>10}")
print("-" * 45)
for k in K_LIST:
    print(f"{k:>4} | {np.mean(agg[k]['p']):>12.4f} | {np.mean(agg[k]['r']):>10.4f} | {np.mean(agg[k]['n']):>10.4f}")

# Coverage
all_recs = set()
for u in sample_users:
    all_recs.update(i for i,_ in model.recommend(u, n=10, exclude=user_seen[u]))
coverage = len(all_recs) / n_i
print(f"\n Catalog Coverage@10  : {coverage:.4f}  ({len(all_recs)}/{n_i} items)")

# Training summary
best_ep = int(np.argmin(model.val_losses)) + 1
gap     = model.val_rmse[-1] - model.train_rmse[-1]
print(f"\n Training Summary:")
print(f"   Best Val Loss  @ epoch : {best_ep}")
print(f"   Best Val RMSE          : {min(model.val_rmse):.4f}")
print(f"   Final Train RMSE       : {model.train_rmse[-1]:.4f}")
print(f"   Final Val   RMSE       : {model.val_rmse[-1]:.4f}")
print(f"   Overfitting gap        : {gap:.4f}  {' OK' if gap < 0.3 else '  Increase reg'}")

# ============================================================
# 6. RECOMMENDATION DEMO
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: PRODUCT RECOMMENDATION DEMO")
print("=" * 70)

def product_info(i_idx):
    pid = idx2item.get(i_idx)
    row = df_products[df_products['product_id'] == pid]
    if row.empty:
        return f"Product {pid}"
    r = row.iloc[0]
    return f"{r['product_name']:20s} | {r['category']:15s} | {r['price']:>10,.0f}đ"

demo = list(user2idx.keys())[:4]
for cid in demo:
    uidx = user2idx[cid]
    cust = df_customers[df_customers['customer_id'] == cid].iloc[0]
    seen_names = [
        df_products[df_products['product_id']==idx2item[i]].iloc[0]['product_name']
        for i in list(user_seen[uidx])[:4]
        if not df_products[df_products['product_id']==idx2item[i]].empty
    ]
    print(f"\n👤 Customer #{cid}")
    print(f"   Preferences : {cust['preferred_categories']}")
    print(f"   Budget      : {cust['budget_level']}  |  City: {cust['city']}")
    print(f"   Purchased   : {seen_names}")
    print(f"   🎯 Recommended Top-5:")
    for rank, (ii, sc) in enumerate(model.recommend(uidx, n=5, exclude=user_seen[uidx]), 1):
        print(f"      {rank}. {product_info(ii)} | Score: {sc:.3f}")

print("\n" + "=" * 70)
print("DONE - PRODUCT RECOMMENDATION MODEL")
print("=" * 70)