"""
Product recommendation system — Content-Enhanced SVD++
Features:
  [1] Early Stopping       — stop when validation stops improving
  [2] Adam Optimizer       — stable adaptive learning updates
  [3] Multi-negative BPR   — optimize ranking with stronger signal
  [4] Popularity Penalty   — down-weight popular items to improve coverage
  [5] Score Normalization  — normalize scores to [0,1]
  [6] Grid Search          — find best hyperparameters on validation set
  [7] Content Features     — user preferred categories + product categories
  [8] SVD++ Implicit Feedback — exploit user history with implicit vectors
  [9] Visualization        — training curves, grid search, test metrics (result/*.png)
"""

import numpy as np
import pandas as pd
import random
import math
import os
from collections import defaultdict

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    plt = None
    HAS_MPL = False

random.seed(42)
np.random.seed(42)

# ============================================================
# RUNTIME BACKEND (GPU/CPU)
# Set USE_GPU=1 to enable GPU when CuPy is available
# ============================================================
USE_GPU = os.getenv("USE_GPU", "0") == "1"
GPU_AVAILABLE = cp is not None and cp.cuda.runtime.getDeviceCount() > 0 if cp is not None else False
USE_GPU = USE_GPU and GPU_AVAILABLE

# Figures saved here when matplotlib is available (set SAVE_FIGURES=0 to skip)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
SAVE_FIGURES = os.getenv("SAVE_FIGURES", "1") == "1"

if USE_GPU:
    print("[GPU] Running with CuPy backend")
else:
    reason = "USE_GPU not set"
    if os.getenv("USE_GPU", "0") == "1" and not GPU_AVAILABLE:
        reason = "CuPy/GPU unavailable, fallback to CPU"
    print(f"[CPU] Running with NumPy backend ({reason})")

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
# DATA ENRICHMENT (on-the-fly, no raw CSV edits)
# ============================================================
pc = np.log1p(interaction["purchase_count"])
oc = np.log1p(interaction["order_count"])
pc = (pc - pc.min()) / (pc.max() - pc.min() + 1e-9)
oc = (oc - oc.min()) / (oc.max() - oc.min() + 1e-9)
interaction["confidence"] = 0.65 * pc + 0.35 * oc

# ============================================================
# [5] SCORE NORMALIZATION TO [0, 1]
# Keeps SGD stable when targets lie in a bounded range
# ============================================================
# Robust clipping to reduce outlier effect on regression loss
s_low = interaction['score'].quantile(0.01)
s_high = interaction['score'].quantile(0.99)
interaction['score_clip'] = interaction['score'].clip(s_low, s_high)
s_min = interaction['score_clip'].min()
s_max = interaction['score_clip'].max()
interaction['score_norm'] = (interaction['score_clip'] - s_min) / (s_max - s_min + 1e-9)

print(f"\n[5] Robust score normalization: [{s_min:.3f}, {s_max:.3f}] -> [0.000, 1.000]")

# ============================================================
# INDEX MAPPING
# ============================================================
all_users = sorted(interaction['customer_id'].unique())
all_items = sorted(interaction['product_id'].unique())
user2idx  = {u: i for i, u in enumerate(all_users)}
item2idx  = {p: i for i, p in enumerate(all_items)}
idx2item  = {i: p for p, i in item2idx.items()}
n_u, n_i  = len(all_users), len(all_items)

# ============================================================
# [7] CONTENT FEATURES (customer preferences + item categories)
# ============================================================
all_categories = sorted(df_products["category"].unique())
cat2idx = {c: idx for idx, c in enumerate(all_categories)}
n_cat = len(all_categories)

product_category_idx = np.zeros(n_i, dtype=np.int32)
item_cat = np.zeros((n_i, n_cat), dtype=np.float64)
for row in df_products.itertuples():
    if row.product_id not in item2idx:
        continue
    iidx = item2idx[row.product_id]
    cidx = cat2idx[row.category]
    product_category_idx[iidx] = cidx
    item_cat[iidx, cidx] = 1.0

user_cat_pref = np.zeros((n_u, n_cat), dtype=np.float64)
for row in df_customers.itertuples():
    if row.customer_id not in user2idx:
        continue
    uidx = user2idx[row.customer_id]
    prefs = str(row.preferred_categories).split("|") if pd.notna(row.preferred_categories) else []
    for cat in prefs:
        cat = cat.strip()
        if cat in cat2idx:
            user_cat_pref[uidx, cat2idx[cat]] = 1.0
    row_sum = user_cat_pref[uidx].sum()
    if row_sum > 0:
        user_cat_pref[uidx] /= row_sum

print(f"[7] Content features: {n_cat} categories encoded")

records = [
    (user2idx[r.customer_id], item2idx[r.product_id], float(r.score_norm), float(r.confidence))
    for r in interaction.itertuples()
]

def user_stratified_split(data, train_ratio=0.70, val_ratio=0.15):
    grouped = defaultdict(list)
    for rec in data:
        grouped[rec[0]].append(rec)
    train, val, test = [], [], []
    for rows in grouped.values():
        random.shuffle(rows)
        m = len(rows)
        if m < 3:
            train.extend(rows)
            continue
        n_val_u = max(1, int(round(val_ratio * m)))
        n_test_u = max(1, int(round((1.0 - train_ratio - val_ratio) * m)))
        n_train_u = m - n_val_u - n_test_u
        if n_train_u < 1:
            n_train_u = 1
            if n_val_u > 1:
                n_val_u -= 1
            elif n_test_u > 1:
                n_test_u -= 1
        train.extend(rows[:n_train_u])
        val.extend(rows[n_train_u:n_train_u + n_val_u])
        test.extend(rows[n_train_u + n_val_u:])
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test

train_data, val_data, test_data = user_stratified_split(records, train_ratio=0.70, val_ratio=0.15)
print(f"[split] user-stratified: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

user_seen = defaultdict(set)
for u, i, _, _ in train_data:
    user_seen[u].add(i)

# ============================================================
# [4] POPULARITY PENALTY WEIGHTS
# Popular items get lower weight -> less dominance -> better coverage
# ============================================================
item_count = defaultdict(int)
for u, i, _, _ in train_data:
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
class ContentSVDpp:
    """
    Content-Enhanced SVD++ with:
    - Early Stopping [1]
    - Adam optimizer [2]
    - Multi-negative BPR ranking [3]
    - Popularity-weighted loss [4]
    - Score normalization [5]
    - Category-aware user vectors [7]
    - SVD++ implicit feedback [8]
    """

    def __init__(self, n_users, n_items, n_factors=30,
                 lr=0.001, reg=0.015, n_epochs=120,
                 patience=20, item_weights=None,
                 user_cat_pref=None, product_category_idx=None,
                 beta1=0.9, beta2=0.999, eps=1e-8,
                 use_gpu=False,
                 verbose=True):
        self.n_users      = n_users
        self.n_items      = n_items
        self.n_factors    = n_factors
        self.lr           = lr
        self.reg          = reg
        self.n_epochs     = n_epochs
        self.patience     = patience
        self.item_weights = item_weights
        self.user_cat_pref = user_cat_pref
        self.product_category_idx = product_category_idx
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.use_gpu = bool(use_gpu and cp is not None)
        self.xp = cp if self.use_gpu else np
        self.verbose      = verbose

        # Parameters: explicit user/item, implicit item, category embeddings
        scale = math.sqrt(2.0 / n_factors)
        self.U    = self.xp.random.normal(0, scale, (n_users, n_factors))   # p_u
        self.V    = self.xp.random.normal(0, scale, (n_items, n_factors))   # q_i
        self.Y    = self.xp.random.normal(0, scale, (n_items, n_factors))   # y_j for SVD++
        n_cat_local = user_cat_pref.shape[1] if user_cat_pref is not None else 1
        self.W_cat = self.xp.random.normal(0, scale, (n_cat_local, n_factors))
        self.b_u  = self.xp.zeros(n_users)
        self.b_i  = self.xp.zeros(n_items)
        self.b_g  = 0.0

        # Adam states
        self.t = 0
        self.m_U = self.xp.zeros_like(self.U); self.v_U = self.xp.zeros_like(self.U)
        self.m_V = self.xp.zeros_like(self.V); self.v_V = self.xp.zeros_like(self.V)
        self.m_Y = self.xp.zeros_like(self.Y); self.v_Y = self.xp.zeros_like(self.Y)
        self.m_W = self.xp.zeros_like(self.W_cat); self.v_W = self.xp.zeros_like(self.W_cat)
        self.m_bu = self.xp.zeros_like(self.b_u); self.v_bu = self.xp.zeros_like(self.b_u)
        self.m_bi = self.xp.zeros_like(self.b_i); self.v_bi = self.xp.zeros_like(self.b_i)
        self.m_bg = 0.0; self.v_bg = 0.0

        self.train_losses, self.val_losses   = [], []
        self.train_rmse,   self.val_rmse     = [], []
        self.best_U = self.best_V = self.best_Y = self.best_W_cat = None
        self.best_b_u = self.best_b_i = None
        self.best_val_loss = float('inf')
        self.best_epoch    = 0
        self.stopped_epoch = n_epochs

    def _implicit_sum(self, u, user_seen):
        seen = user_seen.get(u, set())
        if not seen:
            return self.xp.zeros(self.n_factors)
        idx = self.xp.array(list(seen), dtype=self.xp.int32)
        return self.Y[idx].sum(axis=0) / math.sqrt(len(idx))

    def _user_content_vec(self, u):
        if self.user_cat_pref is None:
            return self.xp.zeros(self.n_factors)
        return self.user_cat_pref[u] @ self.W_cat

    def _user_effective_vec(self, u, user_seen):
        return self.U[u] + self._implicit_sum(u, user_seen) + self._user_content_vec(u)

    def _pred(self, u, i, user_seen):
        pu_eff = self._user_effective_vec(u, user_seen)
        score = self.b_g + self.b_u[u] + self.b_i[i] + self.xp.dot(pu_eff, self.V[i])
        return float(score)

    def predict_batch(self, pairs, user_seen):
        return np.array([self._pred(u, i, user_seen) for u, i in pairs])

    def _save_best(self, epoch):
        self.best_U   = self.U.copy()
        self.best_V   = self.V.copy()
        self.best_Y   = self.Y.copy()
        self.best_W_cat = self.W_cat.copy()
        self.best_b_u = self.b_u.copy()
        self.best_b_i = self.b_i.copy()
        self.best_b_g = self.b_g
        self.best_epoch = epoch

    def _restore_best(self):
        self.U   = self.best_U
        self.V   = self.best_V
        self.Y   = self.best_Y
        self.W_cat = self.best_W_cat
        self.b_u = self.best_b_u
        self.b_i = self.best_b_i
        self.b_g = self.best_b_g

    def _adam_update_mat(self, param, grad, m, v):
        self.t += 1
        m[:] = self.beta1 * m + (1.0 - self.beta1) * grad
        v[:] = self.beta2 * v + (1.0 - self.beta2) * (grad * grad)
        m_hat = m / (1.0 - self.beta1 ** self.t)
        v_hat = v / (1.0 - self.beta2 ** self.t)
        param[:] += self.lr * m_hat / (self.xp.sqrt(v_hat) + self.eps)

    def _adam_update_scalar(self, param_array, idx, grad, m_array, v_array):
        self.t += 1
        m_array[idx] = self.beta1 * m_array[idx] + (1.0 - self.beta1) * grad
        v_array[idx] = self.beta2 * v_array[idx] + (1.0 - self.beta2) * (grad * grad)
        m_hat = m_array[idx] / (1.0 - self.beta1 ** self.t)
        v_hat = v_array[idx] / (1.0 - self.beta2 ** self.t)
        param_array[idx] += self.lr * m_hat / (math.sqrt(v_hat) + self.eps)

    def _adam_update_global_bias(self, grad):
        self.t += 1
        self.m_bg = self.beta1 * self.m_bg + (1.0 - self.beta1) * grad
        self.v_bg = self.beta2 * self.v_bg + (1.0 - self.beta2) * (grad * grad)
        m_hat = self.m_bg / (1.0 - self.beta1 ** self.t)
        v_hat = self.v_bg / (1.0 - self.beta2 ** self.t)
        self.b_g += self.lr * m_hat / (math.sqrt(v_hat) + self.eps)

    def fit(self, train_data, val_data, user_seen, bpr_weight=0.3, n_neg=5):
        """
        bpr_weight: fraction of BPR gradients blended into regression update.
        n_neg: number of negatives sampled for each positive pair.
        """
        self.b_g = float(np.mean([r for _, _, r, _ in train_data]))

        no_improve = 0
        t_pairs = [(u, i) for u, i, _, _ in train_data]
        v_pairs = [(u, i) for u, i, _, _ in val_data]
        t_act   = np.array([r for _, _, r, _ in train_data])
        v_act   = np.array([r for _, _, r, _ in val_data])

        for epoch in range(self.n_epochs):
            random.shuffle(train_data)

            for u, i, r, conf in train_data:
                w_item = self.item_weights[i] if self.item_weights else 1.0
                w = w_item * (0.80 + 0.40 * conf)
                pred = self._pred(u, i, user_seen)
                err  = r - pred
                seen = user_seen.get(u, set())

                # Effective user vector and MSE gradients
                imp = self._implicit_sum(u, user_seen)
                content_vec = self._user_content_vec(u)
                pu_eff = self.U[u] + imp + content_vec
                du_mse = w * err * self.V[i] - self.reg * self.U[u]
                dv_mse = w * err * pu_eff - self.reg * self.V[i]
                dbu = w * err - self.reg * self.b_u[u]
                dbi = w * err - self.reg * self.b_i[i]
                dbg = w * err

                # [3] Multi-negative BPR gradients
                du_bpr = self.xp.zeros(self.n_factors)
                dv_pos = self.xp.zeros(self.n_factors)
                dy_updates = {}
                dw_bpr = self.xp.zeros_like(self.W_cat)
                neg_used = []
                if bpr_weight > 0:
                    pref_cats = set(np.where(self.user_cat_pref[u] > 0)[0]) if self.user_cat_pref is not None else set()
                    for _ in range(n_neg):
                        j = random.randint(0, self.n_items - 1)
                        attempts = 0
                        while (j in seen) and attempts < 20:
                            j = random.randint(0, self.n_items - 1)
                            attempts += 1
                        if j in seen:
                            continue
                        if pref_cats:
                            c_j = self.product_category_idx[j]
                            if c_j in pref_cats and random.random() < 0.5:
                                continue
                        x_uij = self._pred(u, i, user_seen) - self._pred(u, j, user_seen)
                        sig = 1.0 / (1.0 + math.exp(-x_uij))
                        grad = 1.0 - sig
                        du_bpr += grad * (self.V[i] - self.V[j]) - self.reg * self.U[u]
                        dv_pos += grad * pu_eff - self.reg * self.V[i]
                        dv_neg = -grad * pu_eff - self.reg * self.V[j]
                        self._adam_update_mat(self.V[j], dv_neg, self.m_V[j], self.v_V[j])
                        if seen:
                            coeff = grad / math.sqrt(len(seen))
                            for s_item in seen:
                                gy = coeff * (self.V[i] - self.V[j]) - self.reg * self.Y[s_item]
                                if s_item not in dy_updates:
                                    dy_updates[s_item] = gy
                                else:
                                    dy_updates[s_item] += gy
                        if self.user_cat_pref is not None:
                            pref = self.user_cat_pref[u][:, None]
                            dw_bpr += grad * (pref @ (self.V[i] - self.V[j])[None, :]) - self.reg * self.W_cat
                        neg_used.append(j)

                if neg_used:
                    inv = 1.0 / len(neg_used)
                    du_bpr *= inv
                    dv_pos *= inv
                    for s_item, gy in dy_updates.items():
                        dy_updates[s_item] = gy * inv
                    dw_bpr *= inv

                # MSE gradients for implicit Y and content W_cat
                dy_mse = {}
                if seen:
                    coeff = (w * err) / math.sqrt(len(seen))
                    for s_item in seen:
                        dy_mse[s_item] = coeff * self.V[i] - self.reg * self.Y[s_item]
                dw_mse = self.xp.zeros_like(self.W_cat)
                if self.user_cat_pref is not None:
                    pref = self.user_cat_pref[u][:, None]
                    dw_mse = (w * err) * (pref @ self.V[i][None, :]) - self.reg * self.W_cat

                a = 1.0 - bpr_weight
                b = bpr_weight

                # Adam updates
                self._adam_update_global_bias(dbg * 0.05)
                self._adam_update_scalar(self.b_u, u, dbu, self.m_bu, self.v_bu)
                self._adam_update_scalar(self.b_i, i, dbi, self.m_bi, self.v_bi)
                self._adam_update_mat(self.U[u], a * du_mse + b * du_bpr, self.m_U[u], self.v_U[u])
                self._adam_update_mat(self.V[i], a * dv_mse + b * dv_pos, self.m_V[i], self.v_V[i])
                for s_item, g in dy_mse.items():
                    g_all = a * g + b * dy_updates.get(s_item, 0.0)
                    self._adam_update_mat(self.Y[s_item], g_all, self.m_Y[s_item], self.v_Y[s_item])
                if self.user_cat_pref is not None:
                    self._adam_update_mat(self.W_cat, a * dw_mse + b * dw_bpr, self.m_W, self.v_W)

            # Metrics
            tl = float(np.mean((t_act - self.predict_batch(t_pairs, user_seen)) ** 2))
            vl = float(np.mean((v_act - self.predict_batch(v_pairs, user_seen)) ** 2))
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
        scores = [(i, self._pred(u, i, user_seen))
                  for i in range(self.n_items)
                  if not (exclude and i in exclude)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]


def _ensure_result_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def plot_training_curves(model, title_suffix=""):
    """Train/Val MSE and RMSE vs epoch; mark best epoch from early stopping."""
    if not HAS_MPL or not SAVE_FIGURES:
        return None
    _ensure_result_dir()
    n = len(model.train_rmse)
    if n == 0:
        return None
    epochs = np.arange(1, n + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, model.train_losses, label="Train MSE", color="#2563eb")
    axes[0].plot(epochs, model.val_losses, label="Val MSE", color="#dc2626")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE (normalized target)")
    axes[0].set_title("Loss during training" + title_suffix)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, model.train_rmse, label="Train RMSE", color="#2563eb")
    axes[1].plot(epochs, model.val_rmse, label="Val RMSE", color="#dc2626")
    if model.best_epoch and model.best_epoch <= n:
        axes[1].axvline(model.best_epoch, color="#64748b", linestyle="--", linewidth=1,
                        label=f"Best epoch {model.best_epoch}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("RMSE during training" + title_suffix)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_training_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_grid_search_results(rows):
    """Bar chart of Val RMSE for each grid configuration."""
    if not HAS_MPL or not SAVE_FIGURES or not rows:
        return None
    _ensure_result_dir()
    labels = [r["short"] for r in rows]
    vals = [r["val_rmse"] for r in rows]
    best_idx = int(np.argmin(vals))
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2563eb" if i != best_idx else "#16a34a" for i in range(len(labels))]
    ax.barh(range(len(labels)), vals, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Val RMSE (normalized [0,1])")
    ax.set_title("Grid search: validation RMSE by config")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_grid_search_val_rmse.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_test_regression(t_act, t_pred, rmse, r2, mae):
    """Actual vs predicted scatter and residual histogram."""
    if not HAS_MPL or not SAVE_FIGURES:
        return None
    _ensure_result_dir()
    err = t_act - t_pred
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(t_act, t_pred, alpha=0.35, s=12, c="#2563eb", edgecolors="none")
    lo = float(min(t_act.min(), t_pred.min()))
    hi = float(max(t_act.max(), t_pred.max()))
    axes[0].plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="y = x")
    axes[0].set_xlabel("Actual score (original scale)")
    axes[0].set_ylabel("Predicted score")
    axes[0].set_title(f"Test predictions  |  RMSE={rmse:.3f}, MAE={mae:.3f}, R²={r2:.3f}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(err, bins=30, color="#64748b", edgecolor="white", alpha=0.85)
    axes[1].axvline(0, color="#dc2626", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Residual (actual − predicted)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual distribution (test)")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_test_regression.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_ranking_metrics(K_LIST, agg, coverage):
    """Bar chart: Precision@K, Recall@K, NDCG@K."""
    if not HAS_MPL or not SAVE_FIGURES:
        return None
    _ensure_result_dir()
    ks = [str(k) for k in K_LIST]
    x = np.arange(len(K_LIST))
    w = 0.25
    prec = [float(np.mean(agg[k]["p"])) for k in K_LIST]
    rec = [float(np.mean(agg[k]["r"])) for k in K_LIST]
    ndcg = [float(np.mean(agg[k]["n"])) for k in K_LIST]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w, prec, w, label="Precision@K", color="#2563eb")
    ax.bar(x, rec, w, label="Recall@K", color="#16a34a")
    ax.bar(x + w, ndcg, w, label="NDCG@K", color="#ca8a04")
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in ks])
    ax.set_ylabel("Score")
    ax.set_title(f"Ranking metrics (test users)  |  Coverage@10 = {coverage:.3f}")
    ax.legend()
    ax.set_ylim(0, max(1.0, max(prec + rec + ndcg) * 1.1))
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_ranking_metrics.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ============================================================
# [6] GRID SEARCH — find best hyperparameters on validation set
# ============================================================
print("\n" + "=" * 70)
print("[6] GRID SEARCH HYPERPARAMETERS")
print("=" * 70)

param_grid = [
    {'n_factors': 24, 'lr': 0.0003, 'reg': 0.03, 'bpr': 0.05},
    {'n_factors': 24, 'lr': 0.0005, 'reg': 0.03, 'bpr': 0.10},
    {'n_factors': 30, 'lr': 0.0005, 'reg': 0.03, 'bpr': 0.10},
    {'n_factors': 30, 'lr': 0.0008, 'reg': 0.03, 'bpr': 0.15},
    {'n_factors': 30, 'lr': 0.0005, 'reg': 0.05, 'bpr': 0.10},
    {'n_factors': 36, 'lr': 0.0005, 'reg': 0.03, 'bpr': 0.15},
    {'n_factors': 36, 'lr': 0.0008, 'reg': 0.05, 'bpr': 0.20},
    {'n_factors': 30, 'lr': 0.0003, 'reg': 0.05, 'bpr': 0.05},
]

def eval_val_ranking(model, val_data, user_seen, k=10, max_users=250):
    val_rel = defaultdict(set)
    for u, i, _, _ in val_data:
        val_rel[u].add(i)
    users = list(val_rel.keys())[:max_users]
    if not users:
        return 0.0, 0.0
    p_list, n_list = [], []
    for u in users:
        recs = [i for i, _ in model.recommend(u, n=k, exclude=user_seen[u])]
        rel = val_rel[u]
        hit = len(set(recs[:k]) & rel)
        p_list.append(hit / k)
        dcg = sum(1 / math.log2(r + 2) for r, it in enumerate(recs[:k]) if it in rel)
        idcg = sum(1 / math.log2(r + 2) for r in range(min(len(rel), k)))
        n_list.append(dcg / idcg if idcg else 0.0)
    return float(np.mean(p_list)), float(np.mean(n_list))

print(f"\n  {'Config':>46} | {'Val RMSE':>8} | {'P@10':>6} | {'NDCG@10':>8} | {'Score':>7} | {'Ep':>4}")
print("  " + "-" * 98)

best_config = None
best_balance_score = float('inf')
grid_search_rows = []

for cfg in param_grid:
    np.random.seed(42); random.seed(42)
    m = ContentSVDpp(
        n_u, n_i,
        n_factors=cfg['n_factors'],
        lr=cfg['lr'], reg=cfg['reg'],
        n_epochs=200, patience=20,
        item_weights=item_weights,
        user_cat_pref=(cp.asarray(user_cat_pref) if USE_GPU else user_cat_pref),
        product_category_idx=product_category_idx,
        use_gpu=USE_GPU,
        verbose=False
    )
    m.fit(train_data, val_data, user_seen=user_seen, bpr_weight=cfg['bpr'], n_neg=3)
    vr = min(m.val_rmse)
    p10, ndcg10 = eval_val_ranking(m, val_data, user_seen=user_seen, k=10, max_users=250)
    balance_score = 0.65 * vr + 0.35 * (1.0 - ndcg10)
    label = f"factors={cfg['n_factors']}, lr={cfg['lr']}, reg={cfg['reg']}, bpr={cfg['bpr']}"
    marker = " < BEST" if balance_score < best_balance_score else ""
    print(f"  {label:>46} | {vr:>8.4f} | {p10:>6.4f} | {ndcg10:>8.4f} | {balance_score:>7.4f} | {m.best_epoch:>4}{marker}")
    grid_search_rows.append({
        "short": f"f{cfg['n_factors']} lr{cfg['lr']} r{cfg['reg']} bpr{cfg['bpr']}",
        "val_rmse": vr,
        "ndcg10": ndcg10,
        "balance": balance_score,
    })
    if balance_score < best_balance_score:
        best_balance_score = balance_score
        best_config = cfg

print(f"\n  Best config: {best_config}")

_grid_fig = plot_grid_search_results(grid_search_rows)
if _grid_fig:
    print(f"\n[vis] Grid search plot saved: {_grid_fig}")


# ============================================================
# TRAIN FINAL MODEL WITH BEST CONFIG
# ============================================================
print("\n" + "=" * 70)
print("TRAIN FINAL MODEL WITH BEST CONFIG")
print("=" * 70)
print(f"\n  n_factors={best_config['n_factors']}, lr={best_config['lr']}, reg={best_config['reg']}, bpr={best_config['bpr']}")
print(f"  + Early Stopping (patience=20)")
print(f"  + Adam Optimizer (lr={best_config['lr']}) [2]")
print(f"  + BPR Ranking Loss (weight={best_config['bpr']}) [3]")
print(f"  + Popularity Penalty weights [4]")
print(f"  + Score Normalization [0,1] [5]\n")

np.random.seed(42); random.seed(42)
model = ContentSVDpp(
    n_u, n_i,
    n_factors=best_config['n_factors'],
    lr=best_config['lr'],
    reg=best_config['reg'],
    n_epochs=200, patience=20,
    item_weights=item_weights,
    user_cat_pref=(cp.asarray(user_cat_pref) if USE_GPU else user_cat_pref),
    product_category_idx=product_category_idx,
    use_gpu=USE_GPU,
    verbose=True
)
model.fit(train_data, val_data, user_seen=user_seen, bpr_weight=best_config['bpr'], n_neg=3)

_train_fig = plot_training_curves(model, title_suffix=" (final model)")
if _train_fig:
    print(f"[vis] Training curves: {_train_fig}")

# ============================================================
# EVALUATION — denormalize to original scale for interpretability
# ============================================================
print("\n" + "=" * 70)
print("EVALUATE MODEL ON TEST SET")
print("=" * 70)

t_pairs   = [(u, i) for u, i, _, _ in test_data]
# Predictions and actuals are on normalized scale [0,1]
t_pred_n  = model.predict_batch(t_pairs, user_seen=user_seen)
t_act_n   = np.array([r for _, _, r, _ in test_data])

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

_reg_fig = plot_test_regression(t_act, t_pred, rmse, r2, mae)
if _reg_fig:
    print(f"[vis] Test regression: {_reg_fig}")

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
for u, i, _, _ in test_data:
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

_rank_fig = plot_ranking_metrics(K_LIST, agg, coverage)
if _rank_fig:
    print(f"[vis] Ranking metrics: {_rank_fig}")

gap = model.val_rmse[model.best_epoch - 1] - model.train_rmse[model.best_epoch - 1]
print(f"\n Training Summary:")
print(f"   Early stopped @ epoch : {model.stopped_epoch}")
print(f"   Best weights @ epoch : {model.best_epoch}")
print(f"   Best Val RMSE        : {min(model.val_rmse):.4f}")
print(f"   Train RMSE @ best ep : {model.train_rmse[model.best_epoch-1]:.4f}")
print(f"   Overfitting gap      : {gap:.4f}  {'OK' if gap < 0.3 else '(consider increasing reg)'}")

if not HAS_MPL:
    print("\n[vis] Install matplotlib to save training/metric figures: pip install matplotlib")
elif not SAVE_FIGURES:
    print("\n[vis] Figures skipped (SAVE_FIGURES=0)")

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
