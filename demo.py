"""
Demo script — loads the trained Content-Enhanced SVD++ model and:
  1) Runs Top-N recommendations for sample customers
  2) Evaluates regression metrics (RMSE / MAE / R²) on the held-out test set
  3) Evaluates ranking metrics (Precision@K / Recall@K / NDCG@K)

Run after train_model.py:
    python train_model.py    # produces result/best_model.pkl
    python demo.py           # loads it and runs the demo

Optional CLI:
    python demo.py --customer C0001 --topk 10
"""

import os
import math
import pickle
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "result", "best_model.pkl")
PRODUCTS_CSV  = os.path.join(BASE_DIR, "datasets", "products.csv")
CUSTOMERS_CSV = os.path.join(BASE_DIR, "datasets", "customers.csv")


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            f"Run `python train_model.py` first to train and save the model."
        )
    with open(path, "rb") as f:
        state = pickle.load(f)
    # Convert user_seen lists back to sets for fast membership tests
    state["user_seen"] = {u: set(items) for u, items in state["user_seen"].items()}
    return state


# ------------------------------------------------------------
# Predict (self-contained — no class needed)
# ------------------------------------------------------------
class Recommender:
    """Lightweight inference wrapper around the saved model state."""

    def __init__(self, state):
        self.U = state["U"]
        self.V = state["V"]
        self.Y = state["Y"]
        self.W_cat = state["W_cat"]
        self.b_u = state["b_u"]
        self.b_i = state["b_i"]
        self.b_g = state["b_g"]
        self.n_factors = state["n_factors"]
        self.user2idx = state["user2idx"]
        self.item2idx = state["item2idx"]
        self.idx2item = state["idx2item"]
        self.user_seen = state["user_seen"]
        self.user_cat_pref = state["user_cat_pref"]
        self.s_min = state["s_min"]
        self.s_max = state["s_max"]
        self.n_items = self.V.shape[0]

    def _implicit_sum(self, u):
        seen = self.user_seen.get(u, set())
        if not seen:
            return np.zeros(self.n_factors)
        idx = np.fromiter(seen, dtype=np.int64, count=len(seen))
        return self.Y[idx].sum(axis=0) / math.sqrt(len(idx))

    def _user_effective_vec(self, u):
        content = self.user_cat_pref[u] @ self.W_cat
        return self.U[u] + self._implicit_sum(u) + content

    def predict(self, u, i):
        pu_eff = self._user_effective_vec(u)
        return float(self.b_g + self.b_u[u] + self.b_i[i] + pu_eff @ self.V[i])

    def predict_all_items(self, u):
        """Vectorized score for user u over all items."""
        pu_eff = self._user_effective_vec(u)
        return self.b_g + self.b_u[u] + self.b_i + self.V @ pu_eff

    def recommend(self, u, n=10, exclude_seen=True):
        scores = self.predict_all_items(u)
        if exclude_seen:
            for i in self.user_seen.get(u, set()):
                scores[i] = -np.inf
        top = np.argpartition(-scores, min(n, len(scores) - 1))[:n]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top]

    def denormalize(self, score_norm):
        """Convert normalized [0,1] score back to original scale."""
        return float(np.clip(score_norm, 0.0, 1.0)) * (self.s_max - self.s_min) + self.s_min


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------
def regression_metrics(rec, test_data):
    if not test_data:
        return None
    pairs = [(u, i) for u, i, _, _ in test_data]
    actual_n = np.array([r for _, _, r, _ in test_data])
    pred_n = np.array([rec.predict(u, i) for u, i in pairs])

    actual = actual_n * (rec.s_max - rec.s_min) + rec.s_min
    pred = np.clip(pred_n, 0, 1) * (rec.s_max - rec.s_min) + rec.s_min

    mse = float(np.mean((actual - pred) ** 2))
    rmse = math.sqrt(mse)
    mae = float(np.mean(np.abs(actual - pred)))
    ss_res = float(np.sum((actual - pred) ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def ranking_metrics(rec, test_data, k_list=(5, 10, 20), max_users=300):
    test_rel = defaultdict(set)
    for u, i, _, _ in test_data:
        test_rel[u].add(i)
    sample_users = list(test_rel.keys())[:max_users]

    agg = {k: {"p": [], "r": [], "n": []} for k in k_list}
    max_k = max(k_list)
    seen_items_global = set()

    for u in sample_users:
        rel = test_rel[u]
        recs = [i for i, _ in rec.recommend(u, n=max_k)]
        seen_items_global.update(recs[:10])
        for k in k_list:
            hit = set(recs[:k]) & rel
            agg[k]["p"].append(len(hit) / k)
            agg[k]["r"].append(len(hit) / len(rel) if rel else 0.0)
            dcg = sum(1 / math.log2(r + 2) for r, it in enumerate(recs[:k]) if it in rel)
            idcg = sum(1 / math.log2(r + 2) for r in range(min(len(rel), k)))
            agg[k]["n"].append(dcg / idcg if idcg else 0.0)

    coverage = len(seen_items_global) / rec.n_items
    return {
        "per_k": {
            k: {
                "precision": float(np.mean(agg[k]["p"])),
                "recall": float(np.mean(agg[k]["r"])),
                "ndcg": float(np.mean(agg[k]["n"])),
            }
            for k in k_list
        },
        "coverage_at_10": coverage,
        "n_users": len(sample_users),
    }


# ------------------------------------------------------------
# Pretty printing
# ------------------------------------------------------------
def show_recommendations(rec, df_products, df_customers, customer_ids, topk=5):
    print("\n" + "=" * 70)
    print(f"TOP-{topk} RECOMMENDATIONS")
    print("=" * 70)

    for cid in customer_ids:
        if cid not in rec.user2idx:
            print(f"\n  [skip] customer_id {cid} not in trained model")
            continue
        uidx = rec.user2idx[cid]
        cust_row = df_customers[df_customers["customer_id"] == cid]
        cust = cust_row.iloc[0] if not cust_row.empty else None

        print(f"\n Customer #{cid}" + (f"  |  {cust['customer_name']}" if cust is not None else ""))
        if cust is not None:
            print(f"   Preferences : {cust['preferred_categories']}")
            print(f"   Budget      : {cust['budget_level']}  |  City: {cust['city']}")

        # show a few items already purchased
        seen_idxs = list(rec.user_seen.get(uidx, set()))[:4]
        seen_names = []
        for i in seen_idxs:
            pid = rec.idx2item.get(i)
            row = df_products[df_products["product_id"] == pid]
            if not row.empty:
                seen_names.append(row.iloc[0]["product_name"])
        print(f"   Purchased   : {seen_names}")

        print(f"   Top-{topk} recommended:")
        for rank, (ii, score_n) in enumerate(rec.recommend(uidx, n=topk), 1):
            pid = rec.idx2item.get(ii)
            row = df_products[df_products["product_id"] == pid]
            if row.empty:
                line = f"product {pid}"
            else:
                p = row.iloc[0]
                line = f"{p['product_name']:20s} | {p['category']:15s} | {p['price']:>10,.0f} VND"
            print(f"      {rank}. {line}  |  norm_score={score_n:.3f}  est={rec.denormalize(score_n):.2f}")


def print_metrics(reg, rank):
    if reg is not None:
        print("\n" + "=" * 70)
        print("REGRESSION METRICS (test set, original scale)")
        print("=" * 70)
        print(f"  MSE  : {reg['mse']:.4f}")
        print(f"  RMSE : {reg['rmse']:.4f}")
        print(f"  MAE  : {reg['mae']:.4f}")
        print(f"  R2   : {reg['r2']:.4f}")

    if rank is not None:
        print("\n" + "=" * 70)
        print(f"RANKING METRICS  ({rank['n_users']} users)")
        print("=" * 70)
        print(f"  {'K':>4} | {'Precision@K':>12} | {'Recall@K':>10} | {'NDCG@K':>10}")
        print("  " + "-" * 45)
        for k, m in rank["per_k"].items():
            print(f"  {k:>4} | {m['precision']:>12.4f} | {m['recall']:>10.4f} | {m['ndcg']:>10.4f}")
        print(f"\n  Catalog Coverage@10 : {rank['coverage_at_10']:.4f}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Demo for Content-Enhanced SVD++ recommender")
    parser.add_argument("--customer", type=str, default=None,
                        help="Specific customer_id to recommend for (default: first 3)")
    parser.add_argument("--topk", type=int, default=5, help="Number of recommendations to show")
    parser.add_argument("--no-eval", action="store_true", help="Skip metric evaluation")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to saved model pickle")
    args = parser.parse_args()

    print("=" * 70)
    print("LOAD MODEL")
    print("=" * 70)
    state = load_model(args.model)
    rec = Recommender(state)
    print(f"  Loaded     : {args.model}")
    print(f"  Best epoch : {state.get('best_epoch')}")
    print(f"  Val RMSE   : {state.get('best_val_rmse'):.4f}")
    print(f"  Users      : {len(rec.user2idx)} | Items: {rec.n_items} | Factors: {rec.n_factors}")

    df_products  = pd.read_csv(PRODUCTS_CSV)
    df_customers = pd.read_csv(CUSTOMERS_CSV)

    if args.customer:
        customer_ids = [args.customer]
    else:
        customer_ids = list(rec.user2idx.keys())[:3]

    show_recommendations(rec, df_products, df_customers, customer_ids, topk=args.topk)

    if not args.no_eval:
        test_data = state.get("test_data") or []
        reg = regression_metrics(rec, test_data)
        rank = ranking_metrics(rec, test_data)
        print_metrics(reg, rank)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
