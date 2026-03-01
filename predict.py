#!/usr/bin/env python3
"""
Inference: given user_id -> get candidates from model_cf -> rank with model_rank -> return top-N by P(purchase).
Usage:
  python predict.py --user-id U001 [--top-n 10] [--model-dir models] [--data-dir datasets]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader import load_data
from src.stage1_candidate import load_model_cf, get_candidates
from src.stage2_ranking import build_features, load_model_rank, get_feature_columns


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend products for a user")
    parser.add_argument("--user-id", type=str, required=True, help="User ID")
    parser.add_argument("--top-n", type=int, default=10, help="Number of recommendations")
    parser.add_argument("--model-dir", type=str, default="models", help="Path to models")
    parser.add_argument("--data-dir", type=str, default="datasets", help="Path to datasets")
    parser.add_argument("--k-candidates", type=int, default=100, help="Max candidates from CF")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)
    if not model_dir.is_dir():
        print(f"Error: model directory not found: {model_dir}. Run train_pipeline.py first.")
        sys.exit(1)
    if not data_dir.is_dir():
        print(f"Error: data directory not found: {data_dir}")
        sys.exit(1)

    # Load data and models
    data = load_data(data_dir)
    model_cf, user_id_to_idx, _, product_id_to_idx, idx_to_product_id = load_model_cf(
        model_dir / "model_cf.joblib"
    )
    model_rank, label_encoders, feature_cols = load_model_rank(
        model_dir / "model_rank.joblib"
    )
    implicit_matrix = data["implicit_matrix"]
    users = data["users"]
    products = data["products"]
    orders = data["orders"]
    interactions = data["interactions"]

    active_ids = set(products[products["is_active"] == 1]["product_id"].astype(str)) if "is_active" in products.columns else set(products["product_id"].astype(str))
    bought = set()
    for _, row in orders.iterrows():
        if str(row["user_id"]) == args.user_id:
            bought.add(str(row["product_id"]))

    # 1. Candidates from CF
    candidates = get_candidates(
        args.user_id,
        model_cf,
        implicit_matrix,
        user_id_to_idx,
        idx_to_product_id,
        K=args.k_candidates,
        filter_already_bought=True,
        bought_product_ids=bought,
        filter_inactive=True,
        active_product_ids=active_ids,
    )
    if not candidates:
        print(f"No candidates for user {args.user_id} (cold start or no data).")
        sys.exit(0)

    # 2. Build features for (user_id, product_id) pairs
    pairs = [{"user_id": args.user_id, "product_id": pid} for pid, _ in candidates]
    import pandas as pd
    pairs_df = pd.DataFrame(pairs)
    X, _ = build_features(
        pairs_df,
        users,
        products,
        orders,
        interactions,
        model_cf,
        user_id_to_idx,
        product_id_to_idx,
        label_encoders=label_encoders,
    )
    if X.empty:
        print("No features built.")
        sys.exit(0)

    # 3. Rank and top-N
    avail = [c for c in feature_cols if c in X.columns]
    X_pred = X[avail].fillna(-1)
    proba = model_rank.predict_proba(X_pred)[:, 1]
    X["score"] = proba
    top = X.nlargest(args.top_n, "score")[["product_id", "score"]]

    print(f"Top-{args.top_n} recommendations for user {args.user_id}:")
    for _, row in top.iterrows():
        print(f"  {row['product_id']}\tP(buy)={row['score']:.4f}")


if __name__ == "__main__":
    main()
