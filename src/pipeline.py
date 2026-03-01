"""
End-to-end pipeline: load data -> train CF -> generate candidates -> train ranking model.
"""
from pathlib import Path
from typing import Optional

import pandas as pd
from scipy.sparse import csr_matrix

from .data_loader import load_data
from .stage1_candidate import (
    train_cf,
    get_candidates,
    save_model_cf,
    DEFAULT_K_CANDIDATES,
)
from .stage2_ranking import (
    build_features,
    get_feature_columns,
    train_rank,
    save_model_rank,
)


def run_pipeline(
    data_dir: str | Path,
    model_dir: str | Path,
    cf_k_candidates: int = DEFAULT_K_CANDIDATES,
    cf_factors: int = 64,
    cf_iterations: int = 20,
    random_state: int = 42,
) -> None:
    """
    1. Load data and build implicit matrix.
    2. Train model_cf (ALS), save to model_dir/model_cf.joblib.
    3. For each user, get top-K candidates; build (user, product, label) with positives from orders.
    4. Build features and train model_rank, save to model_dir/model_rank.joblib.
    """
    data_dir = Path(data_dir)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load
    data = load_data(data_dir)
    users = data["users"]
    products = data["products"]
    orders = data["orders"]
    interactions = data["interactions"]
    implicit_matrix = data["implicit_matrix"]
    user_id_to_idx = data["user_id_to_idx"]
    idx_to_user_id = data["idx_to_user_id"]
    product_id_to_idx = data["product_id_to_idx"]
    idx_to_product_id = data["idx_to_product_id"]

    # Active products for filtering
    active_ids = set(products[products["is_active"] == 1]["product_id"].astype(str)) if "is_active" in products.columns else set(products["product_id"].astype(str))
    bought_by_user: dict[str, set[str]] = {}
    for _, row in orders.iterrows():
        uid = str(row["user_id"])
        pid = str(row["product_id"])
        bought_by_user.setdefault(uid, set()).add(pid)

    # 2. Train CF
    model_cf = train_cf(
        implicit_matrix,
        factors=cf_factors,
        iterations=cf_iterations,
        random_state=random_state,
    )
    save_model_cf(
        model_cf,
        user_id_to_idx,
        idx_to_user_id,
        product_id_to_idx,
        idx_to_product_id,
        path=model_dir / "model_cf.joblib",
    )

    # 3. Build ranking dataset: positives + negatives (candidates)
    pairs_rows = []
    for _, row in orders.iterrows():
        pairs_rows.append({"user_id": row["user_id"], "product_id": row["product_id"], "label": 1})

    for uid in user_id_to_idx:
        candidates = get_candidates(
            uid,
            model_cf,
            implicit_matrix,
            user_id_to_idx,
            idx_to_product_id,
            K=cf_k_candidates,
            filter_already_bought=True,
            bought_product_ids=bought_by_user.get(uid),
            filter_inactive=True,
            active_product_ids=active_ids,
        )
        for pid, _ in candidates:
            pairs_rows.append({"user_id": uid, "product_id": pid, "label": 0})

    if not pairs_rows:
        raise ValueError("No pairs for ranking (no orders or no candidates).")
    pairs = pd.DataFrame(pairs_rows)

    # 4. Features and train rank
    X, label_encoders = build_features(
        pairs,
        users,
        products,
        orders,
        interactions,
        model_cf,
        user_id_to_idx,
        product_id_to_idx,
    )
    if X.empty or "label" not in X.columns:
        raise ValueError("Feature building produced no rows or no label column.")
    feature_cols = [c for c in get_feature_columns() if c in X.columns]
    model_rank = train_rank(
        X,
        target_col="label",
        feature_cols=feature_cols,
        random_state=random_state,
    )
    save_model_rank(
        model_rank,
        label_encoders,
        feature_cols,
        path=model_dir / "model_rank.joblib",
    )
