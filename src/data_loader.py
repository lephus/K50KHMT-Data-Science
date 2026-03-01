"""
Load CSV datasets, build user/item mappings and implicit user-item matrix
for Matrix Factorization (Stage 1) and ranking (Stage 2).
"""
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


# Weights for implicit feedback (orders = purchase, interactions = signals)
WEIGHT_ORDER = 1.0
WEIGHT_ADD_TO_CART = 0.7
WEIGHT_VIEW = 0.3
WEIGHT_SEARCH = 0.2


def load_csv_dir(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all CSV files from data_dir. Returns (users, products, orders, interactions, email_logs)."""
    data_dir = Path(data_dir)
    users = pd.read_csv(data_dir / "users.csv")
    products = pd.read_csv(data_dir / "products.csv")
    orders = pd.read_csv(data_dir / "orders.csv")
    interactions = pd.read_csv(data_dir / "interactions.csv")
    email_logs = pd.read_csv(data_dir / "email_logs.csv")
    return users, products, orders, interactions, email_logs


def build_mappings(
    users: pd.DataFrame,
    products: pd.DataFrame,
    orders: pd.DataFrame,
    interactions: pd.DataFrame,
) -> tuple[dict[str, int], dict[int, str], dict[str, int], dict[int, str]]:
    """
    Build bidirectional mappings: user_id <-> user_idx, product_id <-> product_idx.
    Indices are 0..n-1 for compatibility with implicit (ALS) and sparse matrices.
    Returns: (user_id_to_idx, idx_to_user_id, product_id_to_idx, idx_to_product_id).
    """
    all_user_ids = pd.concat([
        users["user_id"],
        orders["user_id"],
        interactions["user_id"],
    ]).drop_duplicates()
    all_product_ids = pd.concat([
        products["product_id"],
        orders["product_id"],
        interactions["product_id"],
    ]).drop_duplicates()
    user_id_to_idx = {uid: i for i, uid in enumerate(sorted(all_user_ids.astype(str)))}
    idx_to_user_id = {i: uid for uid, i in user_id_to_idx.items()}
    product_id_to_idx = {pid: i for i, pid in enumerate(sorted(all_product_ids.astype(str)))}
    idx_to_product_id = {i: pid for pid, i in product_id_to_idx.items()}
    return user_id_to_idx, idx_to_user_id, product_id_to_idx, idx_to_product_id


def build_implicit_matrix(
    orders: pd.DataFrame,
    interactions: pd.DataFrame,
    user_id_to_idx: dict[str, int],
    product_id_to_idx: dict[str, int],
    weight_order: float = WEIGHT_ORDER,
    weight_add_to_cart: float = WEIGHT_ADD_TO_CART,
    weight_view: float = WEIGHT_VIEW,
    weight_search: float = WEIGHT_SEARCH,
) -> csr_matrix:
    """
    Build sparse user-item matrix of implicit strengths.
    Shape (n_users, n_items). Same index order as mappings.
    """
    n_users = len(user_id_to_idx)
    n_items = len(product_id_to_idx)
    events: list[tuple[int, int, float]] = []

    for _, row in orders.iterrows():
        u = user_id_to_idx.get(str(row["user_id"]))
        p = product_id_to_idx.get(str(row["product_id"]))
        if u is not None and p is not None:
            # Aggregate by (user, product): take max so multiple orders still 1.0 or sum
            events.append((u, p, weight_order))

    for _, row in interactions.iterrows():
        u = user_id_to_idx.get(str(row["user_id"]))
        p = product_id_to_idx.get(str(row["product_id"]))
        if u is None or p is None:
            continue
        et = str(row.get("event_type", "")).strip().lower()
        if "add_to_cart" in et:
            w = weight_add_to_cart
        elif "view" in et:
            w = weight_view
        elif "search" in et:
            w = weight_search
        else:
            w = weight_view
        events.append((u, p, w))

    if not events:
        return csr_matrix((n_users, n_items))

    # Aggregate (user, item) -> max strength (or sum; plan says "strength" so max is ok)
    from collections import defaultdict
    agg: dict[tuple[int, int], float] = defaultdict(float)
    for u, p, w in events:
        agg[(u, p)] = max(agg[(u, p)], w)

    rows = [k[0] for k in agg]
    cols = [k[1] for k in agg]
    data = list(agg.values())
    return csr_matrix((data, (rows, cols)), shape=(n_users, n_items))


def load_data(
    data_dir: str | Path,
) -> dict:
    """
    Load all data and build mappings + implicit matrix.
    Returns a dict with:
      - users, products, orders, interactions, email_logs (DataFrames)
      - user_id_to_idx, idx_to_user_id, product_id_to_idx, idx_to_product_id
      - implicit_matrix (scipy.sparse.csr_matrix, shape n_users x n_items)
    """
    users, products, orders, interactions, email_logs = load_csv_dir(data_dir)
    user_id_to_idx, idx_to_user_id, product_id_to_idx, idx_to_product_id = build_mappings(
        users, products, orders, interactions
    )
    implicit_matrix = build_implicit_matrix(
        orders, interactions, user_id_to_idx, product_id_to_idx
    )
    return {
        "users": users,
        "products": products,
        "orders": orders,
        "interactions": interactions,
        "email_logs": email_logs,
        "user_id_to_idx": user_id_to_idx,
        "idx_to_user_id": idx_to_user_id,
        "product_id_to_idx": product_id_to_idx,
        "idx_to_product_id": idx_to_product_id,
        "implicit_matrix": implicit_matrix,
    }
