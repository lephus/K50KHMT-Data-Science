"""
Stage 2: Ranking model (XGBoost) to predict P(purchase) for (user, product) pairs.
Feature engineering from users, products, orders, interactions + CF score.
"""
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from .stage1_candidate import get_cf_score


def _safe_le_transform(le: LabelEncoder, values: pd.Series, fill: int = -1) -> np.ndarray:
    """Transform with LabelEncoder; unknown/missing -> fill."""
    arr = values.astype(str).fillna("__nan__").values
    out = np.full(len(arr), fill, dtype=np.int64)
    classes = set(le.classes_)
    for i, v in enumerate(arr):
        if v in classes:
            out[i] = le.transform([v])[0]
    return out


def build_features(
    pairs: pd.DataFrame,
    users: pd.DataFrame,
    products: pd.DataFrame,
    orders: pd.DataFrame,
    interactions: pd.DataFrame,
    model_cf: Any,
    user_id_to_idx: dict[str, int],
    product_id_to_idx: dict[str, int],
    label_encoders: Optional[dict[str, LabelEncoder]] = None,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """
    pairs: DataFrame with columns user_id, product_id (and optionally label).
    Returns (X DataFrame with feature columns, label_encoders dict for inference).
    """
    if label_encoders is None:
        label_encoders = {}

    # User features
    user_cols = ["gender", "city", "device_type", "traffic_source", "is_logged_in"]
    for c in user_cols:
        if c not in label_encoders and c in users.columns:
            le = LabelEncoder()
            le.fit(users[c].astype(str).fillna("unknown"))
            label_encoders[f"user_{c}"] = le
    users = users.copy()
    users["created_at"] = pd.to_datetime(users["created_at"], errors="coerce")
    users["tenure_days"] = (pd.Timestamp.now() - users["created_at"]).dt.days.fillna(0).clip(lower=0)

    # Product features
    cat_cols = ["category", "sub_category", "price_range", "color", "material", "style", "occasion"]
    for c in cat_cols:
        if c not in products.columns:
            continue
        key = f"product_{c}"
        if key not in label_encoders:
            le = LabelEncoder()
            le.fit(products[c].astype(str).fillna("unknown"))
            label_encoders[key] = le
    products = products.copy()
    if "price" in products.columns:
        products["log_price"] = np.log1p(products["price"].fillna(0))
    else:
        products["log_price"] = 0.0
    if "is_active" in products.columns:
        products["is_active"] = products["is_active"].fillna(0).astype(int)
    else:
        products["is_active"] = 1

    # Interaction counts per (user, product)
    inter = interactions.copy()
    inter["event_type"] = inter["event_type"].astype(str).str.strip().str.lower()
    view_count = inter[inter["event_type"].str.contains("view", na=False)].groupby(["user_id", "product_id"]).size().reset_index(name="view_count")
    atc_count = inter[inter["event_type"].str.contains("add_to_cart", na=False)].groupby(["user_id", "product_id"]).size().reset_index(name="add_to_cart_count")
    search_count = inter[inter["event_type"].str.contains("search", na=False)].groupby(["user_id", "product_id"]).size().reset_index(name="search_count")
    view_count["user_id"] = view_count["user_id"].astype(str)
    view_count["product_id"] = view_count["product_id"].astype(str)
    atc_count["user_id"] = atc_count["user_id"].astype(str)
    atc_count["product_id"] = atc_count["product_id"].astype(str)
    search_count["user_id"] = search_count["user_id"].astype(str)
    search_count["product_id"] = search_count["product_id"].astype(str)

    rows = []
    for _, row in pairs.iterrows():
        uid = str(row["user_id"])
        pid = str(row["product_id"])
        u = users[users["user_id"] == uid]
        p = products[products["product_id"] == pid]
        if u.empty or p.empty:
            continue
        u = u.iloc[0]
        p = p.iloc[0]
        feat = {
            "user_id": uid,
            "product_id": pid,
            "cf_score": get_cf_score(uid, pid, model_cf, user_id_to_idx, product_id_to_idx),
            "user_is_logged_in": int(u.get("is_logged_in", 0)),
            "user_tenure_days": u.get("tenure_days", 0),
            "product_log_price": p.get("log_price", 0),
            "product_is_active": int(p.get("is_active", 1)),
        }
        for col in ["gender", "city", "device_type", "traffic_source"]:
            if col in users.columns:
                le = label_encoders.get(f"user_{col}")
                if le is not None:
                    feat[f"user_{col}"] = _safe_le_transform(le, pd.Series([u.get(col)]), -1)[0]
        for col in cat_cols:
            if col not in products.columns:
                continue
            key = f"product_{col}"
            le = label_encoders.get(key)
            if le is not None:
                feat[key] = _safe_le_transform(le, pd.Series([p.get(col)]), -1)[0]
        vc = view_count[(view_count["user_id"] == uid) & (view_count["product_id"] == pid)]
        feat["view_count"] = vc["view_count"].iloc[0] if len(vc) else 0
        ac = atc_count[(atc_count["user_id"] == uid) & (atc_count["product_id"] == pid)]
        feat["add_to_cart_count"] = ac["add_to_cart_count"].iloc[0] if len(ac) else 0
        sc = search_count[(search_count["user_id"] == uid) & (search_count["product_id"] == pid)]
        feat["search_count"] = sc["search_count"].iloc[0] if len(sc) else 0
        if "label" in row:
            feat["label"] = row["label"]
        rows.append(feat)

    X = pd.DataFrame(rows)
    if X.empty:
        return X, label_encoders
    # Ensure consistent feature columns; fill missing with -1
    all_feat = get_feature_columns()
    for c in all_feat:
        if c not in X.columns:
            X[c] = -1
    return X, label_encoders


def get_feature_columns() -> list[str]:
    """Column order for training/predict (excluding user_id, product_id, label)."""
    return [
        "cf_score",
        "user_is_logged_in",
        "user_tenure_days",
        "user_gender",
        "user_city",
        "user_device_type",
        "user_traffic_source",
        "product_log_price",
        "product_is_active",
        "product_category",
        "product_sub_category",
        "product_price_range",
        "product_color",
        "product_material",
        "product_style",
        "product_occasion",
        "view_count",
        "add_to_cart_count",
        "search_count",
    ]


def train_rank(
    X: pd.DataFrame,
    target_col: str = "label",
    feature_cols: Optional[list[str]] = None,
    random_state: int = 42,
    scale_pos_weight: Optional[float] = None,
    **xgb_params: Any,
) -> xgb.XGBClassifier:
    """Train XGBoost binary classifier for P(purchase)."""
    if feature_cols is None:
        feature_cols = [c for c in X.columns if c not in ("user_id", "product_id", target_col)]
    available = [c for c in feature_cols if c in X.columns]
    X_train = X[available].fillna(-1)
    y_train = X[target_col]
    if scale_pos_weight is None and y_train.sum() > 0:
        scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    params = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": random_state,
        "eval_metric": "logloss",
    }
    params.update(xgb_params)
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight
    clf = xgb.XGBClassifier(**params)
    clf.fit(X_train, y_train)
    return clf


def save_model_rank(
    model: xgb.XGBClassifier,
    label_encoders: dict[str, LabelEncoder],
    feature_cols: list[str],
    path: str | Path,
    encoders_path: Optional[str | Path] = None,
) -> None:
    """Save ranking model and encoders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    ep = Path(encoders_path) if encoders_path else path.parent / (path.stem + "_encoders.joblib")
    joblib.dump({"label_encoders": label_encoders, "feature_cols": feature_cols}, ep)


def load_model_rank(
    path: str | Path,
    encoders_path: Optional[str | Path] = None,
) -> tuple[xgb.XGBClassifier, dict[str, LabelEncoder], list[str]]:
    """Load ranking model and encoders."""
    path = Path(path)
    ep = Path(encoders_path) if encoders_path else path.parent / (path.stem + "_encoders.joblib")
    model = joblib.load(path)
    data = joblib.load(ep)
    return model, data["label_encoders"], data["feature_cols"]
