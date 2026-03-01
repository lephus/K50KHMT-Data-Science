"""
Stage 1: Candidate Generation via Matrix Factorization (ALS).
Train model_cf, expose get_candidates(user_id, K), save/load model + mappings.
"""
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix


DEFAULT_FACTORS = 64
DEFAULT_REGULARIZATION = 0.01
DEFAULT_ITERATIONS = 20
DEFAULT_K_CANDIDATES = 100


def train_cf(
    implicit_matrix: csr_matrix,
    factors: int = DEFAULT_FACTORS,
    regularization: float = DEFAULT_REGULARIZATION,
    iterations: int = DEFAULT_ITERATIONS,
    random_state: Optional[int] = None,
) -> AlternatingLeastSquares:
    """
    Train ALS on user-item matrix (shape n_users x n_items).
    Returns fitted AlternatingLeastSquares model.
    """
    if random_state is not None:
        np.random.seed(random_state)
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        random_state=random_state,
    )
    model.fit(implicit_matrix, show_progress=False)
    return model


def get_candidates(
    user_id: str,
    model: AlternatingLeastSquares,
    implicit_matrix: csr_matrix,
    user_id_to_idx: dict[str, int],
    idx_to_product_id: dict[int, str],
    K: int = DEFAULT_K_CANDIDATES,
    filter_already_bought: bool = True,
    bought_product_ids: Optional[set[str]] = None,
    filter_inactive: bool = True,
    active_product_ids: Optional[set[str]] = None,
) -> list[tuple[str, float]]:
    """
    Return top-K (product_id, score) for the given user_id.
    Uses implicit_matrix so model.recommend can filter already-seen items.
    """
    user_idx = user_id_to_idx.get(str(user_id))
    if user_idx is None:
        return []
    bought_set = set(bought_product_ids) if bought_product_ids else set()
    active_set = set(active_product_ids) if active_product_ids else set()
    n_items = model.item_factors.shape[0]
    n_request = min(K * 2, n_items)  # ask for more in case we filter

    try:
        item_indices, scores = model.recommend(
            userid=user_idx,
            user_items=implicit_matrix,
            N=n_request,
            filter_already_liked_items=filter_already_bought,
        )
    except Exception:
        item_indices, scores = model.recommend(
            userid=user_idx,
            user_items=implicit_matrix,
            N=n_request,
            filter_already_liked_items=False,
        )

    out: list[tuple[str, float]] = []
    for idx, score in zip(item_indices, scores):
        pid = idx_to_product_id.get(int(idx))
        if pid is None:
            continue
        if filter_already_bought and pid in bought_set:
            continue
        if filter_inactive and active_set and pid not in active_set:
            continue
        out.append((pid, float(score)))
        if len(out) >= K:
            break
    return out


def get_cf_score(
    user_id: str,
    product_id: str,
    model: AlternatingLeastSquares,
    user_id_to_idx: dict[str, int],
    product_id_to_idx: dict[str, int],
) -> float:
    """Return CF score for (user_id, product_id). Returns 0.0 if user or product unknown."""
    ui = user_id_to_idx.get(str(user_id))
    pi = product_id_to_idx.get(str(product_id))
    if ui is None or pi is None:
        return 0.0
    return float(np.dot(model.user_factors[ui], model.item_factors[pi]))


def save_model_cf(
    model: AlternatingLeastSquares,
    user_id_to_idx: dict[str, int],
    idx_to_user_id: dict[int, str],
    product_id_to_idx: dict[str, int],
    idx_to_product_id: dict[int, str],
    path: str | Path,
    mappings_path: Optional[str | Path] = None,
) -> None:
    """Save ALS model and optionally mappings to separate file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    mappings = {
        "user_id_to_idx": user_id_to_idx,
        "idx_to_user_id": idx_to_user_id,
        "product_id_to_idx": product_id_to_idx,
        "idx_to_product_id": idx_to_product_id,
    }
    mp = Path(mappings_path) if mappings_path else path.parent / (path.stem + "_mappings.joblib")
    joblib.dump(mappings, mp)


def load_model_cf(
    path: str | Path,
    mappings_path: Optional[str | Path] = None,
) -> tuple[AlternatingLeastSquares, dict[str, int], dict[int, str], dict[str, int], dict[int, str]]:
    """Load ALS model and mappings."""
    path = Path(path)
    mp = Path(mappings_path) if mappings_path else path.parent / (path.stem + "_mappings.joblib")
    model = joblib.load(path)
    mappings = joblib.load(mp)
    return (
        model,
        mappings["user_id_to_idx"],
        mappings["idx_to_user_id"],
        mappings["product_id_to_idx"],
        mappings["idx_to_product_id"],
    )
