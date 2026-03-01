#!/usr/bin/env python3
"""
Train the two-stage recommendation pipeline: model_cf (candidate generation) and model_rank (ranking).
Usage:
  python train_pipeline.py [--data-dir datasets] [--model-dir models]
"""
import argparse
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train model_cf and model_rank")
    parser.add_argument("--data-dir", type=str, default="datasets", help="Path to datasets directory")
    parser.add_argument("--model-dir", type=str, default="models", help="Path to save models")
    parser.add_argument("--cf-k", type=int, default=100, help="Number of candidates per user (Stage 1)")
    parser.add_argument("--cf-factors", type=int, default=64, help="ALS factors")
    parser.add_argument("--cf-iterations", type=int, default=20, help="ALS iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"Error: data directory not found: {data_dir}")
        sys.exit(1)

    run_pipeline(
        data_dir=data_dir,
        model_dir=Path(args.model_dir),
        cf_k_candidates=args.cf_k,
        cf_factors=args.cf_factors,
        cf_iterations=args.cf_iterations,
        random_state=args.seed,
    )
    print("Done. model_cf and model_rank saved to", args.model_dir)


if __name__ == "__main__":
    main()
