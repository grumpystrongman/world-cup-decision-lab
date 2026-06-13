from __future__ import annotations

import argparse
from pathlib import Path
from src.ingest.load_data import load_results
from src.features.elo import build_elo_history
from src.features.match_features import add_recent_form_features
from src.models.train_match_model import train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-sample", action="store_true", help="Use the included sample dataset for a quick demo.")
    args = parser.parse_args()

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    results = load_results(use_sample=args.use_sample)
    elo_history, ratings = build_elo_history(results)
    features = add_recent_form_features(elo_history)

    elo_history.to_csv("data/processed/elo_history.csv", index=False)
    ratings.to_csv("data/processed/team_ratings.csv", index=False)
    features.to_csv("data/processed/match_features.csv", index=False)
    metrics = train_model(features)

    print("Pipeline complete")
    print(f"Matches processed: {len(results):,}")
    print(f"Teams rated: {len(ratings):,}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Log loss: {metrics['log_loss']:.3f}")


if __name__ == "__main__":
    main()
