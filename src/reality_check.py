from __future__ import annotations

from pathlib import Path

import pandas as pd

REALITY_CHECK_PATH = Path("data/actual/model_reality_check.csv")


def load_reality_check(path: Path = REALITY_CHECK_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def reality_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "matches_scored": 0,
            "accuracy": 0.0,
            "avg_confidence": 0.0,
            "correct_count": 0,
            "incorrect_count": 0,
        }

    correct = df["correct"].astype(bool)
    return {
        "matches_scored": int(len(df)),
        "accuracy": float(correct.mean()),
        "avg_confidence": float(df["model_confidence"].mean()) if "model_confidence" in df.columns else 0.0,
        "correct_count": int(correct.sum()),
        "incorrect_count": int((~correct).sum()),
    }


def reality_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    columns = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "actual_outcome",
        "model_pick",
        "model_confidence",
        "home_win_prob",
        "draw_prob",
        "away_win_prob",
        "correct",
    ]
    available = [column for column in columns if column in df.columns]
    return df[available].copy()
