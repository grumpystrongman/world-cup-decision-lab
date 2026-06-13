from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = {
    "date", "home_team", "away_team", "home_score", "away_score",
    "tournament", "city", "country", "neutral"
}


def load_results(path="data/raw/results.csv", use_sample=False):
    source = Path("data/sample/results_sample.csv") if use_sample else Path(path)
    if not source.exists():
        source = Path("data/sample/results_sample.csv")
    data = pd.read_csv(source)
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Required columns not found: {sorted(missing)}")
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    data["home_score"] = data["home_score"].astype(int)
    data["away_score"] = data["away_score"].astype(int)
    data["neutral"] = data["neutral"].astype(str).str.lower().isin(["true", "1", "yes"])
    return data.sort_values("date").reset_index(drop=True)


def load_manual_teams(path="data/manual/world_cup_2026_teams.csv"):
    source = Path(path)
    if not source.exists():
        return pd.DataFrame(columns=["team", "confederation", "group"])
    return pd.read_csv(source).dropna(subset=["team", "group"]).copy()
