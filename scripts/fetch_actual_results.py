from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

OUT = Path("data/actual/world_cup_results.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"

TEAM_NAME_MAP = {
    "USA": "United States",
    "USMNT": "United States",
    "Côte d'Ivoire": "Ivory Coast",
    "Korea Republic": "South Korea",
    "Czech Republic": "Czechia",
    "Curacao": "Curacao",
    "Curaçao": "Curacao",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "DR Congo": "DR Congo",
    "Congo DR": "DR Congo",
}


def normalize_team(name: str | None) -> str:
    if not name:
        return ""
    cleaned = str(name).strip()
    return TEAM_NAME_MAP.get(cleaned, cleaned)


def fetch_day(day: date) -> dict[str, Any]:
    response = requests.get(
        ESPN_SCOREBOARD_URL,
        params={"dates": day.strftime("%Y%m%d"), "limit": 200},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def parse_events(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for event in payload.get("events", []):
        competitions = event.get("competitions", [])
        if not competitions:
            continue

        competition = competitions[0]
        status_type = competition.get("status", {}).get("type", {})
        if not status_type.get("completed"):
            continue

        competitors = competition.get("competitors", [])
        if len(competitors) != 2:
            continue

        home = next((team for team in competitors if team.get("homeAway") == "home"), competitors[0])
        away = next((team for team in competitors if team.get("homeAway") == "away"), competitors[1])

        home_team = normalize_team(home.get("team", {}).get("displayName"))
        away_team = normalize_team(away.get("team", {}).get("displayName"))
        home_score = int(home.get("score", 0))
        away_score = int(away.get("score", 0))

        if home_score > away_score:
            actual_outcome = "home_win"
        elif home_score < away_score:
            actual_outcome = "away_win"
        else:
            actual_outcome = "draw"

        rows.append(
            {
                "match_id": event.get("id"),
                "date": event.get("date"),
                "name": event.get("name"),
                "short_name": event.get("shortName"),
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "actual_outcome": actual_outcome,
                "status": status_type.get("description"),
                "source": "ESPN public scoreboard",
            }
        )

    return rows


def fetch_results(start: date, end: date) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    day = start
    while day <= end:
        try:
            rows.extend(parse_events(fetch_day(day)))
        except Exception as exc:  # noqa: BLE001
            print(f"Failed {day}: {exc}")
        day += timedelta(days=1)

    if not rows:
        return pd.DataFrame(
            columns=[
                "match_id",
                "date",
                "name",
                "short_name",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "actual_outcome",
                "status",
                "source",
            ]
        )

    return pd.DataFrame(rows).drop_duplicates(subset=["match_id"]).sort_values("date")


def main() -> None:
    start = date(2026, 6, 11)
    end = date.today()
    results = fetch_results(start, end)
    results.to_csv(OUT, index=False)

    print(f"Wrote {len(results):,} completed matches to {OUT}")
    if not results.empty:
        print(results[["date", "home_team", "away_team", "home_score", "away_score", "actual_outcome"]].to_string(index=False))


if __name__ == "__main__":
    main()
