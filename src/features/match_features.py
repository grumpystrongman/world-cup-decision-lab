from collections import defaultdict, deque

import pandas as pd


def add_recent_form_features(elo_history, window=5):
    form = defaultdict(lambda: deque(maxlen=window))
    rows = []

    for _, row in elo_history.sort_values("date").iterrows():
        home = row["home_team"]
        away = row["away_team"]
        home_form = list(form[home])
        away_form = list(form[away])

        def avg(items, key, default):
            if not items:
                return default
            return float(sum(item[key] for item in items) / len(items))

        def volatility(items, key):
            if len(items) < 2:
                return 0.0
            series = pd.Series([item[key] for item in items], dtype="float")
            return float(series.std())

        enriched = row.to_dict()
        enriched["home_recent_points"] = avg(home_form, "points", 1.5)
        enriched["away_recent_points"] = avg(away_form, "points", 1.5)
        enriched["home_recent_goal_diff"] = avg(home_form, "goal_diff", 0.0)
        enriched["away_recent_goal_diff"] = avg(away_form, "goal_diff", 0.0)
        enriched["home_form_volatility"] = volatility(home_form, "points")
        enriched["away_form_volatility"] = volatility(away_form, "points")
        enriched["recent_points_diff"] = enriched["home_recent_points"] - enriched["away_recent_points"]
        enriched["recent_goal_diff_diff"] = enriched["home_recent_goal_diff"] - enriched["away_recent_goal_diff"]
        enriched["form_volatility_diff"] = enriched["home_form_volatility"] - enriched["away_form_volatility"]

        elo_diff = float(enriched.get("elo_diff_pre", 0.0))
        expected_home = float(enriched.get("expected_home_pre", 0.5))
        enriched["abs_elo_diff_pre"] = abs(elo_diff)
        enriched["expected_uncertainty"] = 1.0 - abs(expected_home - 0.5) * 2.0
        enriched["close_match_flag"] = 1 if abs(elo_diff) < 75 else 0
        enriched["draw_likelihood"] = max(0.0, min(1.0, enriched["expected_uncertainty"]))
        enriched["upset_risk"] = max(0.0, min(1.0, 1.0 - abs(elo_diff) / 400.0))

        rows.append(enriched)

        form[home].append({"points": row["home_points"], "goal_diff": row["home_goal_diff"]})
        form[away].append({"points": row["away_points"], "goal_diff": row["away_goal_diff"]})

    return pd.DataFrame(rows)
