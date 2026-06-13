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

        enriched = row.to_dict()
        enriched["home_recent_points"] = avg(home_form, "points", 1.5)
        enriched["away_recent_points"] = avg(away_form, "points", 1.5)
        enriched["home_recent_goal_diff"] = avg(home_form, "goal_diff", 0.0)
        enriched["away_recent_goal_diff"] = avg(away_form, "goal_diff", 0.0)
        enriched["recent_points_diff"] = enriched["home_recent_points"] - enriched["away_recent_points"]
        enriched["recent_goal_diff_diff"] = enriched["home_recent_goal_diff"] - enriched["away_recent_goal_diff"]
        rows.append(enriched)

        form[home].append({"points": row["home_points"], "goal_diff": row["home_goal_diff"]})
        form[away].append({"points": row["away_points"], "goal_diff": row["away_goal_diff"]})

    return pd.DataFrame(rows)
