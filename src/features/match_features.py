from collections import defaultdict, deque

import pandas as pd


def _clamp(value, low=0.0, high=1.0):
    return max(low, min(high, float(value)))


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

        def trend(items, key):
            if len(items) < 2:
                return 0.0
            recent = float(items[-1][key])
            earlier = float(items[0][key])
            return recent - earlier

        enriched = row.to_dict()
        enriched["home_recent_points"] = avg(home_form, "points", 1.5)
        enriched["away_recent_points"] = avg(away_form, "points", 1.5)
        enriched["home_recent_goal_diff"] = avg(home_form, "goal_diff", 0.0)
        enriched["away_recent_goal_diff"] = avg(away_form, "goal_diff", 0.0)
        enriched["home_form_volatility"] = volatility(home_form, "points")
        enriched["away_form_volatility"] = volatility(away_form, "points")
        enriched["home_goal_trend"] = trend(home_form, "goal_diff")
        enriched["away_goal_trend"] = trend(away_form, "goal_diff")
        enriched["recent_points_diff"] = enriched["home_recent_points"] - enriched["away_recent_points"]
        enriched["recent_goal_diff_diff"] = enriched["home_recent_goal_diff"] - enriched["away_recent_goal_diff"]
        enriched["form_volatility_diff"] = enriched["home_form_volatility"] - enriched["away_form_volatility"]
        enriched["goal_trend_diff"] = enriched["home_goal_trend"] - enriched["away_goal_trend"]

        elo_diff = float(enriched.get("elo_diff_pre", 0.0))
        expected_home = float(enriched.get("expected_home_pre", 0.5))
        abs_elo = abs(elo_diff)
        uncertainty = 1.0 - abs(expected_home - 0.5) * 2.0
        form_signal = enriched["recent_points_diff"] + 0.35 * enriched["recent_goal_diff_diff"]
        elo_signal = elo_diff / 400.0

        enriched["abs_elo_diff_pre"] = abs_elo
        enriched["expected_uncertainty"] = _clamp(uncertainty)
        enriched["close_match_flag"] = 1 if abs_elo < 75 else 0
        enriched["draw_likelihood"] = _clamp(enriched["expected_uncertainty"])
        enriched["upset_risk"] = _clamp(1.0 - abs_elo / 400.0)

        # Discontinuity features: the places where tournament football breaks from a smooth Elo curve.
        enriched["form_spike"] = _clamp(max(0.0, abs(form_signal) - 1.0) / 2.5)
        enriched["form_crash"] = _clamp(max(0.0, -form_signal - 1.0) / 2.5)
        enriched["rating_form_disagreement"] = _clamp(abs(elo_signal - form_signal / 3.0))
        enriched["favorite_fragility"] = _clamp((1.0 if abs_elo > 120 else 0.4) * enriched["upset_risk"] * (1.0 + abs(enriched["form_volatility_diff"]) / 2.0))
        enriched["low_scoring_draw_proxy"] = _clamp(enriched["expected_uncertainty"] * (1.0 - min(1.0, abs(enriched["recent_goal_diff_diff"]) / 3.0)))
        enriched["tactical_mismatch_proxy"] = _clamp(abs(enriched["recent_goal_diff_diff"] - elo_signal) / 3.0)
        enriched["pressure_discontinuity_proxy"] = _clamp(enriched["expected_uncertainty"] * (1.0 + enriched["upset_risk"]) / 2.0)

        rows.append(enriched)

        form[home].append({"points": row["home_points"], "goal_diff": row["home_goal_diff"]})
        form[away].append({"points": row["away_points"], "goal_diff": row["away_goal_diff"]})

    return pd.DataFrame(rows)
