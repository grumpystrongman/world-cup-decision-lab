import math
import pandas as pd


def expected_score(rating_a, rating_b):
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def actual_score(goals_for, goals_against):
    if goals_for > goals_against:
        return 1.0
    if goals_for == goals_against:
        return 0.5
    return 0.0


def margin_multiplier(goal_diff):
    diff = abs(int(goal_diff))
    if diff <= 1:
        return 1.0
    return 1.0 + math.log(diff)


def tournament_weight(tournament):
    name = str(tournament).lower()
    if "world cup" in name and "qual" not in name:
        return 60.0
    if "qual" in name:
        return 35.0
    if any(token in name for token in ["euro", "copa", "africa", "asian", "gold cup", "nations league"]):
        return 45.0
    if "friendly" in name:
        return 20.0
    return 30.0


def build_elo_history(results, base_rating=1500.0, home_advantage=60.0):
    ratings = {}
    rows = []
    for _, match in results.iterrows():
        home = match["home_team"]
        away = match["away_team"]
        ratings.setdefault(home, base_rating)
        ratings.setdefault(away, base_rating)
        home_pre = ratings[home]
        away_pre = ratings[away]
        home_context = home_pre + (0.0 if bool(match["neutral"]) else home_advantage)
        expected_home = expected_score(home_context, away_pre)
        actual_home = actual_score(match["home_score"], match["away_score"])
        k = tournament_weight(match.get("tournament", "")) * margin_multiplier(match["home_score"] - match["away_score"])
        change = k * (actual_home - expected_home)
        ratings[home] = home_pre + change
        ratings[away] = away_pre - change
        if match["home_score"] > match["away_score"]:
            target = "home_win"
        elif match["home_score"] < match["away_score"]:
            target = "away_win"
        else:
            target = "draw"
        row = match.to_dict()
        row.update({
            "home_elo_pre": home_pre,
            "away_elo_pre": away_pre,
            "home_elo_post": ratings[home],
            "away_elo_post": ratings[away],
            "elo_diff_pre": home_context - away_pre,
            "expected_home_pre": expected_home,
            "target": target,
            "home_points": 3 if target == "home_win" else 1 if target == "draw" else 0,
            "away_points": 3 if target == "away_win" else 1 if target == "draw" else 0,
            "home_goal_diff": int(match["home_score"] - match["away_score"]),
            "away_goal_diff": int(match["away_score"] - match["home_score"]),
        })
        rows.append(row)
    history = pd.DataFrame(rows)
    rating_table = pd.DataFrame([{"team": team, "elo": rating} for team, rating in ratings.items()])
    rating_table = rating_table.sort_values("elo", ascending=False).reset_index(drop=True)
    rating_table["rank"] = range(1, len(rating_table) + 1)
    return history, rating_table
