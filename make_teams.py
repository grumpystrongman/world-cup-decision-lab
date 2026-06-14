import pandas as pd
from pathlib import Path

ratings = pd.read_csv("data/processed/team_ratings.csv").head(48).copy()
groups = list("ABCDEFGHIJKL")

ratings["confederation"] = "TBD"
ratings["group"] = [groups[i % 12] for i in range(len(ratings))]

out = ratings[["team", "confederation", "group"]]

Path("data/manual").mkdir(parents=True, exist_ok=True)
out.to_csv("data/manual/world_cup_2026_teams.csv", index=False)

print(out.to_string(index=False))
print("Created data/manual/world_cup_2026_teams.csv")
