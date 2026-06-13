# World Cup 2026 Decision Intelligence Lab

An open-source-first sports analytics platform for World Cup simulation, scenario testing, and executive-style storytelling.

This is intentionally bigger than a basic prediction model. The goal is to show the full analytics product pattern:

- data ingestion
- team strength engineering
- match prediction
- Monte Carlo tournament simulation
- scenario analysis
- explainability
- LinkedIn-ready executive narratives

## Fast start

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python build_pipeline.py --use-sample
streamlit run app.py
```

## Real data setup

Download the public international football results dataset and save the main results file here:

```text
data/raw/results.csv
```

Expected columns:

```text
date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
```

Then run:

```bash
python build_pipeline.py
streamlit run app.py
```

## What V1 does

1. Builds Elo-style team ratings from historical match outcomes.
2. Creates match-level features including Elo difference, expected score, neutral site flag, recent points, and recent goal differential.
3. Trains a multiclass match outcome model: home win, draw, away win.
4. Simulates tournament outcomes thousands of times.
5. Allows scenario adjustments such as a team losing strength because of injury, fatigue, travel burden, or roster instability.
6. Generates explainable outputs and plain-English summaries.

## Project structure

```text
app.py                         Streamlit app
build_pipeline.py              Build ratings, features, and trained model
src/ingest/                    Data loading and validation
src/features/                  Elo and feature engineering
src/models/                    Training and prediction helpers
src/simulation/                Tournament simulation
src/explainability/            Feature importance and narrative explanation
data/manual/                   Hand-maintained tournament teams/scenarios
data/sample/                   Tiny sample dataset for immediate demo
data/raw/                      Real source data, ignored by git
data/processed/                Generated CSV/model artifacts, ignored by git
```

## Databricks path

The repo is local-first, but Databricks-ready:

1. Load raw match results into a Unity Catalog volume or DBFS path.
2. Run `build_pipeline.py` as a Databricks job.
3. Persist processed tables as Delta instead of local parquet.
4. Track model metrics with MLflow.
5. Keep Streamlit local for demo, or deploy as a Databricks app when ready.

## Product positioning

The LinkedIn angle is not "I predicted the World Cup."

The stronger positioning is:

> I built a World Cup Decision Intelligence Lab: an explainable simulation platform that lets users ask how injuries, form, draw difficulty, and team strength change tournament outcomes.

That tells a stronger leadership story: product thinking, AI explainability, simulation, architecture, and communication.
