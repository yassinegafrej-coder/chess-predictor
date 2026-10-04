# Chess Game Predictor

A machine learning system that predicts chess position outcomes using a
Random Forest model combined with Stockfish 19 evaluation.

## What it does

Given a chess position (PGN game or FEN string), predicts:
- White wins
- Draw
- Black wins

With a probability for each outcome.

## Results

- Test accuracy: **92.5%** (40 held-out games)
- Trained on 200 master-level games (2600+ ELO)

## Top features (by importance)

1. stockfish_eval (58.6%)
2. material_balance (11.6%)
3. ply (9.4%)
4. elo_diff (8.0%)
5. stockfish_vs_material
6. total_pieces
7. bishops (White/Black)

## Rule-based overrides

For positions where the model has limited training data:

- Stockfish sees forced mate → 97% confidence
- Stockfish eval >= 700 cp → 90% confidence
- Otherwise → use the ML model

## Tech stack (all free)

Python 3.14 · python-chess · Stockfish 19 · pandas · numpy
· scikit-learn · matplotlib · Jupyter · Streamlit

## How to run

Activate the environment and launch the app:

    E:
    cd chess-predictor
    venv\Scripts\activate
    streamlit run app.py

Then open http://localhost:8501

## How to use

**PGN tab:** paste a full game, app analyzes position at move 40.

**FEN tab:** paste a FEN string + ELOs, get prediction.
Use https://lichess.org/analysis to set up positions visually.

## Files

- `chess-predictor.ipynb` — data loading, feature extraction, model training
- `app.py` — Streamlit web app (two tabs)
- `chess_model.pkl` — trained Random Forest
- `feature_cols.pkl` — feature column names
- `mega2600_part_01.pgn` — training data
- `features_df_backup.csv` — cached feature table

## Built without spending a cent

All components free. No API keys. Fully local.