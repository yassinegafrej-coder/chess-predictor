# ♟️ Chess Game Predictor

An ML-powered chess analysis application that predicts game outcomes, classifies move quality, and provides a complete interactive review dashboard — all running locally with free, open-source tools.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Overview

Chess Game Predictor combines classical chess engine analysis with machine learning to give you a full game review experience. It supports PGN input, FEN positions, an interactive board, pre-match outcome prediction, and a 9-chart review dashboard.

Two trained ML models power the app:

- **Position classifier** — 89% accuracy on 5,000 master games
- **Pre-match dual-model predictor** — ELO-based outcome prediction

Stockfish 19 is used as both a feature source and a move-classification engine.

---

## Features

- **Two trained ML models** — position classifier and pre-match dual-model predictor
- **Stockfish 19 integration** — position analysis and move classification
- **Rule-based overrides** — mate positions (97% confidence) and large advantages (90% confidence)
- **Move classification** — every move tagged ⭐ Best / 🌟 Excellent / ✅ Good / ⚠️ Inaccuracy / ❌ Mistake / 💥 Blunder
- **Opening detection** — local ECO database with 3,500+ openings
- **Move explanations** — automatic text explaining why each move was good or bad
- **Interactive board** — legal move hints, best-move arrows, undo, move history with click-to-jump, and 5 board themes
- **Auto-analysis** — predictions update automatically as you play moves
- **Position browser** — slider to jump to any position in the game, with inline re-analysis

---

## Tabs

| Tab | Purpose |
|-----|---------|
| 📄 **PGN Input** | Paste a complete game, get position prediction + move-by-move analysis |
| 🎯 **Position (FEN)** | Paste any chess position, get Stockfish + ML prediction with color-coded eval |
| 🎨 **Interactive Board** | Play/drag pieces with real-time prediction, legal move hints, best-move arrow, undo, and 5 board themes |
| ⚔️ **Pre-Match** | Predict the outcome before a game starts using ELO-based dual models |
| 📊 **Review** | Complete game analysis dashboard with 9 interactive charts |

---

## Review Dashboard

When you load a game, the Review tab gives you 9 visualizations:

1. **Evaluation graph** — dual-color (blue = White, red = Black) showing how the position swung over time
2. **Material balance** — piece values tracked move by move
3. **Piece activity heatmap** — which squares each side controls in the final position
4. **Piece square heatmap** — where each piece type traveled over the game
5. **King safety timeline** — how exposed each king became
6. **Move quality distribution** — count of each move classification per player
7. **Accuracy per player** — computed from average centipawn loss
8. **CPL timeline** — colored bars showing how much each move cost, with filter by side
9. **Mistake density by phase** — opening / middlegame / endgame

Plus a **position browser** with a slider that jumps to any move in the game and lets you re-analyze on the spot.

Toggle between **Tabs view** (focused) and **Full scroll view** (all charts stacked, good for screenshots).

---

## How It Works

### Position Model

The position model is a Random Forest classifier trained on 5,000 master games. It uses features such as material count, piece-square values, mobility, king safety, castling rights, side to move, and Stockfish evaluation. It outputs probabilities for White win, Draw, and Black win.

Rule-based overrides handle special cases:

- Mate positions are predicted with 97% confidence
- Large advantages are predicted with 90% confidence

### Pre-Match Model

A separate classifier uses only:

- `elo_diff`
- `avg_elo`
- `white_advantage`

It has a heuristic for close matchups that predicts a Draw, and overrides for extreme ELO gaps where the higher-rated player wins about 92% of the time.

### Move Classification

Each move is tagged using Stockfish evaluation changes and centipawn loss:

- ⭐ Best
- 🌟 Excellent
- ✅ Good
- ⚠️ Inaccuracy
- ❌ Mistake
- 💥 Blunder

### Pipeline

```text
PGN / FEN
    │
    ▼
python-chess
    │
    ├──► Feature extraction ──► Random Forest ──► Prediction
    │
    └──► Stockfish ──► Evaluation + Best move ──► Move classification
```

---

## Tech Stack

| Component | Purpose |
|-----------|---------|
| Python 3.10+ | Core language |
| python-chess | Chess rules, PGN parsing, board representation |
| Stockfish 19 | Chess engine for position analysis and move classification |
| pandas + numpy | Data manipulation |
| scikit-learn | Machine learning (Random Forest) |
| Plotly | Interactive charts |
| matplotlib | Supplementary plotting |
| Streamlit | Web application framework |
| Jupyter | Model training notebooks |

---

## Project Structure

```text
chess-predictor/
├── app.py                    # Main Streamlit application
├── modules/
│   ├── chess.py              # Chessboard component (HTML/JS wrapper)
│   ├── states.py             # Session state helpers
│   └── utility.py            # Small helpers
├── img/                      # Chess piece images
├── board.html                # Chessboard.js template
├── chess_model.pkl           # Trained position model
├── prematch_model.pkl        # Balanced pre-match model
├── prematch_model_unbalanced.pkl
├── feature_cols.pkl          # Feature column names
├── eco_*.tsv                 # ECO opening database (5 files)
├── chess-predictor.ipynb     # Model training notebook
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10 or newer
- Stockfish 19
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/yassinegafrej-coder/chess-predictor.git
cd chess-predictor

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

### Stockfish Setup

1. Download Stockfish from [stockfishchess.org/download](https://stockfishchess.org/download/).
2. Place the executable in the `stockfish/` folder.
3. Update the engine path in `app.py` if your filename differs.

Example Windows path:

```text
stockfish/stockfish-windows-x86-64-universal.exe
```

For macOS or Linux, use the appropriate binary name. For Streamlit Cloud, you must use a Linux-compatible Stockfish binary.

### Run the App

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal.

---

## Deployment

The app is designed to run locally first. For Streamlit Cloud deployment:

- Use a Linux-compatible Stockfish binary.
- Do not use the Windows `.exe` on Streamlit Cloud.
- Pin Python to a supported version such as 3.11 or 3.12 if needed.
- Ensure all `.pkl` model files and ECO `.tsv` files are committed.

---

## Dataset & Performance

- **Training data:** 5,000 master games
- **Position model accuracy:** 89%
- **Move classification:** based on Stockfish evaluation changes
- **Opening database:** 3,500+ ECO openings

---

## What I Learned

- How to integrate Stockfish with Python for real-time evaluation
- How to build a Random Forest classifier for chess position prediction
- How to classify move quality using centipawn loss
- How to design an interactive Streamlit dashboard with multiple views
- How to combine rule-based overrides with ML predictions for better reliability
- How to structure a full ML project from data collection to deployment

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgements

- [python-chess](https://python-chess.readthedocs.io/)
- [Stockfish](https://stockfishchess.org/)
- [Streamlit](https://streamlit.io/)
- [scikit-learn](https://scikit-learn.org/)
- [Plotly](https://plotly.com/)