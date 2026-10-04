import streamlit as st
import streamlit.components.v1 as components
import chess
import chess.pgn
import io
import joblib
import pandas as pd
import numpy as np
from stockfish import Stockfish
from st_bridge import bridge
from modules.chess import Chess
import json
import plotly.graph_objects as go
import base64
import chess.svg

# ---------- Configuration ----------
STOCKFISH_PATH = r"E:\chess-predictor\stockfish\stockfish-windows-x86-64-universal.exe"
MODEL_PATH = "chess_model.pkl"
FEATURES_PATH = "feature_cols.pkl"
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}

# ---------- Load assets ----------
@st.cache_resource
def load_assets():
    clf = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURES_PATH)
    prematch_model = joblib.load("prematch_model.pkl")
    prematch_model_unb = joblib.load("prematch_model_unbalanced.pkl")
    prematch_features = joblib.load("prematch_feature_cols.pkl")
    sf = Stockfish(path=STOCKFISH_PATH, depth=15, parameters={"Threads": 2, "Hash": 128})
    return clf, feature_cols, sf, prematch_model, prematch_model_unb, prematch_features

clf, feature_cols, sf, prematch_model, prematch_model_unb, prematch_features = load_assets()


# ---------- Feature helpers ----------
def board_material(board):
    mat_w = mat_b = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p is None:
            continue
        if p.color == chess.WHITE:
            mat_w += PIECE_VALUES[p.piece_type]
        else:
            mat_b += PIECE_VALUES[p.piece_type]
    return mat_w, mat_b


def stockfish_best_move(board, depth=15):
    sf.set_fen_position(board.fen())
    best_uci = sf.get_best_move()
    if not best_uci:
        return None
    try:
        move = chess.Move.from_uci(best_uci)
        san = board.san(move)
        return {"uci": best_uci, "san": san}
    except Exception:
        return {"uci": best_uci, "san": best_uci}


def stockfish_eval_white_pov(board):
    sf.set_fen_position(board.fen())
    ev = sf.get_evaluation()
    if ev["type"] == "cp":
        cp = ev["value"]
    elif ev["type"] == "mate":
        cp = 10000 if ev["value"] > 0 else -10000
    else:
        cp = 0
    if board.turn == chess.BLACK:
        cp = -cp
    return cp


def _eval_cp_white_pov(board):
    sf.set_fen_position(board.fen())
    ev = sf.get_evaluation()
    if ev["type"] == "cp":
        cp = ev["value"]
    elif ev["type"] == "mate":
        cp = 10000 if ev["value"] > 0 else -10000
    else:
        cp = 0
    if board.turn == chess.BLACK:
        cp = -cp
    return cp


def build_features(board, white_elo, black_elo, ply):
    mat_w, mat_b = board_material(board)
    cp = stockfish_eval_white_pov(board)
    return {
        "material_balance": mat_w - mat_b,
        "white_bishops": len(board.pieces(chess.BISHOP, chess.WHITE)),
        "black_bishops": len(board.pieces(chess.BISHOP, chess.BLACK)),
        "total_pieces": len(board.piece_map()),
        "ply": ply,
        "elo_diff": white_elo - black_elo,
        "stockfish_eval": cp,
        "stockfish_vs_material": cp - (mat_w - mat_b) * 100,
    }


def predict_from_features(feats):
    cp = feats["stockfish_eval"]
    if cp == 10000:
        return {"label": "White wins", "confidence": 0.97,
                "p_white": 0.97, "p_draw": 0.02, "p_black": 0.01, "cp": cp}
    if cp == -10000:
        return {"label": "Black wins", "confidence": 0.97,
                "p_white": 0.01, "p_draw": 0.02, "p_black": 0.97, "cp": cp}
    if cp >= 500:
        return {"label": "White wins", "confidence": 0.90,
                "p_white": 0.90, "p_draw": 0.08, "p_black": 0.02, "cp": cp}
    if cp <= -500:
        return {"label": "Black wins", "confidence": 0.90,
                "p_white": 0.02, "p_draw": 0.08, "p_black": 0.90, "cp": cp}
    X_new = pd.DataFrame([feats])[feature_cols]
    proba = clf.predict_proba(X_new)[0]
    pred_class = clf.classes_[np.argmax(proba)]
    probs = dict(zip(clf.classes_, proba))
    label = {1: "White wins", 0: "Draw", -1: "Black wins"}[pred_class]
    return {
        "label": label, "confidence": max(proba),
        "p_white": probs.get(1, 0), "p_draw": probs.get(0, 0),
        "p_black": probs.get(-1, 0), "cp": cp,
    }


def predict_from_pgn(pgn_string, snapshot_ply=80):
    game = chess.pgn.read_game(io.StringIO(pgn_string))
    if game is None:
        return {"error": "Could not parse PGN"}
    headers = game.headers
    white_elo = int(headers.get("WhiteElo", 0) or 0)
    black_elo = int(headers.get("BlackElo", 0) or 0)
    board = game.board()
    moves = list(game.mainline_moves())
    for i, mv in enumerate(moves):
        if i >= snapshot_ply:
            break
        board.push(mv)
    ply = min(snapshot_ply, len(moves))
    feats = build_features(board, white_elo, black_elo, ply)
    pred = predict_from_features(feats)
    return {
        "white": headers.get("White", "?"), "black": headers.get("Black", "?"),
        "white_elo": white_elo, "black_elo": black_elo,
        "snapshot_ply": ply, "material_balance": feats["material_balance"],
        "stockfish_eval": feats["stockfish_eval"], "fen": board.fen(), **pred,
    }


def predict_from_fen(fen_string, white_elo, black_elo):
    try:
        board = chess.Board(fen_string.strip())
    except ValueError as e:
        return {"error": f"Invalid FEN: {e}"}
    actual_ply = (board.fullmove_number - 1) * 2 + (0 if board.turn == chess.WHITE else 1)
    if actual_ply < 60 or actual_ply > 100:
        ply = 80
        ply_warning = f"⚠️ Position is at ply {actual_ply}. Model was trained on mid-game (~ply 80). Results may be less reliable."
    else:
        ply = actual_ply
        ply_warning = None
    feats = build_features(board, white_elo, black_elo, ply)
    pred = predict_from_features(feats)
    return {
        "white_elo": white_elo, "black_elo": black_elo,
        "snapshot_ply": ply, "actual_ply": actual_ply,
        "ply_warning": ply_warning,
        "material_balance": feats["material_balance"],
        "stockfish_eval": feats["stockfish_eval"],
        "fen": board.fen(), **pred,
    }


def _raw_model_predict(model, elo_diff, avg_elo):
    X = pd.DataFrame([{
        "elo_diff": elo_diff, "avg_elo": avg_elo, "white_advantage": 1,
    }])[prematch_features]
    proba = model.predict_proba(X)[0]
    probs = dict(zip(model.classes_, proba))
    pred = model.classes_[np.argmax(proba)]
    label = {1: "White wins", 0: "Draw", -1: "Black wins"}[pred]
    return {
        "label": label, "confidence": max(proba),
        "p_white": probs.get(1, 0), "p_draw": probs.get(0, 0),
        "p_black": probs.get(-1, 0),
    }


def _add_decisive(p):
    total = p["p_white"] + p["p_black"]
    if total > 0.01:
        p["w_if_decisive"] = p["p_white"] / total
        p["b_if_decisive"] = p["p_black"] / total
    else:
        p["w_if_decisive"] = None
        p["b_if_decisive"] = None
    return p


def predict_prematch_both(white_elo, black_elo):
    elo_diff = white_elo - black_elo
    avg_elo = (white_elo + black_elo) / 2
    if abs(elo_diff) < 30:
        shared = {"label": "Draw", "confidence": 0.55,
                  "p_white": 0.28, "p_draw": 0.55, "p_black": 0.17,
                  "source": "heuristic (even matchup)"}
        return {"unbalanced": _add_decisive(dict(shared)),
                "balanced": _add_decisive(dict(shared))}
    if abs(elo_diff) >= 400:
        if elo_diff > 0:
            shared = {"label": "White wins", "confidence": 0.92,
                      "p_white": 0.92, "p_draw": 0.05, "p_black": 0.03,
                      "source": "override (White much stronger)"}
        else:
            shared = {"label": "Black wins", "confidence": 0.92,
                      "p_white": 0.03, "p_draw": 0.05, "p_black": 0.92,
                      "source": "override (Black much stronger)"}
        return {"unbalanced": _add_decisive(dict(shared)),
                "balanced": _add_decisive(dict(shared))}
    unb = _raw_model_predict(prematch_model_unb, elo_diff, avg_elo)
    unb["source"] = "unbalanced ML"
    bal = _raw_model_predict(prematch_model, elo_diff, avg_elo)
    bal["source"] = "balanced ML"
    return {"unbalanced": _add_decisive(unb), "balanced": _add_decisive(bal)}


def eval_to_win_prob(cp):
    return 1.0 / (1.0 + np.exp(-cp / 400.0))


def _rebuild_pgn(moves_list):
    parts = []
    for i in range(0, len(moves_list), 2):
        num = i // 2 + 1
        w = moves_list[i][1] if i < len(moves_list) and moves_list[i][0] == "w" else ""
        b = moves_list[i+1][1] if i+1 < len(moves_list) and moves_list[i+1][0] == "b" else ""
        parts.append(f"{num}. {w} {b}".strip())
    return " ".join(parts)


def explain_move(move_info, fen_before):
    san = move_info.get("san", "?")
    color = move_info.get("color", "?")
    cat = move_info.get("category", "?")
    cpl = move_info.get("cpl", 0)
    best = move_info.get("best_alt", "")
    try:
        board = chess.Board(fen_before)
        mv = board.parse_san(san)
        piece = board.piece_at(mv.from_square)
        captured = board.piece_at(mv.to_square)
        is_check = board.gives_check(mv) if hasattr(board, 'gives_check') else False
        is_castle = board.is_castling(mv)
    except Exception:
        piece = None; captured = None; is_check = False; is_castle = False
    piece_names = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
                   chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king"}
    what = ""
    if is_castle:
        what = "Castles the king to safety."
    elif captured:
        what = f"Captures the {piece_names.get(captured.piece_type, 'piece')}."
    elif is_check:
        what = "Gives check."
    elif piece:
        p = piece_names.get(piece.piece_type, "piece")
        what = f"A strong {p} move." if cat in ("Best", "Excellent") else f"Moves the {p}."
    if cat == "Best":
        why = "This is the engine's top choice — no better move exists."
    elif cat == "Excellent":
        why = "Nearly the best move available."
    elif cat == "Good":
        why = "A solid move."
    elif cat == "Inaccuracy":
        why = f"Slightly imprecise — lost {cpl/100:.1f} pawns of advantage."
    elif cat == "Mistake":
        why = f"This is a mistake — it drops {cpl/100:.1f} pawns."
    elif cat == "Blunder":
        why = f"This is a blunder — it gives away {cpl/100:.1f} pawns."
    else:
        why = ""
    alt = ""
    if best and cat in ("Inaccuracy", "Mistake", "Blunder"):
        alt = f"Better was {best}."
    parts = [p for p in [what, why, alt] if p]
    return " ".join(parts) if parts else "No explanation available."


# ============================================================
# OPENING DETECTION
# ============================================================
_ECO_CACHE = None

def load_eco_database():
    global _ECO_CACHE
    if _ECO_CACHE is not None:
        return _ECO_CACHE
    import os
    entries = []
    for letter in ["a", "b", "c", "d", "e"]:
        path = f"eco_{letter}.tsv"
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                f.readline()
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 3:
                        continue
                    eco = parts[0].strip()
                    name = parts[1].strip()
                    moves_str = parts[2].strip()
                    normalized = normalize_moves(moves_str)
                    if normalized:
                        entries.append((normalized, name, eco))
        except Exception:
            continue
    entries.sort(key=lambda x: len(x[0]), reverse=True)
    _ECO_CACHE = entries
    return entries


def normalize_moves(moves_str):
    import re
    cleaned = re.sub(r"\d+\.+", " ", moves_str)
    tokens = cleaned.split()
    moves = []
    for t in tokens:
        t = t.strip()
        if not t or t in ("1-0", "0-1", "1/2-1/2", "*"):
            continue
        moves.append(t)
    return moves


def detect_opening(pgn_string):
    entries = load_eco_database()
    if not entries:
        return None, None
    game = chess.pgn.read_game(io.StringIO(pgn_string))
    if game is None:
        return None, None
    board = game.board()
    game_moves = []
    for mv in game.mainline_moves():
        try:
            san = board.san(mv)
            game_moves.append(san)
            board.push(mv)
        except Exception:
            break
        if len(game_moves) >= 20:
            break
    if not game_moves:
        return None, None
    for moves_tuple, name, eco in entries:
        n = len(moves_tuple)
        if n <= len(game_moves):
            if game_moves[:n] == moves_tuple:
                return name, eco
    return None, None


def classify_cpl(cpl, eval_before=0, eval_after=0, mover_is_white=True):
    if mover_is_white:
        e_before = eval_before
        e_after = eval_after
    else:
        e_before = -eval_before
        e_after = -eval_after
    if e_before >= 300 and e_after >= 200:
        if cpl <= 5: return "⭐ Best", "Best"
        elif cpl <= 20: return "🌟 Excellent", "Excellent"
        elif cpl <= 50: return "✅ Good", "Good"
        else: return "⚠️ Inaccuracy", "Inaccuracy"
    if e_before <= -300 and e_after <= -200:
        if cpl <= 5: return "⭐ Best", "Best"
        elif cpl <= 20: return "🌟 Excellent", "Excellent"
        elif cpl <= 50: return "✅ Good", "Good"
        else: return "⚠️ Inaccuracy", "Inaccuracy"
    if cpl <= 5: return "⭐ Best", "Best"
    elif cpl <= 20: return "🌟 Excellent", "Excellent"
    elif cpl <= 50: return "✅ Good", "Good"
    elif cpl <= 100: return "⚠️ Inaccuracy", "Inaccuracy"
    elif cpl <= 300: return "❌ Mistake", "Mistake"
    else: return "💥 Blunder", "Blunder"


# ---- Heatmap / Chart Functions ----
def piece_activity_heatmap(fen):
    try:
        board = chess.Board(fen)
    except Exception:
        return [[0]*8 for _ in range(8)]
    grid = [[0]*8 for _ in range(8)]
    for sq in chess.SQUARES:
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)
        try:
            white_att = len(board.attackers(chess.WHITE, sq))
            black_att = len(board.attackers(chess.BLACK, sq))
        except Exception:
            white_att = 0; black_att = 0
        grid[7 - rank][file] = white_att - black_att
    return grid


def piece_square_heatmap(pgn_string, piece_type=None):
    game = chess.pgn.read_game(io.StringIO(pgn_string))
    if game is None:
        return None, None
    white_grid = [[0]*8 for _ in range(8)]
    black_grid = [[0]*8 for _ in range(8)]
    board = game.board()
    for mv in game.mainline_moves():
        piece = board.piece_at(mv.from_square)
        if piece is not None:
            if piece_type is None or piece.piece_type == piece_type:
                for sq in (mv.from_square, mv.to_square):
                    f = chess.square_file(sq)
                    r = chess.square_rank(sq)
                    row = 7 - r
                    if piece.color == chess.WHITE:
                        white_grid[row][f] += 1
                    else:
                        black_grid[row][f] += 1
        board.push(mv)
    return white_grid, black_grid


def king_safety_score(board, color):
    king_sq = board.king(color)
    if king_sq is None:
        return 0
    score = 0
    file = chess.square_file(king_sq)
    rank = chess.square_rank(king_sq)
    pawn_dir = 1 if color == chess.WHITE else -1
    for df in (-1, 0, 1):
        f = file + df
        if f < 0 or f > 7: continue
        for dr in (1, 2):
            r = rank + pawn_dir * dr
            if r < 0 or r > 7: continue
            sq = chess.square(f, r)
            p = board.piece_at(sq)
            if p and p.piece_type == chess.PAWN and p.color == color:
                score += 3 if dr == 1 else 1
                break
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if df == 0 and dr == 0: continue
            f = file + df; r = rank + dr
            if f < 0 or f > 7 or r < 0 or r > 7: continue
            sq = chess.square(f, r)
            score += len(board.attackers(color, sq)) * 0.3
    enemy = chess.BLACK if color == chess.WHITE else chess.WHITE
    enemy_attacks = 0
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            f = file + df; r = rank + dr
            if f < 0 or f > 7 or r < 0 or r > 7: continue
            sq = chess.square(f, r)
            enemy_attacks += len(board.attackers(enemy, sq))
    score -= enemy_attacks * 1.5
    return max(0, score)


def king_safety_timeline(fens):
    white_scores = []
    black_scores = []
    for fen in fens:
        try:
            b = chess.Board(fen)
            white_scores.append(round(king_safety_score(b, chess.WHITE), 1))
            black_scores.append(round(king_safety_score(b, chess.BLACK), 1))
        except Exception:
            white_scores.append(0); black_scores.append(0)
    return white_scores, black_scores


def material_balance_from_fen(fen):
    try:
        board = chess.Board(fen)
    except Exception:
        return 0
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
              chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
    diff = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p is None: continue
        v = values.get(p.piece_type, 0)
        diff += v if p.color == chess.WHITE else -v
    return diff


def compute_eval_graph(pgn_string, max_plies=120, depth=14):
    game = chess.pgn.read_game(io.StringIO(pgn_string))
    if game is None:
        return {"error": "Could not parse PGN"}
    try:
        sf.set_depth(depth)
    except Exception:
        pass
    board = game.board()
    moves = list(game.mainline_moves())[:max_plies]
    evals = [_eval_cp_white_pov(board)]
    fens = [board.fen()]
    sans = ["(start)"]
    cpls = [0]
    categories = ["Start"]
    best_alts = [""]
    for mv in moves:
        mover_is_white = board.turn == chess.WHITE
        san = board.san(mv)
        fen_before = board.fen()
        before_cp = evals[-1]
        sf.set_fen_position(fen_before)
        best_uci = sf.get_best_move() or ""
        best_san = ""
        if best_uci:
            try:
                best_san = board.san(chess.Move.from_uci(best_uci))
            except Exception:
                best_san = ""
        board.push(mv)
        after_cp = _eval_cp_white_pov(board)
        if mover_is_white:
            cpl = max(0, before_cp - after_cp)
        else:
            cpl = max(0, after_cp - before_cp)
        if best_uci and mv.uci() == best_uci:
            cpl = 0
        label, category = classify_cpl(cpl, before_cp, after_cp, mover_is_white)
        evals.append(after_cp)
        fens.append(board.fen())
        sans.append(san)
        cpls.append(cpl)
        categories.append(category)
        best_alts.append(best_san)
    try:
        sf.set_depth(15)
    except Exception:
        pass
    return {
        "evals": evals, "fens": fens, "sans": sans,
        "cpls": cpls, "categories": categories,
        "best_alts": best_alts, "num_moves": len(moves),
    }


def analyze_game_moves(pgn_string, max_plies=60, depth=10):
    game = chess.pgn.read_game(io.StringIO(pgn_string))
    if game is None:
        return {"error": "Could not parse PGN"}
    try:
        sf.set_depth(depth)
    except Exception:
        pass
    board = game.board()
    moves = list(game.mainline_moves())[:max_plies]
    before_cp = _eval_cp_white_pov(board)
    results = []
    for i, mv in enumerate(moves):
        mover_is_white = board.turn == chess.WHITE
        san = board.san(mv)
        sf.set_fen_position(board.fen())
        best_uci = sf.get_best_move()
        board.push(mv)
        after_cp = _eval_cp_white_pov(board)
        if mover_is_white:
            cpl = max(0, before_cp - after_cp)
        else:
            cpl = max(0, after_cp - before_cp)
        if best_uci and mv.uci() == best_uci:
            cpl = 0
        label, category = classify_cpl(cpl, before_cp, after_cp, mover_is_white)
        results.append({
            "move_number": i // 2 + 1, "ply": i + 1,
            "color": "White" if mover_is_white else "Black",
            "san": san, "cpl": cpl, "label": label, "category": category,
        })
        before_cp = after_cp
    try:
        sf.set_depth(15)
    except Exception:
        pass
    return {"moves": results}


def render_analysis(r):
    if "error" in r:
        st.error(r["error"]); return
    if r.get("ply_warning"):
        st.warning(r["ply_warning"])
    st.subheader("Position analysis")
    c1, c2, c3 = st.columns(3)
    c1.metric("Material balance", f"{r['material_balance']:+d}")
    eval_cp = r["stockfish_eval"]
    if eval_cp >= 200:
        eval_label = f"🟢 +{eval_cp} cp (White winning)"
    elif eval_cp <= -200:
        eval_label = f"🔴 {eval_cp} cp (Black winning)"
    elif eval_cp >= 50:
        eval_label = f"🟡 +{eval_cp} cp (White slightly better)"
    elif eval_cp <= -50:
        eval_label = f"🟡 {eval_cp} cp (Black slightly better)"
    else:
        eval_label = f"⚪ {eval_cp:+d} cp (Even)"
    c2.metric("Stockfish eval", eval_label)
    c3.metric("Ply", r["snapshot_ply"])
    white_prob = eval_to_win_prob(r["stockfish_eval"])
    st.markdown("**Evaluation bar:**")
    eb1, eb2, eb3 = st.columns([1, 4, 1])
    with eb1:
        st.markdown(
            f"<div style='background:#eee;color:#000;padding:8px;border-radius:4px;"
            f"text-align:center;font-weight:bold;'>White {white_prob:.0%}</div>",
            unsafe_allow_html=True)
    with eb2:
        st.markdown(
            f"<div style='display:flex;height:36px;border-radius:6px;"
            f"overflow:hidden;background:#ddd;'>"
            f"<div style='width:{white_prob:.1%};background:#f0f0f0;'></div>"
            f"<div style='width:{1-white_prob:.1%};background:#222;'></div>"
            f"</div>", unsafe_allow_html=True)
    with eb3:
        st.markdown(
            f"<div style='background:#111;color:#fff;padding:8px;border-radius:4px;"
            f"text-align:center;font-weight:bold;'>Black {1 - white_prob:.0%}</div>",
            unsafe_allow_html=True)
    st.divider()
    st.subheader("🎯 Prediction")
    st.markdown(f"### **{r['label']}**  ({r['confidence']:.1%} confidence)")
    st.bar_chart({"White wins": r["p_white"], "Draw": r["p_draw"], "Black wins": r["p_black"]})
    with st.expander("Position details"):
        st.code(r["fen"])
        st.markdown(f"[Open on Lichess](https://lichess.org/analysis/standard/{r['fen'].replace(' ', '_')})")


# ---------- UI ----------
st.set_page_config(page_title="Chess Predictor", page_icon="♟️", layout="wide")
st.title("♟️ Chess Game Predictor")
st.markdown(
    "Three ways to predict: **PGN** (past game), **FEN** (any position), "
    "or the **Interactive Board** (drag pieces and analyze).")

tab_pgn, tab_fen, tab_board, tab_prematch, tab_review = st.tabs([
    "📄 PGN Input", "🎯 Position (FEN)", "🎨 Interactive Board",
    "⚔️ Pre-Match", "📊 Review"])


# ============ TAB 1 — PGN ============
with tab_pgn:
    st.markdown("Paste a full chess game in PGN format. Analyzed at move 40.")
    if st.button("Load sample game", key="load_pgn"):
        st.session_state["pgn"] = """[Event "Sample"]
[White "Speelman, Jonathan"]
[Black "Gulko, Boris"]
[Result "1-0"]
[WhiteElo "2615"]
[BlackElo "2610"]

1. e4 c5 2. Nf3 Nc6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 d6 6. Bg5 Qb6 7. Nb3 e6 8. Bf4 Ne5 9. Be3 Qc7 10. f4 Nc6 11. g4 d5 12. e5 Nd7 13. Nb5 Qd8 14. h4 f6 15. Nd6+ Bxd6 16. exd6 Nb6 17. g5 O-O 18. Qd2 Na4 19. O-O-O a6 20. g6 Qxd6 21. gxh7+ Kh8 22. h5 Bd7 23. Bd3 d4 24. h6 gxh6 25. Qg2 Nb4 26. Rdg1 Nxd3+ 27. Kb1 1-0"""

    pgn_input = st.text_area("PGN:", height=220, key="pgn")

    if st.button("Predict", type="primary", key="predict_pgn"):
        if not pgn_input.strip():
            st.warning("Please paste a PGN first.")
        else:
            with st.spinner("Analyzing..."):
                r = predict_from_pgn(pgn_input)
            if "error" in r:
                st.error(r["error"])
            else:
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("Players")
                    st.write(f"**White:** {r['white']} ({r['white_elo']})")
                    st.write(f"**Black:** {r['black']} ({r['black_elo']})")
                    st.write(f"**Snapshot:** after ply {r['snapshot_ply']}")
                with c2:
                    st.subheader("Position analysis")
                    st.write(f"**Material balance:** {r['material_balance']:+d}")
                    st.write(f"**Stockfish eval:** {r['stockfish_eval']:+d} cp")
                st.divider()
                st.subheader("🎯 Prediction")
                st.markdown(f"### **{r['label']}**  ({r['confidence']:.1%} confidence)")
                st.bar_chart({"White wins": r["p_white"], "Draw": r["p_draw"], "Black wins": r["p_black"]})
                with st.expander("View FEN"):
                    st.code(r["fen"])
                open_name, open_eco = detect_opening(pgn_input)
                if open_name:
                    st.info(f"📖 Opening: **{open_name}** (ECO {open_eco})")
                with st.expander("Copy PGN"):
                    st.code(pgn_input, language="text")
                    st.caption("Use the copy icon in the top-right of the code box above.")

    # ---- Move-by-move analysis ----
    st.divider()
    st.subheader("📊 Move-by-Move Analysis")
    st.caption("Classifies every move: ⭐ Best / 🌟 Excellent / ✅ Good / ⚠️ Inaccuracy / ❌ Mistake / 💥 Blunder")

    if st.button("🔍 Analyze every move", key="analyze_moves", type="primary"):
        if not pgn_input.strip():
            st.warning("Please paste a PGN first.")
        else:
            with st.spinner("Analyzing every move... (~15-30 seconds)"):
                move_results = analyze_game_moves(pgn_input)
            if "error" in move_results:
                st.error(move_results["error"])
            else:
                moves = move_results["moves"]
                from collections import Counter
                w_counts = Counter(m["category"] for m in moves if m["color"] == "White")
                b_counts = Counter(m["category"] for m in moves if m["color"] == "Black")
                categories = ["Best", "Excellent", "Good", "Inaccuracy", "Mistake", "Blunder"]
                emojis = {"Best": "⭐", "Excellent": "🌟", "Good": "✅",
                          "Inaccuracy": "⚠️", "Mistake": "❌", "Blunder": "💥"}
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**White move quality**")
                    for cat in categories:
                        st.write(f"{emojis[cat]} {cat}: **{w_counts.get(cat, 0)}**")
                with c2:
                    st.markdown("**Black move quality**")
                    for cat in categories:
                        st.write(f"{emojis[cat]} {cat}: **{b_counts.get(cat, 0)}**")
                st.markdown("**All moves:**")
                df_moves = pd.DataFrame(moves)
                df_display = df_moves[["move_number", "color", "san", "cpl", "label"]].rename(columns={
                    "move_number": "#", "color": "Side", "san": "Move", "cpl": "CPL", "label": "Rating"})
                st.dataframe(df_display, use_container_width=True, hide_index=True)
                key = [m for m in moves if m["category"] in ("Mistake", "Blunder")]
                if key:
                    st.markdown("**🔴 Key mistakes and blunders:**")
                    for m in key:
                        st.write(f"- Move {m['move_number']} ({m['color']}): **{m['san']}** → {m['label']} (lost **{m['cpl']}** cp)")
                else:
                    st.success("No mistakes or blunders detected! 🎉")


# ============ TAB 2 — FEN ============
with tab_fen:
    st.markdown("Paste a **FEN** string. Use [lichess.org/analysis](https://lichess.org/analysis) to set up positions visually and copy the FEN.")
    fen_input = st.text_input("FEN:", value="r4r1k/1p1b3P/p2qpp1p/8/n2p1P2/1N1nB3/PPP3Q1/1K4RR b - - 1 27", key="fen")
    col_w, col_b = st.columns(2)
    with col_w:
        white_elo = st.number_input("White ELO:", 0, 3500, 2615, 10, key="w_elo")
    with col_b:
        black_elo = st.number_input("Black ELO:", 0, 3500, 2610, 10, key="b_elo")
    if st.button("Predict", type="primary", key="predict_fen"):
        if not fen_input.strip():
            st.warning("Please paste a FEN first.")
        else:
            with st.spinner("Analyzing position..."):
                r = predict_from_fen(fen_input, int(white_elo), int(black_elo))
            render_analysis(r)


# ============ TAB 3 — Interactive Board ============
with tab_board:
    st.markdown("Drag pieces to set up any position. The prediction updates automatically.")

    if "board_fen" not in st.session_state:
        st.session_state.board_fen = START_FEN
    if "board_analysis" not in st.session_state:
        st.session_state.board_analysis = None
    if "analyzed_fen" not in st.session_state:
        st.session_state.analyzed_fen = None
    if "analyzed_elo" not in st.session_state:
        st.session_state.analyzed_elo = None
    if "board_orientation" not in st.session_state:
        st.session_state.board_orientation = "white"
    if "just_reset" not in st.session_state:
        st.session_state.just_reset = False
    if "game_pgn" not in st.session_state:
        st.session_state.game_pgn = ""
    if "moves_list" not in st.session_state:
        st.session_state.moves_list = []
    if "last_move_highlight" not in st.session_state:
        st.session_state.last_move_highlight = None
    if "current_view_index" not in st.session_state:
        st.session_state.current_view_index = -1
    if "board_theme" not in st.session_state:
        st.session_state.board_theme = "classic"

    data = bridge("my-bridge")
    if data and isinstance(data, dict) and "fen" in data:
        if st.session_state.just_reset:
            st.session_state.just_reset = False
        else:
            new_fen = data["fen"]
            if new_fen != st.session_state.board_fen:
                st.session_state.board_fen = new_fen
                st.session_state.board_analysis = None
                st.session_state.best_move = None
                history = data.get("history")
                if isinstance(history, str):
                    try:
                        history = json.loads(history)
                    except Exception:
                        history = None
                if isinstance(history, list) and history:
                    st.session_state.moves_list = [
                        (m.get("color", "w"), m.get("san", ""),
                         m.get("fen", ""), m.get("from", "") + m.get("to", ""))
                        for m in history]
                    st.session_state.current_view_index = len(st.session_state.moves_list) - 1
                    last = history[-1]
                    st.session_state.last_move_highlight = last.get("from", "") + last.get("to", "")
                    st.session_state.game_pgn = _rebuild_pgn(st.session_state.moves_list)

    col_w, col_b = st.columns(2)
    with col_w:
        board_white_elo = st.number_input("White ELO:", 0, 3500, 2615, 10, key="board_w_elo")
    with col_b:
        board_black_elo = st.number_input("Black ELO:", 0, 3500, 2610, 10, key="board_b_elo")

    opt_c1, opt_c2, opt_c3 = st.columns([2, 1, 1])
    with opt_c1:
        auto_analyze = st.checkbox("Auto-analyze after each move", value=True, key="auto_analyze")
    with opt_c2:
        if st.button("🔄 Flip Board", key="flip_board", use_container_width=True):
            st.session_state.board_orientation = ("black" if st.session_state.board_orientation == "white" else "white")
            st.rerun()
    with opt_c3:
        st.caption(f"View: **{st.session_state.board_orientation.capitalize()}**")

    theme_col1, theme_col2 = st.columns([2, 2])
    with theme_col1:
        st.selectbox("Board theme:", ["classic", "green", "blue", "gray", "purple"], key="board_theme")

    bm = st.session_state.get("best_move")
    arrow_uci = bm.get("uci") if isinstance(bm, dict) else None

    board_render = Chess(
        400,
        st.session_state.board_fen,
        orientation=st.session_state.board_orientation,
        highlight_move=st.session_state.last_move_highlight,
        move_history=[
            {"san": m[1], "from": m[3][:2], "to": m[3][2:4], "color": m[0], "fen": m[2]}
            for m in st.session_state.moves_list],
        theme=st.session_state.board_theme,
        arrow_move=arrow_uci,
    )
    components.html(board_render.puzzle_board(), height=720, scrolling=False)

    current_elo = (int(board_white_elo), int(board_black_elo))
    needs_analysis = (
        st.session_state.analyzed_fen != st.session_state.board_fen
        or st.session_state.analyzed_elo != current_elo)

    if auto_analyze and needs_analysis:
        with st.spinner("Analyzing position..."):
            st.session_state.board_analysis = predict_from_fen(
                st.session_state.board_fen, current_elo[0], current_elo[1])
            st.session_state.analyzed_fen = st.session_state.board_fen
            st.session_state.analyzed_elo = current_elo

    if not auto_analyze:
        if st.button("🔍 Analyze Position", type="primary", key="manual_analyze", use_container_width=True):
            with st.spinner("Analyzing position..."):
                st.session_state.board_analysis = predict_from_fen(
                    st.session_state.board_fen, int(board_white_elo), int(board_black_elo))
                st.session_state.analyzed_fen = st.session_state.board_fen
                st.session_state.analyzed_elo = (int(board_white_elo), int(board_black_elo))

    bm_col1, bm_col2 = st.columns(2)
    with bm_col1:
        if st.button("💡 Show Best Move", key="best_move_btn", use_container_width=True):
            with st.spinner("Asking Stockfish..."):
                try:
                    temp_board = chess.Board(st.session_state.board_fen)
                    st.session_state.best_move = stockfish_best_move(temp_board)
                except Exception as e:
                    st.session_state.best_move = {"error": str(e)}
            st.rerun()
    with bm_col2:
        if st.button("🔄 Reset to Starting Position", key="reset_board", use_container_width=True):
            st.session_state.board_fen = START_FEN
            st.session_state.board_analysis = None
            st.session_state.analyzed_fen = None
            st.session_state.analyzed_elo = None
            st.session_state.best_move = None
            st.session_state.game_pgn = ""
            st.session_state.moves_list = []
            st.session_state.last_move_highlight = None
            st.session_state.just_reset = True
            st.rerun()

    if st.session_state.get("best_move"):
        bm = st.session_state.best_move
        if "error" in bm:
            st.error(f"Could not compute best move: {bm['error']}")
        else:
            side = "White" if chess.Board(st.session_state.board_fen).turn else "Black"
            st.success(f"💡 **Best move for {side}: {bm['san']}**  (coordinate: `{bm['uci']}`)")

    st.caption(f"Current FEN: `{st.session_state.board_fen}`")

    if st.session_state.board_analysis:
        render_analysis(st.session_state.board_analysis)
    else:
        if not auto_analyze:
            st.info("Click **Analyze** above to run a prediction.")

    # Move History Panel
    with st.expander("📜 Move History (click any move to jump)", expanded=False):
        if not st.session_state.moves_list:
            st.caption("No moves yet — drag pieces on the board.")
        else:
            moves = st.session_state.moves_list
            pairs = []
            for i in range(0, len(moves), 2):
                num = i // 2 + 1
                w = moves[i]
                b = moves[i+1] if i+1 < len(moves) else None
                pairs.append((num, w, b, i))
            for num, w, b, base_idx in pairs:
                cols = st.columns([1, 2, 2])
                cols[0].markdown(f"**{num}.**")
                if cols[1].button(w[1], key=f"mv_w_{base_idx}", use_container_width=True):
                    st.session_state.board_fen = w[2]
                    st.session_state.last_move_highlight = w[3]
                    st.session_state.current_view_index = base_idx
                    st.session_state.board_analysis = None
                    st.session_state.just_reset = True
                    st.rerun()
                if b is not None:
                    if cols[2].button(b[1], key=f"mv_b_{base_idx+1}", use_container_width=True):
                        st.session_state.board_fen = b[2]
                        st.session_state.last_move_highlight = b[3]
                        st.session_state.current_view_index = base_idx + 1
                        st.session_state.board_analysis = None
                        st.session_state.just_reset = True
                        st.rerun()

    # Copy PGN / Send to Review
    st.divider()
    if st.button("📋 Copy Current PGN", key="copy_board_pgn", use_container_width=True):
        pgn_to_copy = st.session_state.get("game_pgn", "").strip()
        if pgn_to_copy:
            st.code(pgn_to_copy, language="text")
            st.caption("Use the copy icon in the top-right of the code box above.")
        else:
            st.warning("No moves yet — play some moves first.")

    if st.button("📊 Send Game to Review Tab", key="send_to_review", use_container_width=True):
        pgn_to_send = st.session_state.get("game_pgn", "").strip()
        if not pgn_to_send:
            st.warning("Play some moves first, then send the game for review.")
        else:
            st.session_state["pending_review_pgn"] = pgn_to_send
            st.success("✅ Game sent! Switch to the **📊 Review** tab and click Generate Review.")

    # Review moves
    st.divider()
    st.subheader("📊 Review Your Moves")
    st.caption("Play some moves on the board, then click below to rate each one: "
               "⭐ Best / 🌟 Excellent / ✅ Good / ⚠️ Inaccuracy / ❌ Mistake / 💥 Blunder")
    pgn_preview = st.session_state.get("game_pgn", "")
    if pgn_preview:
        st.caption(f"Moves recorded: `{pgn_preview[:120]}{'...' if len(pgn_preview) > 120 else ''}`")
    else:
        st.info("No moves yet. Drag some pieces on the board above.")

    if st.button("🔍 Review My Moves", key="review_moves", type="primary"):
        pgn_text = st.session_state.get("game_pgn", "").strip()
        if not pgn_text:
            st.warning("Make a few moves first.")
        else:
            with st.spinner("Rating your moves... (~15-30 seconds)"):
                move_results = analyze_game_moves(pgn_text, max_plies=80, depth=10)
            if "error" in move_results:
                st.error(f"Could not analyze: {move_results['error']}")
            else:
                moves = move_results["moves"]
                if not moves:
                    st.warning("No moves found in your game.")
                else:
                    from collections import Counter
                    w_counts = Counter(m["category"] for m in moves if m["color"] == "White")
                    b_counts = Counter(m["category"] for m in moves if m["color"] == "Black")
                    categories = ["Best", "Excellent", "Good", "Inaccuracy", "Mistake", "Blunder"]
                    emojis = {"Best": "⭐", "Excellent": "🌟", "Good": "✅",
                              "Inaccuracy": "⚠️", "Mistake": "❌", "Blunder": "💥"}
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**White move quality**")
                        for cat in categories:
                            st.write(f"{emojis[cat]} {cat}: **{w_counts.get(cat, 0)}**")
                    with c2:
                        st.markdown("**Black move quality**")
                        for cat in categories:
                            st.write(f"{emojis[cat]} {cat}: **{b_counts.get(cat, 0)}**")
                    st.markdown("**All moves:**")
                    df_moves = pd.DataFrame(moves)
                    st.dataframe(
                        df_moves[["move_number", "color", "san", "cpl", "label"]].rename(
                            columns={"move_number": "#", "color": "Side",
                                     "san": "Move", "cpl": "CPL", "label": "Rating"}),
                        use_container_width=True, hide_index=True)
                    key = [m for m in moves if m["category"] in ("Mistake", "Blunder")]
                    if key:
                        st.markdown("**🔴 Key mistakes and blunders:**")
                        for m in key:
                            st.write(f"- Move {m['move_number']} ({m['color']}): "
                                     f"**{m['san']}** → {m['label']} (lost **{m['cpl']}** cp)")
                    else:
                        st.success("No mistakes or blunders detected! 🎉")


# ============ TAB 4 — Pre-Match ============
with tab_prematch:
    st.markdown("Predict the outcome **before the game starts**. Shows both models "
                "side by side so you can compare the accurate (unbalanced) view with "
                "the decisive (balanced) view.")
    col_w, col_b = st.columns(2)
    with col_w:
        pm_white_elo = st.number_input("White player ELO:", 0, 3500, 2650, 10, key="pm_w_elo")
    with col_b:
        pm_black_elo = st.number_input("Black player ELO:", 0, 3500, 2650, 10, key="pm_b_elo")
    if st.button("⚔️ Predict Match Outcome", type="primary", key="pm_predict"):
        results = predict_prematch_both(int(pm_white_elo), int(pm_black_elo))
        st.divider()
        st.markdown(f"#### ELO gap: **{int(pm_white_elo) - int(pm_black_elo):+d}**")
        c1, c2 = st.columns(2)
        with c1:
            r = results["unbalanced"]
            st.markdown("### 🎯 Accurate model")
            st.caption("(unbalanced — trained on real class frequencies)")
            st.markdown(f"**{r['label']}**  ({r['confidence']:.1%})")
            st.caption(f"Source: {r['source']}")
            st.bar_chart({"White": r["p_white"], "Draw": r["p_draw"], "Black": r["p_black"]})
            if r["w_if_decisive"] is not None:
                st.markdown("**If decisive:**")
                st.write(f"White **{r['w_if_decisive']:.1%}** · Black **{r['b_if_decisive']:.1%}**")
        with c2:
            r = results["balanced"]
            st.markdown("### ⚔️ Decisive model")
            st.caption("(balanced — forces a winner, lower accuracy)")
            st.markdown(f"**{r['label']}**  ({r['confidence']:.1%})")
            st.caption(f"Source: {r['source']}")
            st.bar_chart({"White": r["p_white"], "Draw": r["p_draw"], "Black": r["p_black"]})
            if r["w_if_decisive"] is not None:
                st.markdown("**If decisive:**")
                st.write(f"White **{r['w_if_decisive']:.1%}** · Black **{r['b_if_decisive']:.1%}**")
        with st.expander("Full details"):
            for name, r in results.items():
                st.markdown(f"**{name.capitalize()}** — source: `{r['source']}`")
                st.write(f"  P(W)={r['p_white']:.3f}  P(D)={r['p_draw']:.3f}  P(B)={r['p_black']:.3f}")


# ============ TAB 5 — Review ============
with tab_review:
    st.markdown("Full game review: evaluation graph, accuracy, and move quality for both players.")
    if "pending_review_pgn" in st.session_state:
        st.session_state["review_pgn"] = st.session_state.pop("pending_review_pgn")
        st.info("📥 Game loaded from the Interactive Board — click Generate Review below.")

    review_pgn = st.text_area(
        "Paste a PGN to review:",
        height=180,
        key="review_pgn",
        placeholder="[Event \"...\"]\n[White \"...\"]\n...\n\n1. e4 e5 2. Nf3 ...")

    if st.button("📊 Generate Review", type="primary", key="review_generate"):
        if not review_pgn.strip():
            st.warning("Please paste a PGN first.")
        else:
            with st.spinner("Analyzing every move... (~20-40 seconds)"):
                result = compute_eval_graph(review_pgn, max_plies=120, depth=14)
            if "error" in result:
                st.session_state.review_data = {"error": result["error"]}
            else:
                st.session_state.review_data = result
                st.session_state.current_review_ply = 0

    graph_data = st.session_state.get("review_data")
    if graph_data and "error" not in graph_data:
        evals = graph_data["evals"]
        num_moves = graph_data["num_moves"]

        st.success(f"Analyzed {num_moves} moves.")
        st.divider()

        final_eval = evals[-1]
        max_eval = max(evals)
        min_eval = min(evals)
        c1, c2, c3 = st.columns(3)
        c1.metric("Final eval", f"{final_eval:+d} cp")
        c2.metric("Peak White", f"{max_eval:+d} cp")
        c3.metric("Peak Black", f"{min_eval:+d} cp")

        st.divider()

        opening_name, opening_eco = detect_opening(review_pgn)
        if opening_name:
            st.info(f"📖 **Opening:** {opening_name}  *(ECO {opening_eco})*")
        st.divider()

                # ---- View mode toggle ----
        view_mode = st.radio(
            "View mode:",
            ["Tabs", "Full scroll"],
            horizontal=True,
            key="review_view_mode",
        )

        if view_mode == "Tabs":
            subtab_eval, subtab_heat, subtab_quality, subtab_explain, subtab_browse = st.tabs([
                "📈 Evaluation", "🔥 Heatmaps", "🎯 Move Quality", "📝 Explanations", "🔍 Browser"])
        else:
            from contextlib import nullcontext
            subtab_eval = subtab_heat = subtab_quality = subtab_explain = subtab_browse = nullcontext()
            st.caption("📜 Full scroll mode — all sections below, scroll to explore.")
            st.divider()

        # ---- Sub-tab 1: Evaluation ----
        with subtab_eval:
            st.subheader("📈 Evaluation Graph")
            x = list(range(len(evals)))
            y = [max(-800, min(800, v)) for v in evals]
            white_adv = [max(0, v) for v in y]
            black_adv = [min(0, v) for v in y]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x, y=white_adv, mode="lines",
                line=dict(color="#4a90d9", width=1.5),
                fill="tozeroy", fillcolor="rgba(74, 144, 217, 0.6)",
                name="White advantage",
                hovertemplate="Ply %{x}<br>White +%{y} cp<extra></extra>"))
            fig.add_trace(go.Scatter(
                x=x, y=black_adv, mode="lines",
                line=dict(color="#c0392b", width=1.5),
                fill="tozeroy", fillcolor="rgba(192, 57, 43, 0.6)",
                name="Black advantage",
                hovertemplate="Ply %{x}<br>Black %{y} cp<extra></extra>"))
            fig.add_hline(y=0, line_dash="dash", line_color="#888", line_width=1)
            fig.update_layout(
                title="Position evaluation — White vs Black",
                xaxis_title="Ply (half-move)",
                yaxis_title="Eval (cp, White POV)",
                yaxis=dict(range=[-850, 850]),
                height=420, showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=20, t=60, b=40),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("🔵 Blue = White advantage · 🔴 Red = Black advantage.")

        # ---- Sub-tab 2: Heatmaps ----
        with subtab_heat:
            # Material Balance
            st.subheader("⚖️ Material Balance")
            fens = graph_data.get("fens", [])
            material = [material_balance_from_fen(f) for f in fens]
            x_mat = list(range(len(material)))
            white_mat = [max(0, m) for m in material]
            black_mat = [min(0, m) for m in material]
            fig_mat = go.Figure()
            fig_mat.add_trace(go.Scatter(
                x=x_mat, y=white_mat, mode="lines",
                line=dict(color="#4a90d9", width=1.5),
                fill="tozeroy", fillcolor="rgba(74, 144, 217, 0.55)",
                name="White material",
                hovertemplate="Ply %{x}<br>White +%{y} pawns<extra></extra>"))
            fig_mat.add_trace(go.Scatter(
                x=x_mat, y=black_mat, mode="lines",
                line=dict(color="#c0392b", width=1.5),
                fill="tozeroy", fillcolor="rgba(192, 57, 43, 0.55)",
                name="Black material",
                hovertemplate="Ply %{x}<br>Black %{y} pawns<extra></extra>"))
            fig_mat.add_hline(y=0, line_dash="dash", line_color="#888", line_width=1)
            fig_mat.update_layout(
                title="Material balance over the game",
                xaxis_title="Ply (half-move)", yaxis_title="Material advantage (pawns)",
                height=320, showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=20, t=60, b=40),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_mat, use_container_width=True)
            st.caption("🔵 Blue = White material · 🔴 Red = Black material.")
            st.divider()

            # Piece Activity Heatmap
            st.subheader("🔥 Piece Activity Heatmap")
            if fens:
                final_fen = fens[-1]
                grid = piece_activity_heatmap(final_fen)
                row_labels = ["8", "7", "6", "5", "4", "3", "2", "1"]
                col_labels = ["a", "b", "c", "d", "e", "f", "g", "h"]
                fig_heat = go.Figure(data=go.Heatmap(
                    z=grid, x=col_labels, y=row_labels,
                    colorscale=[[0.0, "#c0392b"], [0.5, "#1a1a1a"], [1.0, "#4a90d9"]],
                    zmid=0,
                    hovertemplate="Square %{x}%{y}<br>Control: %{z:+d}<extra></extra>",
                    showscale=True, colorbar=dict(title="White − Black")))
                fig_heat.update_layout(
                    title="Who controls each square (final position)",
                    height=420, yaxis=dict(autorange="reversed"), xaxis=dict(side="top"),
                    margin=dict(l=40, r=40, t=60, b=40),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_heat, use_container_width=True)
                st.caption("🔵 Blue = White controls · 🔴 Red = Black controls · ⚫ Dark = balanced.")
            st.divider()

            # Piece Square Heatmap
            st.subheader("♟️ Piece Square Heatmap")
            piece_choice = st.selectbox(
                "Select piece type:",
                ["All pieces", "Knights", "Bishops", "Rooks", "Queens", "Pawns", "Kings"],
                key="piece_heatmap_choice")
            piece_map = {"All pieces": None, "Knights": chess.KNIGHT, "Bishops": chess.BISHOP,
                         "Rooks": chess.ROOK, "Queens": chess.QUEEN, "Pawns": chess.PAWN, "Kings": chess.KING}
            w_grid, b_grid = piece_square_heatmap(review_pgn, piece_type=piece_map[piece_choice])
            if w_grid and b_grid:
                row_labels = ["8", "7", "6", "5", "4", "3", "2", "1"]
                col_labels = ["a", "b", "c", "d", "e", "f", "g", "h"]
                white_scale = [[0.0, "#1a1a1a"], [1.0, "#4a90d9"]]
                black_scale = [[0.0, "#1a1a1a"], [1.0, "#c0392b"]]
                col_w, col_b = st.columns(2)
                with col_w:
                    fig_w = go.Figure(data=go.Heatmap(
                        z=w_grid, x=col_labels, y=row_labels, colorscale=white_scale,
                        hovertemplate="Square %{x}%{y}<br>White visits: %{z}<extra></extra>",
                        showscale=True))
                    fig_w.update_layout(
                        title="White's pieces", height=380,
                        yaxis=dict(autorange="reversed"), xaxis=dict(side="top"),
                        margin=dict(l=30, r=30, t=60, b=30),
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_w, use_container_width=True)
                with col_b:
                    fig_b = go.Figure(data=go.Heatmap(
                        z=b_grid, x=col_labels, y=row_labels, colorscale=black_scale,
                        hovertemplate="Square %{x}%{y}<br>Black visits: %{z}<extra></extra>",
                        showscale=True))
                    fig_b.update_layout(
                        title="Black's pieces", height=380,
                        yaxis=dict(autorange="reversed"), xaxis=dict(side="top"),
                        margin=dict(l=30, r=30, t=60, b=30),
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_b, use_container_width=True)
                st.caption("Brighter square = visited more often.")
            st.divider()

            # King Safety Timeline
            st.subheader("👑 King Safety Timeline")
            if fens:
                w_safety, b_safety = king_safety_timeline(fens)
                x_ks = list(range(len(w_safety)))
                fig_ks = go.Figure()
                fig_ks.add_trace(go.Scatter(
                    x=x_ks, y=w_safety, mode="lines",
                    line=dict(color="#4a90d9", width=2), name="White king",
                    hovertemplate="Ply %{x}<br>White safety: %{y}<extra></extra>"))
                fig_ks.add_trace(go.Scatter(
                    x=x_ks, y=b_safety, mode="lines",
                    line=dict(color="#c0392b", width=2), name="Black king",
                    hovertemplate="Ply %{x}<br>Black safety: %{y}<extra></extra>"))
                fig_ks.update_layout(
                    title="King safety over the game (higher = safer)",
                    xaxis_title="Ply (half-move)", yaxis_title="Safety score",
                    height=340, showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=40, r=20, t=60, b=40),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_ks, use_container_width=True)
                st.caption("🔵 Blue = White king · 🔴 Red = Black king.")

        # ---- Sub-tab 3: Move Quality ----
        with subtab_quality:
            # Move Quality Distribution
            st.subheader("🎯 Move Quality Distribution")
            categories = graph_data.get("categories", [])
            sans = graph_data.get("sans", [])
            white_cats = []
            black_cats = []
            for i in range(1, len(categories)):
                if i % 2 == 1:
                    white_cats.append(categories[i])
                else:
                    black_cats.append(categories[i])
            cat_list = ["Best", "Excellent", "Good", "Inaccuracy", "Mistake", "Blunder"]
            emoji_map = {"Best": "⭐", "Excellent": "🌟", "Good": "✅",
                         "Inaccuracy": "⚠️", "Mistake": "❌", "Blunder": "💥"}
            white_counts = [white_cats.count(c) for c in cat_list]
            black_counts = [black_cats.count(c) for c in cat_list]
            labels = [f"{emoji_map[c]} {c}" for c in cat_list]
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                name="White", y=labels, x=white_counts, orientation="h",
                marker=dict(color="#4a90d9"),
                hovertemplate="White %{y}: %{x}<extra></extra>"))
            fig2.add_trace(go.Bar(
                name="Black", y=labels, x=[-v for v in black_counts], orientation="h",
                marker=dict(color="#c0392b"),
                hovertemplate="Black %{y}: %{customdata}<extra></extra>",
                customdata=black_counts))
            max_count = max(white_counts + black_counts + [1])
            fig2.update_layout(
                barmode="overlay", height=380,
                xaxis=dict(title="Number of moves", range=[-max_count - 1, max_count + 1],
                           tickvals=[-max_count, 0, max_count],
                           ticktext=[str(max_count), "0", str(max_count)]),
                yaxis=dict(title=""),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=20, t=40, b=40),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)
            st.divider()

            # Accuracy
            st.subheader("🎯 Accuracy per Player")
            cpls = graph_data.get("cpls", [])
            white_cpls = []
            black_cpls = []
            for i in range(1, len(cpls)):
                if i % 2 == 1:
                    white_cpls.append(cpls[i])
                else:
                    black_cpls.append(cpls[i])
            def compute_accuracy(cpl_list):
                if not cpl_list:
                    return 0.0
                avg_cpl = sum(cpl_list) / len(cpl_list)
                return max(0.0, min(100.0, 100.0 - avg_cpl / 5.0))
            white_acc = compute_accuracy(white_cpls)
            black_acc = compute_accuracy(black_cpls)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("⚪ White accuracy", f"{white_acc:.1f}%")
                st.caption(f"Avg CPL: {sum(white_cpls)/len(white_cpls):.0f}" if white_cpls else "No moves")
            with col2:
                st.metric("⚫ Black accuracy", f"{black_acc:.1f}%")
                st.caption(f"Avg CPL: {sum(black_cpls)/len(black_cpls):.0f}" if black_cpls else "No moves")
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(
                x=["White", "Black"], y=[white_acc, black_acc],
                marker=dict(color=["#4a90d9", "#c0392b"]),
                text=[f"{white_acc:.1f}%", f"{black_acc:.1f}%"],
                textposition="outside",
                hovertemplate="%{x}: %{y:.1f}%<extra></extra>"))
            fig3.update_layout(
                height=300, yaxis=dict(title="Accuracy %", range=[0, 105]),
                showlegend=False, margin=dict(l=40, r=20, t=20, b=40),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig3, use_container_width=True)
            st.divider()

            # CPL Timeline
            st.subheader("📊 CPL Timeline (centipawn loss per move)")
            cpls = graph_data.get("cpls", [])
            sans = graph_data.get("sans", [])
            if cpls and len(cpls) > 1:
                white_x, white_y, white_colors, white_text = [], [], [], []
                black_x, black_y, black_colors, black_text = [], [], [], []
                def quality_color(cpl):
                    if cpl <= 20: return "#27ae60"
                    elif cpl <= 50: return "#3498db"
                    elif cpl <= 100: return "#f39c12"
                    elif cpl <= 300: return "#e67e22"
                    else: return "#c0392b"
                for i, cpl in enumerate(cpls[1:]):
                    ply_index = i + 1
                    c = quality_color(cpl)
                    if ply_index < len(sans):
                        hover = f"Move {ply_index}: {sans[ply_index]}<br>CPL: {cpl}"
                    else:
                        hover = f"Ply {ply_index}<br>CPL: {cpl}"
                    if ply_index % 2 == 1:
                        white_x.append(ply_index); white_y.append(cpl)
                        white_colors.append(c); white_text.append(hover)
                    else:
                        black_x.append(ply_index); black_y.append(cpl)
                        black_colors.append(c); black_text.append(hover)
                fig4 = go.Figure()
                fig4.add_trace(go.Bar(
                    x=white_x, y=white_y, name="White",
                    marker=dict(color=white_colors, line=dict(color="#4a90d9", width=4)),
                    text=white_text, hovertemplate="%{text}<extra></extra>"))
                fig4.add_trace(go.Bar(
                    x=black_x, y=black_y, name="Black",
                    marker=dict(color=black_colors, line=dict(color="#c0392b", width=4),
                                pattern=dict(shape="/", fgcolor="rgba(0, 0, 0, 0.35)",
                                             fillmode="overlay", solidity=0.5)),
                    text=black_text, hovertemplate="%{text}<extra></extra>"))
                fig4.update_layout(
                    height=340,
                    xaxis=dict(title="Ply (half-move)"),
                    yaxis=dict(title="CPL (centipawns lost)"),
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="right", x=1),
                    margin=dict(l=40, r=20, t=60, b=40),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    updatemenus=[dict(
                        type="buttons", direction="right", x=0.0, y=1.14, showactive=True,
                        buttons=[
                            dict(label="All", method="update", args=[{"visible": [True, True]}]),
                            dict(label="White only", method="update", args=[{"visible": [True, False]}]),
                            dict(label="Black only", method="update", args=[{"visible": [False, True]}]),
                        ],
                        bgcolor="#333", bordercolor="#666", font=dict(color="#eee"))])
                st.plotly_chart(fig4, use_container_width=True)
                st.caption("🔵 Blue border = White · 🔴 Red border = Black.")
            st.divider()

            # Mistake Density by Phase
            st.subheader("🎯 Mistake Density by Game Phase")
            categories = graph_data.get("categories", [])
            opening_end = 30
            middlegame_end = 80
            phases = {"Opening (1-15)": [0, 0], "Middlegame (16-40)": [0, 0], "Endgame (40+)": [0, 0]}
            for i, cat in enumerate(categories):
                if i == 0: continue
                is_bad = cat in ("Inaccuracy", "Mistake", "Blunder")
                if not is_bad: continue
                if i <= opening_end: phase = "Opening (1-15)"
                elif i <= middlegame_end: phase = "Middlegame (16-40)"
                else: phase = "Endgame (40+)"
                if i % 2 == 1: phases[phase][1] += 1
                else: phases[phase][0] += 1
            phase_names = list(phases.keys())
            black_mistakes = [phases[p][0] for p in phase_names]
            white_mistakes = [phases[p][1] for p in phase_names]
            fig5 = go.Figure()
            fig5.add_trace(go.Bar(name="Black", x=phase_names, y=black_mistakes,
                                  marker=dict(color="#c0392b"), text=black_mistakes, textposition="auto"))
            fig5.add_trace(go.Bar(name="White", x=phase_names, y=white_mistakes,
                                  marker=dict(color="#4a90d9"), text=white_mistakes, textposition="auto"))
            fig5.update_layout(
                barmode="group", height=320,
                yaxis=dict(title="Mistakes + Inaccuracies + Blunders"),
                xaxis=dict(title="Game phase"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=20, t=40, b=40),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig5, use_container_width=True)

        # ---- Sub-tab 4: Explanations ----
        with subtab_explain:
            st.subheader("📝 Move Explanations")
            best_alts = graph_data.get("best_alts", [])
            cats = graph_data.get("categories", [])
            cpls_all = graph_data.get("cpls", [])
            sans_all = graph_data.get("sans", [])
            fens_all = graph_data.get("fens", [])
            highlights = []
            for i in range(1, len(sans_all)):
                if cats[i] in ("Mistake", "Blunder"):
                    highlights.append({
                        "ply": i,
                        "move_number": (i + 1) // 2,
                        "color": "White" if i % 2 == 1 else "Black",
                        "san": sans_all[i],
                        "category": cats[i],
                        "cpl": cpls_all[i],
                        "best_alt": best_alts[i] if i < len(best_alts) else "",
                        "fen_before": fens_all[i - 1] if i - 1 < len(fens_all) else "",
                    })
            if not highlights:
                st.success("No mistakes or blunders in this game! Excellent play.")
            else:
                for h in highlights:
                    explanation = explain_move(h, h["fen_before"])
                    emoji = "❌" if h["category"] == "Mistake" else "💥"
                    st.markdown(f"**Move {h['move_number']} ({h['color']}): {h['san']}** "
                                f"— {emoji} {h['category']} (lost **{h['cpl']}** cp)")
                    st.caption(f"→ {explanation}")

        # ---- Sub-tab 5: Browser ----
        with subtab_browse:
            st.subheader("🔍 Position Browser")
            st.caption("Drag the slider to browse any position in the game.")
            fens = graph_data.get("fens", [])
            sans = graph_data.get("sans", [])
            cats = graph_data.get("categories", [])
            cpls_all = graph_data.get("cpls", [])
            best_alts = graph_data.get("best_alts", [])
            sans_all = graph_data.get("sans", [])
            fens_all = graph_data.get("fens", [])
            if len(fens) <= 1:
                st.info("No moves to browse — the game appears to be empty or could not be parsed.")
            else:
                if "current_review_ply" not in st.session_state:
                    st.session_state.current_review_ply = 0
                selected_ply = st.slider(
                    "Select ply (half-move)",
                    min_value=0, max_value=len(fens) - 1,
                    key="current_review_ply")
                col_board, col_info = st.columns([1, 1])
                with col_board:
                    if selected_ply == 0:
                        label = "Starting position"
                    else:
                        move_num = (selected_ply + 1) // 2
                        side = "White" if selected_ply % 2 == 1 else "Black"
                        label = f"Move {move_num} ({side}): {sans[selected_ply]}"
                    st.markdown(f"**{label}**")
                    try:
                        board_obj = chess.Board(fens[selected_ply])
                        svg_str = chess.svg.board(board_obj, size=380)
                        svg_b64 = base64.b64encode(svg_str.encode("utf-8")).decode("utf-8")
                        st.markdown(f'<img src="data:image/svg+xml;base64,{svg_b64}" width="380" />',
                                    unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Could not render board: {e}")
                with col_info:
                    if selected_ply > 0:
                        this_cat = cats[selected_ply] if selected_ply < len(cats) else "?"
                        this_cpl = cpls_all[selected_ply] if selected_ply < len(cpls_all) else 0
                        this_best = best_alts[selected_ply] if selected_ply < len(best_alts) else ""
                        emoji_map = {"Best": "⭐", "Excellent": "🌟", "Good": "✅",
                                     "Inaccuracy": "⚠️", "Mistake": "❌", "Blunder": "💥"}
                        em = emoji_map.get(this_cat, "")
                        st.markdown(f"**Move played:** {sans_all[selected_ply]} — {em} {this_cat} (lost {this_cpl} cp)")
                        move_info = {"san": sans_all[selected_ply],
                                     "color": "White" if selected_ply % 2 == 1 else "Black",
                                     "category": this_cat, "cpl": this_cpl, "best_alt": this_best}
                        fen_before = fens_all[selected_ply - 1] if selected_ply - 1 < len(fens_all) else ""
                        try:
                            explanation = explain_move(move_info, fen_before)
                            st.caption(f"→ {explanation}")
                        except Exception:
                            pass
                        st.divider()
                    st.markdown("**FEN:**")
                    st.code(fens[selected_ply], language="text")
                    current_eval = evals[selected_ply]
                    if current_eval >= 200: color = "🟢"
                    elif current_eval <= -200: color = "🔴"
                    elif abs(current_eval) < 50: color = "⚪"
                    else: color = "🟡"
                    st.metric("Eval at this position", f"{color} {current_eval:+d} cp")
                    if st.button("🎯 Analyze this position", key=f"review_analyze_{selected_ply}"):
                        with st.spinner("Analyzing position..."):
                            inline_result = predict_from_fen(fens[selected_ply], 2615, 2610)
                        render_analysis(inline_result)

    if graph_data and "error" in graph_data:
        st.error(graph_data["error"])


# ---------- Sidebar ----------
st.sidebar.markdown("### About")
st.sidebar.markdown(
    "Built with **python-chess**, **Stockfish 19**, and **scikit-learn**. "
    "Trained on 5000 master-level games (2600+ ELO). Test accuracy: ~89%.")