import chess
import chess.engine

STOCKFISH_PATH = r"E:\chess-predictor\stockfish\stockfish-windows-x86-64-universal.exe"

engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
print("Engine name:", engine.id["name"])

board = chess.Board()
info = engine.analyse(board, chess.engine.Limit(depth=12))
score = info["score"].white()
print("Eval (centipawns):", score.score())
print("Best move:", info["pv"][0])

engine.quit()
print("Engine closed cleanly.")