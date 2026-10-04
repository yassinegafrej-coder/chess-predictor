import streamlit as st
from stchess import board, DEFAULT_FEN

st.set_page_config(page_title="Chess Board Test", layout="wide")
st.title("Chess Board Component Test")

# Render the chess board
fen = board(fen=DEFAULT_FEN, key="test_board")

# Show the current FEN below the board
st.write("Current FEN:")
st.code(fen)