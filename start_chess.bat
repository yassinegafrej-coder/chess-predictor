@echo off
title Chess Predictor
cd /d E:\chess-predictor
call venv\Scripts\activate.bat
streamlit run app.py
pause