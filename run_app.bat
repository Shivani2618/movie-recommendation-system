@echo off
title CineMatch AI - Movie Recommender System
echo Starting CineMatch AI...
if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe run.py
) else (
    python run.py
)
pause

