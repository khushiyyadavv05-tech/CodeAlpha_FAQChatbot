@echo off
title CodeAlpha FAQ Chatbot
echo Installing requirements (first time only, please wait)...
pip install -r requirements.txt
echo.
echo Starting chatbot... your browser will open in a few seconds.
start "" http://127.0.0.1:5000
python app.py
pause
