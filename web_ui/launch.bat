@echo off
cd /d "%~dp0"
echo Starting Deep-TimeSeries Dashboard...
echo.
start "Deep-TimeSeries Dashboard" py -m streamlit run app.py
echo.
echo Dashboard is starting in a new window...
echo Check http://localhost:8501 in your browser
echo.
pause
