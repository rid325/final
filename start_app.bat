@echo off
echo Starting Backend...
start "Backend" cmd /k "python app.py"

echo Starting Frontend...
start "Frontend" cmd /k "npm run dev"

echo Both services started!
