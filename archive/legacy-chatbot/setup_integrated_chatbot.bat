@echo off
REM Integrated Hemophilia Chatbot Setup Script for Windows
REM This script sets up both Express backend and Streamlit frontend

setlocal enabledelayedexpansion
set ScriptPath=%~dp0

echo.
echo ================================================================================
echo    Integrated Hemophilia Chatbot Setup for Windows
echo ================================================================================
echo.

REM Check Python
echo [1/5] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.11+ from https://python.org
    pause
    exit /b 1
) else (
    for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
    echo ✅ Python !PYTHON_VERSION! found
)

REM Check Node.js
echo.
echo [2/5] Checking Node.js...
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js not found. Please install Node.js 16+ from https://nodejs.org
    pause
    exit /b 1
) else (
    for /f %%i in ('node --version') do set NODE_VERSION=%%i
    echo ✅ Node.js !NODE_VERSION! found
)

REM Setup Python virtual environment
echo.
echo [3/5] Setting up Python environment...
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
) else (
    echo ✅ Virtual environment already exists
)

call .venv\Scripts\activate.bat
echo ✅ Python environment activated

REM Install Python dependencies
echo Updating Python dependencies...
pip install -r requirements.txt >nul 2>&1
pip install requests >nul 2>&1
echo ✅ Python dependencies installed

REM Setup Express backend
echo.
echo [4/5] Setting up Express backend...
cd /d "%ScriptPath%chatbot-service"

if not exist node_modules (
    echo Installing Node.js dependencies...
    call npm install >nul 2>&1
    if errorlevel 1 (
        echo ❌ Failed to install Node.js dependencies
        pause
        exit /b 1
    )
    echo ✅ Node.js dependencies installed
) else (
    echo ✅ Node.js modules already installed
)

if not exist .env (
    echo Creating .env configuration...
    (
        echo PORT=5001
        echo NODE_ENV=development
        echo MONGODB_URI=mongodb://localhost:27017/hemophilia-chatbot
        echo CORS_ORIGIN=http://localhost:8501
    ) > .env
    echo ✅ .env file created (edit if using MongoDB Atlas)
) else (
    echo ✅ .env already exists
)

cd /d "%ScriptPath%"

REM Check if chatbot_service_client exists
echo.
echo [5/5] Verifying integration files...
if not exist chatbot_service_client.py (
    echo ❌ Error: chatbot_service_client.py not found
    pause
    exit /b 1
) else (
    echo ✅ chatbot_service_client.py verified
)

if not exist app.py (
    echo ❌ Error: app.py not found
    pause
    exit /b 1
) else (
    echo ✅ app.py verified
)

REM Setup complete
echo.
echo ================================================================================
echo    ✅ Setup Complete!
echo ================================================================================
echo.
echo Next Steps:
echo.
echo 1. Ensure MongoDB is Running:
echo    - If Local: Open another terminal and run: mongod
echo    - If Cloud: Ensure MongoDB Atlas cluster is running
echo    - Edit chatbot-service\.env if using MongoDB Atlas
echo.
echo 2. Start Backend Service (NEW TERMINAL):
echo    cd chatbot-service
echo    npm start
echo    (Should show: "✅ MongoDB connected" and "🚀 Chatbot Service started on port 5001")
echo.
echo 3. Start Streamlit Frontend (ANOTHER NEW TERMINAL):
echo    call .venv\Scripts\activate.bat
echo    streamlit run app.py
echo    (Should open http://localhost:8501)
echo.
echo 4. Open Streamlit in Browser:
echo    Go to http://localhost:8501
echo    Navigate to "💬 Quick Chat" page
echo    Should see "✅ Connected to chatbot service"
echo.
echo For Troubleshooting, see: INTEGRATED_CHATBOT_SETUP.md
echo.
echo ================================================================================
echo.
pause
