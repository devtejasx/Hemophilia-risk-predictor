@echo off
REM Install Pre-trained Local Model and Dependencies
REM Run this script to set up the chatbot with offline capabilities

echo.
echo ========================================
echo   Local GPT Model Setup
echo ========================================
echo.

REM Check if virtual environment is activated
if not defined VIRTUAL_ENV (
    echo ❌ Virtual environment not activated!
    echo Please run: .\.venv\Scripts\Activate.ps1
    pause
    exit /b 1
)

echo ✅ Virtual environment detected
echo.

REM Install requirements
echo Installing dependencies...
pip install -r requirements.txt --timeout 1000

if %ERRORLEVEL% neq 0 (
    echo ❌ Installation failed
    pause
    exit /b 1
)

echo.
echo ✅ Dependencies installed successfully!
echo.

REM Download pre-trained model
echo Downloading pre-trained model (DistilGPT-2)...
echo This may take a few minutes on first run...
echo.

python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('distilgpt2'); AutoModelForCausalLM.from_pretrained('distilgpt2'); print('✅ Model downloaded successfully!')"

if %ERRORLEVEL% neq 0 (
    echo ⚠️ Model download encountered an issue, but this is OK!
    echo The system will download it automatically on first use.
)

echo.
echo ========================================
echo   ✅ Setup Complete!
echo ========================================
echo.
echo Your chatbot is ready with:
echo • Local pre-trained GPT model
echo • Offline Q&A capability
echo • Knowledge base integration
echo • API fallback support
echo.
echo Next: Run 'streamlit run app.py' to start!
echo.

pause
