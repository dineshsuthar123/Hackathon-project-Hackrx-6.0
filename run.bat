@echo off
REM Windows batch script to run the LLM-Powered Query-Retrieval System

echo 🎯 LLM-Powered Intelligent Query-Retrieval System
echo 🏆 Hack 6.0 Hackathon Submission
echo ================================================

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if .env file exists
if not exist .env (
    echo ⚠️  .env file not found
    echo Creating .env file template...
    echo OPENAI_API_KEY=your_openai_api_key_here > .env
    echo HACKRX_API_TOKEN=a3d1b4849a33b0269ac53fd27a8552eb1fbcc9cea01c70a1a85e11e330eb7c36 >> .env
    echo.
    echo 📝 Please edit .env file and add your OpenAI API key
    echo Then run this script again
    pause
    exit /b 1
)

REM Install dependencies if needed
if not exist venv (
    echo 📦 Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo 📦 Installing dependencies...
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

REM Run the system
echo 🚀 Starting the system...
python start.py

pause
