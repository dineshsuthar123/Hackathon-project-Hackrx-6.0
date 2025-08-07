@echo off
REM Vercel Deployment Script for Windows
REM Groq Hyper-Intelligence API

echo 🚀 Groq Hyper-Intelligence API - Vercel Deployment
echo ==================================================

REM Check if Vercel CLI is installed
where vercel >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Vercel CLI not found. Installing...
    npm install -g vercel
)

echo ✅ Vercel CLI is available

REM Login to Vercel (if not already logged in)
echo 🔐 Checking Vercel authentication...
vercel whoami || vercel login

echo 📁 Current directory: %CD%
echo 📦 Starting deployment...

REM Deploy to Vercel
vercel --prod

echo.
echo 🎉 Deployment completed!
echo.
echo 📋 Next steps:
echo 1. Set up environment variables in Vercel dashboard:
echo    - GROQ_API_KEY
echo    - MONGODB_URI
echo    - HACKRX_API_TOKEN
echo.
echo 2. Test your deployment:
echo    - Health check: https://your-project.vercel.app/health
echo    - API endpoint: https://your-project.vercel.app/hackrx/run
echo.
echo 📖 See VERCEL_DEPLOYMENT.md for detailed instructions

pause
