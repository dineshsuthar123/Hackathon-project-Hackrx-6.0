@echo off
echo 🔧 Vercel Debug Deployment for Groq API
echo =======================================

REM Step 1: Check current directory and files
echo 📁 Current directory: %CD%
echo 📄 Checking key files...

if exist "main.py" (
    echo ✅ main.py exists
) else (
    echo ❌ main.py missing
)

if exist "app_groq_ultimate.py" (
    echo ✅ app_groq_ultimate.py exists
) else (
    echo ❌ app_groq_ultimate.py missing
)

if exist "vercel.json" (
    echo ✅ vercel.json exists
) else (
    echo ❌ vercel.json missing
)

if exist "requirements.txt" (
    echo ✅ requirements.txt exists
) else (
    echo ❌ requirements.txt missing
)

echo.
echo 🚀 Starting Vercel deployment...
echo.

REM Deploy with verbose output
vercel --prod --debug

echo.
echo 🔍 Deployment completed. Check the logs above for any errors.
echo.
echo 📋 Debug tips:
echo 1. Check the build logs for import errors
echo 2. Verify environment variables are set
echo 3. Test the /health endpoint first
echo 4. Check function logs: vercel logs
echo.

pause
