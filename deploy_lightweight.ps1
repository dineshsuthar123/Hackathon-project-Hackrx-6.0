# PowerShell deployment script for Lightweight Intelligent Document Reader

Write-Host "🚀 Deploying Lightweight Intelligent Document Reader to Render..." -ForegroundColor Cyan

# Check if git is available
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Git is not installed. Please install git first." -ForegroundColor Red
    exit 1
}

# Check if we're in a git repository
try {
    git rev-parse --git-dir *>$null 2>&1
} catch {
    Write-Host "⚠️  Not in a git repository. Initializing..." -ForegroundColor Yellow
    git init
    git remote add origin https://github.com/dineshsuthar123/Hackathon-project-Hackrx-6.0.git
}

Write-Host "📝 Preparing lightweight deployment files..." -ForegroundColor Blue

# Copy the lightweight reader as the main app
Copy-Item "lightweight_intelligent_reader.py" "app.py" -Force

# Use the lightweight requirements
Copy-Item "requirements_lightweight.txt" "requirements.txt" -Force

# Create/update render.yaml for lightweight deployment
$renderYaml = @"
services:
  - type: web
    name: hackrx-lightweight-document-reader
    env: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: python app.py
    envVars:
      - key: HACKRX_API_TOKEN
        value: a3d1b4849a33b0269ac53fd27a8552eb1fbcc9cea01c70a1a85e11e330eb7c36
      - key: PYTHON_VERSION
        value: 3.9.18
    healthCheckPath: /
    numInstances: 1
    region: oregon
    disk:
      name: data
      mountPath: /opt/render/project/src/data
      sizeGB: 1
"@

$renderYaml | Out-File -FilePath "render.yaml" -Encoding UTF8

Write-Host "✅ Deployment files prepared" -ForegroundColor Green

# Git operations
Write-Host "📚 Committing changes..." -ForegroundColor Blue
git add .
git status

# Check if there are changes to commit
$gitDiff = git diff --staged --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "⚠️  No changes to commit" -ForegroundColor Yellow
} else {
    $commitMessage = @"
Deploy lightweight intelligent document reader v4.0.0

Features:
- Enhanced pattern matching with comprehensive insurance terms
- Intelligent keyword extraction and content analysis  
- Structured document parsing with section identification
- Smart answer extraction using multiple strategies
- Optimized for reliable deployment (no heavy ML dependencies)
- Fast startup and minimal resource usage
- Comprehensive fallback content for insurance policy analysis

Improvements over previous version:
- Removed heavy ML dependencies (sentence-transformers, sklearn)
- Enhanced pattern matching library with 20+ insurance-specific patterns
- Better keyword-based content search and matching
- More reliable document processing and caching
- Optimized for Render free tier deployment

Ready for production deployment on Render.
"@
    
    git commit -m $commitMessage
    Write-Host "✅ Changes committed" -ForegroundColor Green
}

# Push to repository
Write-Host "🔄 Pushing to repository..." -ForegroundColor Blue
git push origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Successfully pushed to repository" -ForegroundColor Green
    Write-Host ""
    Write-Host "🎉 DEPLOYMENT READY!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Blue
    Write-Host "1. Go to https://render.com/dashboard"
    Write-Host "2. Click 'New +' → 'Web Service'"
    Write-Host "3. Connect your GitHub repository: dineshsuthar123/Hackathon-project-Hackrx-6.0"
    Write-Host "4. Use these settings:"
    Write-Host "   - Name: hackrx-lightweight-document-reader"
    Write-Host "   - Environment: Python 3"
    Write-Host "   - Build Command: pip install -r requirements.txt"
    Write-Host "   - Start Command: python app.py"
    Write-Host "   - Plan: Free"
    Write-Host ""
    Write-Host "📋 Environment Variables to set:" -ForegroundColor Yellow
    Write-Host "HACKRX_API_TOKEN = a3d1b4849a33b0269ac53fd27a8552eb1fbcc9cea01c70a1a85e11e330eb7c36"
    Write-Host ""
    Write-Host "🔗 Your API will be available at: https://your-service-name.onrender.com" -ForegroundColor Green
    Write-Host "📊 Endpoints:" -ForegroundColor Blue
    Write-Host "  GET  /hackrx/run?documents=URL&questions=question1,question2"
    Write-Host "  POST /hackrx/run (JSON body with documents and questions)"
    Write-Host ""
    Write-Host "✨ This lightweight version is optimized for reliable deployment!" -ForegroundColor Green
} else {
    Write-Host "❌ Failed to push to repository" -ForegroundColor Red
    exit 1
}
