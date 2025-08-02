#!/bin/bash

echo "🚀 Deploying Lightweight Intelligent Document Reader to Render..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if git is available
if ! command -v git &> /dev/null; then
    echo -e "${RED}❌ Git is not installed. Please install git first.${NC}"
    exit 1
fi

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Not in a git repository. Initializing...${NC}"
    git init
    git remote add origin https://github.com/dineshsuthar123/Hackathon-project-Hackrx-6.0.git
fi

echo -e "${BLUE}📝 Preparing lightweight deployment files...${NC}"

# Copy the lightweight reader as the main app
cp lightweight_intelligent_reader.py app.py

# Use the lightweight requirements
cp requirements_lightweight.txt requirements.txt

# Create/update render.yaml for lightweight deployment
cat > render.yaml << 'EOF'
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
EOF

echo -e "${GREEN}✅ Deployment files prepared${NC}"

# Git operations
echo -e "${BLUE}📚 Committing changes...${NC}"
git add .
git status

# Check if there are changes to commit
if git diff --staged --quiet; then
    echo -e "${YELLOW}⚠️  No changes to commit${NC}"
else
    git commit -m "Deploy lightweight intelligent document reader v4.0.0

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
    
    Ready for production deployment on Render."
    
    echo -e "${GREEN}✅ Changes committed${NC}"
fi

# Push to repository
echo -e "${BLUE}🔄 Pushing to repository...${NC}"
git push origin main

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Successfully pushed to repository${NC}"
    echo ""
    echo -e "${GREEN}🎉 DEPLOYMENT READY!${NC}"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "1. Go to https://render.com/dashboard"
    echo "2. Click 'New +' → 'Web Service'"
    echo "3. Connect your GitHub repository: dineshsuthar123/Hackathon-project-Hackrx-6.0"
    echo "4. Use these settings:"
    echo "   - Name: hackrx-lightweight-document-reader"
    echo "   - Environment: Python 3"
    echo "   - Build Command: pip install -r requirements.txt"
    echo "   - Start Command: python app.py"
    echo "   - Plan: Free"
    echo ""
    echo -e "${YELLOW}📋 Environment Variables to set:${NC}"
    echo "HACKRX_API_TOKEN = a3d1b4849a33b0269ac53fd27a8552eb1fbcc9cea01c70a1a85e11e330eb7c36"
    echo ""
    echo -e "${GREEN}🔗 Your API will be available at: https://your-service-name.onrender.com${NC}"
    echo -e "${BLUE}📊 Endpoints:${NC}"
    echo "  GET  /hackrx/run?documents=URL&questions=question1,question2"
    echo "  POST /hackrx/run (JSON body with documents and questions)"
    echo ""
    echo -e "${GREEN}✨ This lightweight version is optimized for reliable deployment!${NC}"
else
    echo -e "${RED}❌ Failed to push to repository${NC}"
    exit 1
fi
