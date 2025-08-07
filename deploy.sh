#!/bin/bash

# Vercel Deployment Script for Groq Hyper-Intelligence API
# Run this script to deploy your API to Vercel

echo "🚀 Groq Hyper-Intelligence API - Vercel Deployment"
echo "=================================================="

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI not found. Installing..."
    npm install -g vercel
fi

echo "✅ Vercel CLI is available"

# Login to Vercel (if not already logged in)
echo "🔐 Checking Vercel authentication..."
vercel whoami || vercel login

echo "📁 Current directory: $(pwd)"
echo "📦 Starting deployment..."

# Deploy to Vercel
vercel --prod

echo ""
echo "🎉 Deployment completed!"
echo ""
echo "📋 Next steps:"
echo "1. Set up environment variables in Vercel dashboard:"
echo "   - GROQ_API_KEY"
echo "   - MONGODB_URI" 
echo "   - HACKRX_API_TOKEN"
echo ""
echo "2. Test your deployment:"
echo "   - Health check: https://your-project.vercel.app/health"
echo "   - API endpoint: https://your-project.vercel.app/hackrx/run"
echo ""
echo "📖 See VERCEL_DEPLOYMENT.md for detailed instructions"
