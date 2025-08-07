#!/bin/bash
# Render startup script for Groq ReAct Intelligence API

echo "🚀 Starting Groq ReAct Intelligence API - Protocol 7.1/7.2"
echo "🛡️ Contextual Guardrail + Generalized RAG Active"
echo "================================================="

# Set default port if not provided
export PORT=${PORT:-8000}

echo "📡 Starting server on port $PORT"

# Start the FastAPI application with uvicorn
uvicorn app_groq_ultimate:app --host 0.0.0.0 --port $PORT --workers 1
