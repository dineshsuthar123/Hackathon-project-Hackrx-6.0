#!/usr/bin/env python3
import sys
import os
import logging

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Add parent directory to Python path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    logger.info(f"Parent directory: {parent_dir}")
    logger.info(f"Python path: {sys.path[:3]}...")
    
    # Import the main application
    from app_groq_ultimate import app
    logger.info("✅ Successfully imported app_groq_ultimate")
    
except ImportError as e:
    logger.error(f"❌ Failed to import app_groq_ultimate: {e}")
    
    # Create a fallback FastAPI app
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    
    app = FastAPI(
        title="Groq API - Fallback Mode",
        description="Fallback application due to import error",
        version="1.0.0"
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/")
    async def root():
        return {
            "status": "error",
            "message": f"Failed to import main application: {str(e)}",
            "parent_dir": parent_dir,
            "python_path": sys.path[:5]
        }
    
    @app.get("/health")
    async def health():
        return {
            "status": "fallback_mode",
            "error": str(e),
            "python_version": sys.version
        }

except Exception as e:
    logger.error(f"❌ Unexpected error: {e}")
    
    # Create minimal fallback
    from fastapi import FastAPI
    
    app = FastAPI(title="Groq API - Error Mode")
    
    @app.get("/")
    async def error_root():
        return {"status": "error", "message": str(e)}

# This variable name 'app' is what Vercel looks for
logger.info("🚀 App ready for Vercel")
