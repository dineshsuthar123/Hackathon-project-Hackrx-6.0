#!/usr/bin/env python3
"""
Startup script for the LLM-Powered Query-Retrieval System
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def check_environment():
    """Check if all required environment variables are set"""
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\n💡 Please set these variables in your .env file")
        return False
    
    return True

def install_dependencies():
    """Install required dependencies"""
    import subprocess
    import sys
    
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

async def initialize_system():
    """Initialize the system components"""
    try:
        print("🔧 Initializing system components...")
        
        # Import after path setup
        from embedding_engine import EmbeddingEngine
        
        # Initialize embedding engine (downloads model if needed)
        embedding_engine = EmbeddingEngine()
        await embedding_engine.initialize()
        
        print("✅ System components initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize system: {str(e)}")
        return False

def start_server():
    """Start the FastAPI server"""
    import subprocess
    import sys
    
    print("🚀 Starting FastAPI server...")
    print("📡 Server will be available at: http://localhost:8000")
    print("📚 API documentation at: http://localhost:8000/docs")
    print("\n⏹️  Press Ctrl+C to stop the server")
    
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {str(e)}")

async def main():
    """Main startup function"""
    print("🎯 LLM-Powered Intelligent Query-Retrieval System")
    print("🏆 Hack 6.0 Hackathon Submission")
    print("=" * 60)
    
    # Check environment
    if not check_environment():
        print("\n💡 Create a .env file with your OpenAI API key:")
        print("   OPENAI_API_KEY=your_key_here")
        return
    
    # Install dependencies if needed
    if not os.path.exists("venv") and "--skip-install" not in sys.argv:
        if not install_dependencies():
            return
    
    # Initialize system
    if not await initialize_system():
        return
    
    print("\n✅ System ready!")
    
    # Start server
    if "--no-server" not in sys.argv:
        start_server()
    else:
        print("📝 Server startup skipped (--no-server flag)")

if __name__ == "__main__":
    # Handle command line arguments
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: python start.py [options]")
        print("\nOptions:")
        print("  --help, -h      Show this help message")
        print("  --skip-install  Skip dependency installation")
        print("  --no-server     Initialize system but don't start server")
        sys.exit(0)
    
    asyncio.run(main())
