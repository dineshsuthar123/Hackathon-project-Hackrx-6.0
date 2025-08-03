"""
RAG Document Reading API
Production deployment with Mission 10/10 precision system
"""

# Import the complete 10/10 mission system
from app_mission10 import *

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
