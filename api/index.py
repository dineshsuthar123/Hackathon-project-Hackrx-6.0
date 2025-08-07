import sys
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the app from the main module
try:
    from app_groq_ultimate import app
except ImportError as e:
    # Fallback if import fails
    print(f"Import error: {e}")
    app = FastAPI(title="Groq API - Import Error", description="Failed to import main app")
    
    @app.get("/")
    async def root():
        return {"error": "Failed to import main application", "status": "error"}

# Ensure CORS is properly configured for Vercel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# This is the handler that Vercel will call
def handler(event, context):
    return app

# Export for Vercel
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
