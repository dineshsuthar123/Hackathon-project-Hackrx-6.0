"""Compatibility entrypoint: expose the FastAPI app for simple runs.

This allows `python app.py` or servers that import `app` from this file
to work without touching the real implementation in app_groq_ultimate.py.
"""

from app_groq_ultimate import app  # re-export FastAPI app
