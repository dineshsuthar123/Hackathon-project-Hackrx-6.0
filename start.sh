#!/bin/bash
# Production start script for Mission 10/10 system

# Use the Mission 10/10 system for maximum accuracy
uvicorn app_mission10:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
