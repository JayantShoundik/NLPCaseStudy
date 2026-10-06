"""
server.py - Easy launcher for QuickBasket AI Full-Stack Platform
Usage:
    python3 server.py
"""
import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import uvicorn

if __name__ == "__main__":
    print("=" * 60)
    print("⚡ QuickBasket AI — Market Basket & NLP Culinary Intelligence")
    print("=" * 60)
    print("🚀 Server starting...")
    print("👉 Web App URL: http://localhost:8000")
    print("👉 API Docs:    http://localhost:8000/docs")
    print("👤 Demo User:   demo@antigravity.ai / Password123!")
    print("=" * 60)
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
