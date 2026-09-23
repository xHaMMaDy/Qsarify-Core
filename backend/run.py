# The MODIFIED run.py for Qsarify

import os

from waitress import serve
from app import app, load_model_data

# This is the production-ready way to run your application.
# It loads the model once at startup, then starts a stable server.

if __name__ == '__main__':
    print("--- Preparing to start production server ---")
    
    # 1. Load the model data first.
    print("Attempting to load model data...")
    if load_model_data():
        print("Model loaded successfully.")
        
        # 2. Start the Waitress server to serve the app.
        # Keep the backend separate from the Next.js development port (5001).
        # A reverse proxy can expose this loopback listener publicly.
        host = os.environ.get("QSARIFY_BACKEND_HOST", "127.0.0.1")
        port = int(os.environ.get("QSARIFY_BACKEND_PORT", os.environ.get("PORT", "5051")))
        print(f"Starting Qsarify Waitress server on http://{host}:{port}")
        serve(app, host=host, port=port)
    else:
        print("FATAL: Could not load model data. Server will not start.")
