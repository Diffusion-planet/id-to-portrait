#!/bin/bash
# Run FastAPI backend server

cd "$(dirname "$0")"

# Activate virtual environment if exists
if [ -d "venv_mps" ]; then
    source venv_mps/bin/activate
fi

# Install backend dependencies
pip install -r backend/requirements.txt -q

# Create necessary directories
mkdir -p uploads output models_cache

# Run server
cd backend
python main.py
