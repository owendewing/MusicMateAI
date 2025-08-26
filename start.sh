#!/bin/bash

# Start nginx in the background
echo "Starting nginx..."
nginx

# Start the FastAPI backend
echo "Starting FastAPI backend..."
cd /app/backend
uvicorn main:app --host 0.0.0.0 --port 8000 