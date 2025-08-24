#!/bin/bash

# Start nginx in the background
echo "Starting nginx..."
nginx

# Start the FastAPI backend
echo "Starting FastAPI backend..."
cd /app/backend
python main.py 