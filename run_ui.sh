#!/bin/bash
# Start MSSDPPG Web UI Server

echo "🌬️  Starting MSSDPPG Web UI..."
echo "================================"

# Check if Flask is installed
python3 -c "import flask" 2>/dev/null || {
    echo "Installing dependencies..."
    pip3 install -r requirements.txt -q
}

echo "✓ Dependencies ready"
echo ""
echo "🚀 Starting server on http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 app.py
