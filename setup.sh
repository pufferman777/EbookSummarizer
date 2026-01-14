#!/bin/bash
# Setup script for Ebook Summarizer

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Setting up Ebook Summarizer..."

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate and install deps
echo "Installing dependencies..."
source venv/bin/activate
pip install -q -r requirements.txt

echo ""
echo "Setup complete!"
echo ""
echo "To run:"
echo "  ./run.sh"
echo ""
echo "Or manually:"
echo "  source venv/bin/activate"
echo "  streamlit run app.py --server.port 3002"
