#!/bin/bash
# Run Ebook Summarizer on port 3002

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Try local venv first, then system-wide venv location
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "$HOME/Documents/PythonProject/EbookSummarizer_venv" ]; then
    source "$HOME/Documents/PythonProject/EbookSummarizer_venv/bin/activate"
fi

streamlit run app.py --server.port 3002 --server.address localhost --server.headless true
