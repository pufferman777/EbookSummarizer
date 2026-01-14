# Ebook Summarizer

A Streamlit GUI for summarizing books (PDF/EPUB) chapter by chapter using local LLMs via Ollama.

## Features

- Drag-and-drop PDF/EPUB upload
- Automatic chapter extraction from table of contents
- Model selection from your local Ollama models
- Multiple summary styles (bulleted notes, research arguments, concise, etc.)
- Progress tracking during processing
- Download summaries as Markdown

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) running locally with at least one model

## Quick Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/EbookSummarizer.git
cd EbookSummarizer

# Run setup script
./setup.sh

# Start the app
./run.sh
```

Then open http://localhost:3002

## Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run
streamlit run app.py --server.port 3002
```

## Run as System Service (Linux)

To start on boot:

```bash
# Copy service file
mkdir -p ~/.config/systemd/user
cp ebook-summarizer.service ~/.config/systemd/user/

# Edit the service file to match your paths
nano ~/.config/systemd/user/ebook-summarizer.service

# Enable and start
systemctl --user daemon-reload
systemctl --user enable --now ebook-summarizer

# Enable linger for boot persistence
loginctl enable-linger $USER
```

## Recommended Models

For 500+ page books, these models offer good speed/quality balance:
- `llama3.1:8b`
- `mistral:7b`
- `qwen3:latest`
- `gemma2:9b`

## Credits

Based on [ollama-ebook-summary](https://github.com/cognitivetech/ollama-ebook-summary) by cognitivetech.
