# Ebook Summarizer

A Streamlit GUI for summarizing books (PDF/EPUB) chapter by chapter using local LLMs via Ollama.

## Features

### Core
- Drag-and-drop PDF/EPUB upload
- Automatic chapter extraction from table of contents
- Model selection from your local Ollama models
- Multiple summary styles (Trading Setups, Bulleted Notes, Research Arguments, Concise, etc.)
- Progress tracking with ETA during processing
- Download summaries as Markdown

### Batch Processing
- Process entire directories of books at once
- Configurable input directory for source files
- Option to move processed files to "Processed" subfolder

### Smart Fallback
- Primary + fallback style support (e.g., try "Trading Setups" first, fall back to "Bulleted Notes")
- Samples first few chapters to detect content type before full processing
- Avoids reprocessing entire book when switching styles

### Reliability
- **Background worker** - Jobs continue even if you close the browser
- **Job queue** - Submit multiple jobs, processed in order
- **Job cancellation** - Cancel running or pending jobs from the UI
- **Retry logic** - Automatic retry with exponential backoff if Ollama fails
- **Model validation** - Checks if model exists before starting job

### Job Management
- View all jobs (pending, running, completed, failed, cancelled)
- Job age display (e.g., "2h ago")
- Clear completed jobs / Clear all jobs buttons
- Persistent preferences (model, style, directories saved between sessions)

### Configuration
All settings in one place at the top of the page:
- **Model** - Select from available Ollama models
- **Primary Style** - Main summarization style
- **Fallback Style** - Backup style if primary doesn't find relevant content
- **Chunk Size** - Tokens per chunk (larger = more context, slower)
- **Output Directory** - Where all summaries are saved

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) running locally with at least one model

## Quick Setup

```bash
# Clone the repo
git clone https://github.com/pufferman777/EbookSummarizer.git
cd EbookSummarizer

# Run setup script
./setup.sh

# Start the background worker (processes jobs)
systemctl --user start ebook-worker

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

# Start the background worker
python worker.py &

# Run the UI
streamlit run app.py --server.port 3002
```

## Run as System Services (Linux)

The app has two components:
1. **ebook-summarizer.service** - The Streamlit UI
2. **ebook-worker.service** - Background job processor

```bash
# Copy service files
mkdir -p ~/.config/systemd/user
cp ebook-summarizer.service ~/.config/systemd/user/
cp ebook-worker.service ~/.config/systemd/user/

# Edit the service files to match your paths
nano ~/.config/systemd/user/ebook-summarizer.service
nano ~/.config/systemd/user/ebook-worker.service

# Enable and start both
systemctl --user daemon-reload
systemctl --user enable --now ebook-summarizer
systemctl --user enable --now ebook-worker

# Enable linger for boot persistence
loginctl enable-linger $USER
```

## Summary Styles

| Style | Description |
|-------|-------------|
| **Trading Setups** | Extracts actionable trading setups, entry/exit criteria, patterns |
| **Bulleted Notes** | Comprehensive bulleted notes with headings and bold terms |
| **Research Arguments** | Lists arguments made in the text |
| **Concise Summary** | Condensed version of the content |
| **Teacher Questions** | Questions that can be answered by readers |
| **Key Quotes** | Notable quotes from the text |

## Recommended Models

For 500+ page books, these models offer good speed/quality balance:
- `llama3.1:8b` / `llama3.1:70b`
- `mistral:7b`
- `qwen3:latest`
- `gemma2:9b`

## File Structure

```
EbookSummarizer/
├── app.py              # Streamlit UI
├── worker.py           # Background job processor
├── config.py           # Centralized configuration
├── jobs/               # Job data and results
├── lib/                # PDF/EPUB processing utilities
└── .user_prefs.json    # Saved user preferences
```

## Credits

Based on [ollama-ebook-summary](https://github.com/cognitivetech/ollama-ebook-summary) by cognitivetech.
