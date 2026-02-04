"""
Centralized configuration for Ebook Summarizer.
"""

from pathlib import Path

# Directory configuration
BASE_DIR = Path(__file__).parent
JOBS_DIR = BASE_DIR / "jobs"
PREFS_FILE = BASE_DIR / ".user_prefs.json"

# Ollama API configuration
OLLAMA_API_BASE = "http://localhost:11434/api"
POLL_INTERVAL = 2  # seconds between worker polls

# Default directories (user can change in UI, saved to preferences)
DEFAULT_INPUT_DIR = str(Path.home() / "Documents")
DEFAULT_OUTPUT_DIR = str(Path.home() / "Documents" / "Summaries")

# Retry configuration
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1  # seconds, doubles each retry

# Ensure jobs directory exists
JOBS_DIR.mkdir(exist_ok=True)
