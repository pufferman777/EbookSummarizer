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

# Default batch processing directories
DEFAULT_INPUT_DIR = str(
    Path.home() / "Dropbox" /
    "1. Kai Gao - Personal and Confidential - Dropbox" /
    "Personal" / "Trading" / "Readings" / "Need Summaries"
)
DEFAULT_OUTPUT_DIR = str(
    Path.home() / "Dropbox" /
    "1. Kai Gao - Personal and Confidential - Dropbox" /
    "Personal" / "Trading" / "Readings" / "Summaries"
)

# Retry configuration
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1  # seconds, doubles each retry

# Ensure jobs directory exists
JOBS_DIR.mkdir(exist_ok=True)
