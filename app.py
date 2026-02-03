"""
Ebook Summarizer - Streamlit GUI
A user-friendly interface for summarizing books chapter by chapter using local LLMs.
"""

import os
import json
import time
import re
import tempfile
import traceback
import requests
import streamlit as st
from pathlib import Path
from typing import List, Tuple
from datetime import datetime

# PDF/EPUB processing
import pypdf
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import ebooklib
from ebooklib import epub

# Local imports
from lib.epubunz import extract_html_files
from lib.epubsplit import SplitEpub
from lib.pdf_splitter import split_pdf, get_toc, prepare_page_ranges

# -----------------------------
# Configuration
# -----------------------------

OLLAMA_API_BASE = "http://localhost:11434/api"
JOBS_DIR = Path(__file__).parent / "jobs"
PREFS_FILE = Path(__file__).parent / ".user_prefs.json"

# Default directories for batch processing
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

# Ensure jobs directory exists
JOBS_DIR.mkdir(exist_ok=True)

PROMPTS = {
    "Trading Setups": {
        "alias": "trading",
        "prompt": """Analyze this text for actionable trading setups and strategies. Extract and list:

1. **Technical Setups**: Chart patterns, indicators, entry/exit signals, price action patterns, support/resistance levels, timeframes
2. **Fundamental Setups**: Earnings plays, sector rotations, macroeconomic triggers, valuation-based entries
3. **Other Setups**: Sentiment-based, seasonal patterns, intermarket relationships, risk management rules

For each setup found, provide:
- Setup name/type
- Entry criteria (specific conditions that must be met)
- Exit criteria (profit targets, stop losses)
- Risk management notes if mentioned

If the text doesn't contain trading setups, summarize any market insights or principles that could inform trading decisions."""
    },
    "Bulleted Notes": {
        "alias": "bnotes",
        "prompt": "Write comprehensive bulleted notes summarizing the provided text, with headings and terms in bold."
    },
    "Research Arguments": {
        "alias": "research",
        "prompt": "Does this text make any arguments? If so list them here."
    },
    "Concise Summary": {
        "alias": "concise",
        "prompt": "Repeat the provided passage, with Concision."
    },
    "Teacher Questions": {
        "alias": "teacher",
        "prompt": "Write a list of questions that can be answered by readers of the provided text."
    },
    "Key Quotes": {
        "alias": "quotes",
        "prompt": "Write a few dozen quotes inspired by the provided text."
    }
}

# -----------------------------
# User Preferences
# -----------------------------

def load_prefs() -> dict:
    """Load user preferences from file."""
    if PREFS_FILE.exists():
        try:
            return json.loads(PREFS_FILE.read_text())
        except:
            pass
    return {}

def save_prefs(prefs: dict) -> None:
    """Save user preferences to file."""
    PREFS_FILE.write_text(json.dumps(prefs, indent=2))

# -----------------------------
# Job Management
# -----------------------------

def create_job(book_name: str, chapters: List[dict], model: str, prompt: str,
               style_alias: str, chunk_size: int) -> str:
    """Create a new summarization job and return job ID."""
    job_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{book_name[:30].replace(' ', '_')}"
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Write job file
    job_data = {
        "book_name": book_name,
        "chapters": chapters,
        "model": model,
        "prompt": prompt,
        "style_alias": style_alias,
        "chunk_size": chunk_size,
        "created_at": datetime.now().isoformat()
    }

    with open(job_dir / "job.json", 'w') as f:
        json.dump(job_data, f, indent=2)

    # Write initial status
    status_data = {
        "state": "pending",
        "created_at": datetime.now().isoformat(),
        "progress_pct": 0,
        "current_chapter": 0,
        "total_chapters": len(chapters)
    }

    with open(job_dir / "status.json", 'w') as f:
        json.dump(status_data, f, indent=2)

    return job_id

def get_job_status(job_id: str) -> dict:
    """Get current status of a job."""
    status_file = JOBS_DIR / job_id / "status.json"
    if status_file.exists():
        with open(status_file, 'r') as f:
            return json.load(f)
    return {"state": "not_found"}

def get_job_results(job_id: str) -> dict:
    """Get results of a completed job."""
    results_file = JOBS_DIR / job_id / "results.json"
    if results_file.exists():
        with open(results_file, 'r') as f:
            return json.load(f)
    return {"chapters": []}

def get_all_jobs() -> List[dict]:
    """Get all jobs with their status."""
    jobs = []
    if not JOBS_DIR.exists():
        return jobs

    for job_dir in sorted(JOBS_DIR.iterdir(), reverse=True):
        if not job_dir.is_dir():
            continue

        job_file = job_dir / "job.json"
        status_file = job_dir / "status.json"

        if job_file.exists() and status_file.exists():
            try:
                with open(job_file, 'r') as f:
                    job = json.load(f)
                with open(status_file, 'r') as f:
                    status = json.load(f)

                jobs.append({
                    "job_id": job_dir.name,
                    "book_name": job.get("book_name", "Unknown"),
                    "created_at": job.get("created_at", ""),
                    **status
                })
            except:
                pass

    return jobs

def delete_job(job_id: str):
    """Delete a job and its files."""
    import shutil
    job_dir = JOBS_DIR / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)

# -----------------------------
# Ollama Integration
# -----------------------------

def get_ollama_models() -> List[str]:
    """Fetch available models from Ollama."""
    try:
        response = requests.get(f"{OLLAMA_API_BASE.replace('/api', '')}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [m["name"] for m in models]
    except Exception as e:
        st.warning(f"Could not connect to Ollama: {e}")
    return []

# -----------------------------
# File Processing
# -----------------------------

def sanitize_filename(filename: str) -> str:
    """Remove or replace unsafe characters in filenames."""
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = filename.rstrip('. ')
    return filename[:255] or 'untitled'

def natural_sort_key(s: str):
    """Natural sort key for filenames."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def split_epub_by_sections(input_file: str, output_dir: str) -> bool:
    """Split an EPUB file into multiple EPUBs by sections/chapters."""
    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(input_file, 'rb') as epub_file:
            splitter = SplitEpub(epub_file)
            split_lines = splitter.get_split_lines()

            for i, line in enumerate(split_lines):
                if line['toc'] and len(line['toc']) > 0:
                    section_title = line['toc'][0]
                    section_filename = sanitize_filename(section_title)
                    sequence_number = f"{i+1:04}"
                    output_path = os.path.join(output_dir, f"{sequence_number}_{section_filename}.epub")

                    with open(output_path, 'wb') as out_file:
                        splitter.write_split_epub(
                            out_file,
                            linenums=[i],
                            titleopt=section_title,
                            authoropts=splitter.origauthors,
                            descopt=f"Split section from {splitter.origtitle}"
                        )
        return True
    except Exception as e:
        st.warning(f"EPUB split error: {e}")
        return False

def epub_to_text(epub_path: str) -> str:
    """Convert EPUB to text."""
    book = epub.read_epub(epub_path)
    text_content = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            chapter_text = soup.get_text(separator=' ', strip=True)
            text_content.append(chapter_text)
    return '\n'.join(text_content)

def html_to_text(html_path: str) -> str:
    """Convert HTML to text."""
    with open(html_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
        return soup.get_text()

def pdf_to_text(pdf_path: str) -> str:
    """Convert PDF to text."""
    reader = PdfReader(pdf_path)
    return '\n'.join(page.extract_text() or '' for page in reader.pages)

def get_title_from_html(filepath: str) -> str:
    """Extract title from HTML file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')
            title_tag = soup.find('title')
            if title_tag and title_tag.string:
                return title_tag.string.strip()
            h1_tag = soup.find('h1')
            if h1_tag and h1_tag.string:
                return h1_tag.string.strip()
    except:
        pass
    return os.path.splitext(os.path.basename(filepath))[0]

def process_uploaded_file(uploaded_file, work_dir: str) -> Tuple[List[dict], str]:
    """
    Process uploaded PDF/EPUB and return list of chapters with their text.
    Returns: (chapters_list, error_message)
    """
    chapters = []
    file_ext = Path(uploaded_file.name).suffix.lower()
    file_name = Path(uploaded_file.name).stem
    file_name_clean = re.sub(r'[^\w\-_]', '', file_name.replace(" ", "-"))

    # Save uploaded file
    input_path = os.path.join(work_dir, uploaded_file.name)
    with open(input_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    output_dir = os.path.join(work_dir, f"split_{file_name_clean}")
    os.makedirs(output_dir, exist_ok=True)

    try:
        if file_ext == '.epub':
            success = split_epub_by_sections(input_path, output_dir)
            file_type = 'epub'
            if not success:
                extract_html_files(input_path, output_dir)
                file_type = 'html'
        elif file_ext == '.pdf':
            pdf = pypdf.PdfReader(input_path)
            toc = get_toc(pdf)
            page_count = len(pdf.pages)

            if toc:
                page_ranges = prepare_page_ranges(toc, regex=None, overlap=False, page_count=page_count)
                split_pdf(pdf, page_ranges, prefix=None, output_dir=output_dir)
                file_type = 'pdf'
            else:
                # No ToC - treat whole PDF as one chapter
                text = pdf_to_text(input_path)
                chapters.append({
                    'title': file_name,
                    'text': text,
                    'filename': uploaded_file.name
                })
                return chapters, ""
        else:
            return [], f"Unsupported file type: {file_ext}"

        # Process split files
        files = sorted(os.listdir(output_dir), key=natural_sort_key)
        for filename in files:
            filepath = os.path.join(output_dir, filename)

            if file_type == 'html' and filename.endswith('.html'):
                text = html_to_text(filepath)
                title = get_title_from_html(filepath)
            elif file_type == 'epub' and filename.endswith('.epub'):
                text = epub_to_text(filepath)
                try:
                    book = epub.read_epub(filepath)
                    title = book.get_metadata('DC', 'title')[0][0]
                except:
                    title = os.path.splitext(filename)[0]
            elif file_type == 'pdf' and filename.endswith('.pdf'):
                text = pdf_to_text(filepath)
                title = os.path.splitext(filename)[0]
            else:
                continue

            text = text.replace('\t', ' ').strip()
            if title:
                chapters.append({
                    'title': title,
                    'text': text,
                    'filename': filename
                })

        return chapters, ""

    except Exception as e:
        tb = traceback.format_exc()
        return [], f"{str(e)}\n\nTraceback:\n{tb}"

# -----------------------------
# Worker Status
# -----------------------------

def is_worker_running() -> bool:
    """Check if the background worker is running."""
    import subprocess
    try:
        result = subprocess.run(
            ["systemctl", "--user", "is-active", "ebook-worker.service"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() == "active"
    except:
        return False


def process_local_file(file_path: str, work_dir: str) -> Tuple[List[dict], str]:
    """
    Process a local PDF/EPUB file and return list of chapters with their text.
    Returns: (chapters_list, error_message)
    """
    chapters = []
    file_ext = Path(file_path).suffix.lower()
    file_name = Path(file_path).stem
    file_name_clean = re.sub(r'[^\w\-_]', '', file_name.replace(" ", "-"))

    output_dir = os.path.join(work_dir, f"split_{file_name_clean}")
    os.makedirs(output_dir, exist_ok=True)

    try:
        if file_ext == '.epub':
            success = split_epub_by_sections(file_path, output_dir)
            file_type = 'epub'
            if not success:
                extract_html_files(file_path, output_dir)
                file_type = 'html'
        elif file_ext == '.pdf':
            pdf = pypdf.PdfReader(file_path)
            toc = get_toc(pdf)
            page_count = len(pdf.pages)

            if toc:
                page_ranges = prepare_page_ranges(toc, regex=None, overlap=False, page_count=page_count)
                split_pdf(pdf, page_ranges, prefix=None, output_dir=output_dir)
                file_type = 'pdf'
            else:
                # No ToC - treat whole PDF as one chapter
                text = pdf_to_text(file_path)
                chapters.append({
                    'title': file_name,
                    'text': text,
                    'filename': Path(file_path).name
                })
                return chapters, ""
        else:
            return [], f"Unsupported file type: {file_ext}"

        # Process split files
        files = sorted(os.listdir(output_dir), key=natural_sort_key)
        for filename in files:
            filepath = os.path.join(output_dir, filename)

            if file_type == 'html' and filename.endswith('.html'):
                text = html_to_text(filepath)
                title = get_title_from_html(filepath)
            elif file_type == 'epub' and filename.endswith('.epub'):
                text = epub_to_text(filepath)
                try:
                    book = epub.read_epub(filepath)
                    title = book.get_metadata('DC', 'title')[0][0]
                except:
                    title = os.path.splitext(filename)[0]
            elif file_type == 'pdf' and filename.endswith('.pdf'):
                text = pdf_to_text(filepath)
                title = os.path.splitext(filename)[0]
            else:
                continue

            text = text.replace('\t', ' ').strip()
            if title:
                chapters.append({
                    'title': title,
                    'text': text,
                    'filename': filename
                })

        return chapters, ""

    except Exception as e:
        tb = traceback.format_exc()
        return [], f"{str(e)}\n\nTraceback:\n{tb}"


def get_files_in_directory(directory: str) -> List[str]:
    """Get all PDF and EPUB files in a directory."""
    if not os.path.isdir(directory):
        return []

    files = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.pdf', '.epub')):
            files.append(os.path.join(directory, filename))

    return sorted(files, key=lambda f: os.path.basename(f).lower())


def create_batch_jobs(files: List[str], model: str, prompt: str, style_alias: str,
                      chunk_size: int, output_dir: str, fallback_style: str = None) -> List[str]:
    """Create jobs for multiple files and return list of job IDs."""
    job_ids = []

    for file_path in files:
        work_dir = tempfile.mkdtemp()
        book_name = Path(file_path).stem

        chapters, error = process_local_file(file_path, work_dir)

        if error or not chapters:
            continue

        job_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{book_name[:30].replace(' ', '_')}"
        job_dir = JOBS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # Write job file with fallback info
        job_data = {
            "book_name": book_name,
            "chapters": chapters,
            "model": model,
            "prompt": prompt,
            "style_alias": style_alias,
            "chunk_size": chunk_size,
            "output_dir": output_dir,
            "source_file": file_path,
            "created_at": datetime.now().isoformat()
        }

        # Add fallback style if specified
        if fallback_style:
            job_data["fallback_prompt"] = PROMPTS[fallback_style]["prompt"]
            job_data["fallback_alias"] = PROMPTS[fallback_style]["alias"]

        with open(job_dir / "job.json", 'w') as f:
            json.dump(job_data, f, indent=2)

        # Write initial status
        status_data = {
            "state": "pending",
            "created_at": datetime.now().isoformat(),
            "progress_pct": 0,
            "current_chapter": 0,
            "total_chapters": len(chapters)
        }

        with open(job_dir / "status.json", 'w') as f:
            json.dump(status_data, f, indent=2)

        job_ids.append(job_id)

        # Small delay to ensure unique timestamps
        time.sleep(0.1)

    return job_ids

# -----------------------------
# Streamlit App
# -----------------------------

def main():
    st.set_page_config(
        page_title="Ebook Summarizer",
        page_icon="📚",
        layout="wide"
    )

    st.title("📚 Ebook Summarizer")
    st.caption("Upload a PDF or EPUB and get chapter-by-chapter summaries using local LLMs")

    # Sidebar configuration
    with st.sidebar:
        # Display username centered with gradient
        username = Path.home().name.capitalize()
        st.markdown(f"""
            <h1 style='
                text-align: center;
                background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                font-size: 2.5rem;
                font-weight: 700;
            '>{username}</h1>
        """, unsafe_allow_html=True)
        st.divider()

        # Worker status
        if is_worker_running():
            st.success("🟢 Worker: Running")
        else:
            st.error("🔴 Worker: Stopped")
            st.caption("Start with: `systemctl --user start ebook-worker`")

        st.divider()
        st.header("Configuration")

        # Model selection
        models = get_ollama_models()
        if not models:
            st.error("No Ollama models found. Make sure Ollama is running.")
            st.stop()

        # Suggest good summarization models first
        preferred = ['mistral', 'llama3', 'gemma', 'qwen']
        sorted_models = sorted(models, key=lambda m: (
            0 if any(p in m.lower() for p in preferred) else 1, m
        ))

        # Load saved model preference
        prefs = load_prefs()
        saved_model = prefs.get("model")
        default_idx = 0
        if saved_model and saved_model in sorted_models:
            default_idx = sorted_models.index(saved_model)

        selected_model = st.selectbox(
            "Select Model",
            sorted_models,
            index=default_idx,
            help="Choose the LLM to use for summarization"
        )

        # Save model preference when changed
        if selected_model != saved_model:
            prefs["model"] = selected_model
            save_prefs(prefs)

        # Prompt style (with persistence)
        prompt_styles = list(PROMPTS.keys())
        saved_style = prefs.get("style")
        style_idx = 0
        if saved_style and saved_style in prompt_styles:
            style_idx = prompt_styles.index(saved_style)

        prompt_style = st.selectbox(
            "Summary Style",
            prompt_styles,
            index=style_idx,
            help="Choose how you want the summaries formatted"
        )

        if prompt_style != saved_style:
            prefs["style"] = prompt_style
            save_prefs(prefs)

        # Chunk size
        chunk_size = st.slider(
            "Chunk Size (tokens)",
            min_value=500,
            max_value=4000,
            value=2000,
            step=250,
            help="Larger chunks = more context but slower processing"
        )

        st.divider()

        # Batch processing directories
        st.header("Batch Directories")

        # Load saved directories or use defaults
        saved_input_dir = prefs.get("input_dir", DEFAULT_INPUT_DIR)
        saved_output_dir = prefs.get("output_dir", DEFAULT_OUTPUT_DIR)

        input_dir = st.text_input(
            "Input Directory",
            value=saved_input_dir,
            help="Directory containing PDF/EPUB files to process"
        )

        output_dir = st.text_input(
            "Output Directory",
            value=saved_output_dir,
            help="Directory where summaries will be saved"
        )

        # Save directory preferences when changed
        if input_dir != saved_input_dir:
            prefs["input_dir"] = input_dir
            save_prefs(prefs)
        if output_dir != saved_output_dir:
            prefs["output_dir"] = output_dir
            save_prefs(prefs)

        st.divider()
        st.caption("**Tip:** Jobs run in background - you can close this tab and check back later!")

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📤 New Job", "📁 Batch Process", "📋 Job Queue"])

    with tab1:
        # Main area - File upload
        uploaded_file = st.file_uploader(
            "Drag and drop your book here, or click to browse",
            type=['pdf', 'epub'],
            help="Supports PDF and EPUB formats",
            key="book_uploader"
        )

        if uploaded_file:
            st.caption(f"📄 {uploaded_file.name} ({uploaded_file.size:,} bytes)")

            # Initialize session state
            if 'processed_file' not in st.session_state:
                st.session_state.processed_file = None
            if 'chapters' not in st.session_state:
                st.session_state.chapters = []

            # Process file if new
            if st.session_state.processed_file != uploaded_file.name:
                with st.spinner("Extracting chapters..."):
                    work_dir = tempfile.mkdtemp()
                    chapters, error = process_uploaded_file(uploaded_file, work_dir)

                    if error:
                        st.error(f"Error processing file: {error}")
                        st.stop()

                    st.session_state.processed_file = uploaded_file.name
                    st.session_state.book_name = Path(uploaded_file.name).stem
                    st.session_state.chapters = chapters

            chapters = st.session_state.chapters

            if not chapters:
                st.warning("No chapters found in the document.")
                st.stop()

            # Display chapter info
            st.success(f"Found **{len(chapters)}** chapters")

            # Chapter preview
            with st.expander("Preview Chapters", expanded=False):
                for i, ch in enumerate(chapters):
                    st.write(f"**{i+1}.** {ch['title']} ({len(ch['text']):,} chars)")

            # Submit job button
            if st.button("🚀 Start Summarization", type="primary", use_container_width=True):
                if not is_worker_running():
                    st.error("Worker is not running! Start it first with: `systemctl --user start ebook-worker`")
                else:
                    # Create job
                    job_id = create_job(
                        book_name=st.session_state.book_name,
                        chapters=chapters,
                        model=selected_model,
                        prompt=PROMPTS[prompt_style]["prompt"],
                        style_alias=PROMPTS[prompt_style]["alias"],
                        chunk_size=chunk_size
                    )
                    st.success(f"✅ Job submitted! ID: `{job_id}`")
                    st.info("Switch to **Job Queue** tab to monitor progress. You can close this tab - the job will continue in the background.")
                    st.session_state.active_job = job_id

    with tab2:
        st.subheader("Batch Processing")
        st.caption("Process multiple documents from a directory")

        # Show current directories
        st.markdown(f"**Input:** `{input_dir}`")
        st.markdown(f"**Output:** `{output_dir}`")

        # Check if directories exist
        input_valid = os.path.isdir(input_dir)
        output_valid = os.path.isdir(output_dir)

        if not input_valid:
            st.error(f"Input directory does not exist: {input_dir}")
        if not output_valid:
            st.warning(f"Output directory does not exist and will be created: {output_dir}")

        if input_valid:
            # Get files in input directory
            files = get_files_in_directory(input_dir)

            if not files:
                st.info("No PDF or EPUB files found in the input directory.")
            else:
                st.success(f"Found **{len(files)}** documents to process")

                # Preview files
                with st.expander("Preview Files", expanded=True):
                    for i, f in enumerate(files, 1):
                        size = os.path.getsize(f)
                        st.write(f"**{i}.** {os.path.basename(f)} ({size:,} bytes)")

                # Batch options
                st.divider()
                st.markdown("### Batch Options")

                col1, col2 = st.columns(2)
                with col1:
                    use_fallback = st.checkbox(
                        "Use fallback style",
                        value=True,
                        help="If Trading Setups finds no setups, fall back to another style"
                    )
                with col2:
                    fallback_style = st.selectbox(
                        "Fallback Style",
                        [s for s in PROMPTS.keys() if s != "Trading Setups"],
                        index=0,  # Bulleted Notes
                        disabled=not use_fallback
                    )

                # Start batch button
                st.divider()
                if st.button("🚀 Start Batch Processing", type="primary", use_container_width=True):
                    if not is_worker_running():
                        st.error("Worker is not running! Start it first with: `systemctl --user start ebook-worker`")
                    else:
                        # Create output directory if needed
                        if not output_valid:
                            os.makedirs(output_dir, exist_ok=True)

                        with st.spinner(f"Creating {len(files)} jobs..."):
                            job_ids = create_batch_jobs(
                                files=files,
                                model=selected_model,
                                prompt=PROMPTS[prompt_style]["prompt"],
                                style_alias=PROMPTS[prompt_style]["alias"],
                                chunk_size=chunk_size,
                                output_dir=output_dir,
                                fallback_style=fallback_style if use_fallback else None
                            )

                        if job_ids:
                            st.success(f"✅ Created {len(job_ids)} jobs!")
                            st.info("Switch to **Job Queue** tab to monitor progress.")
                        else:
                            st.error("No jobs could be created. Check that files are valid PDF/EPUB documents.")

    with tab3:
        st.subheader("Job Queue")

        # Refresh button
        if st.button("🔄 Refresh", key="refresh_jobs"):
            st.rerun()

        # Auto-refresh for running jobs
        jobs = get_all_jobs()
        running_jobs = [j for j in jobs if j.get("state") == "running"]

        if running_jobs:
            st.caption("⏳ Auto-refreshing every 5 seconds while jobs are running...")
            time.sleep(0.1)  # Small delay to prevent UI flicker
            st.empty()

        if not jobs:
            st.info("No jobs yet. Upload a book and start a summarization!")
        else:
            for job in jobs:
                job_id = job["job_id"]
                state = job.get("state", "unknown")
                book_name = job.get("book_name", "Unknown")
                progress = job.get("progress_pct", 0)

                # Job card
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])

                    with col1:
                        if state == "completed":
                            st.markdown(f"✅ **{book_name}**")
                        elif state == "running":
                            st.markdown(f"⏳ **{book_name}**")
                        elif state == "pending":
                            st.markdown(f"⏸️ **{book_name}**")
                        else:
                            st.markdown(f"❓ **{book_name}**")

                    with col2:
                        if state == "running":
                            eta = job.get("eta", "calculating...")
                            st.caption(f"ETA: {eta}")
                        elif state == "completed":
                            st.caption("Done!")
                        else:
                            st.caption(state.capitalize())

                    with col3:
                        if state == "completed":
                            if st.button("🗑️", key=f"del_{job_id}", help="Delete job"):
                                delete_job(job_id)
                                st.rerun()

                    # Progress bar
                    if state in ["running", "pending"]:
                        st.progress(progress / 100)
                        current_ch = job.get("current_chapter", 0)
                        total_ch = job.get("total_chapters", 0)
                        current_title = job.get("current_chapter_title", "")
                        if current_title:
                            st.caption(f"Chapter {current_ch}/{total_ch}: {current_title[:50]}...")

                    # Show results for completed jobs
                    if state == "completed":
                        output_path = job.get("output_path", "")
                        if output_path:
                            st.caption(f"📁 Saved to: `{output_path}`")

                        # Show summaries
                        results = get_job_results(job_id)
                        if results.get("chapters"):
                            with st.expander("📖 View Summaries", expanded=False):
                                for ch in results["chapters"]:
                                    st.markdown(f"### {ch['title']}")
                                    st.markdown(ch['summary'])
                                    st.divider()

                            # Download button
                            all_content = "\n\n---\n\n".join([
                                f"## {ch['title']}\n\n{ch['summary']}"
                                for ch in results["chapters"]
                            ])
                            st.download_button(
                                "📥 Download Summary",
                                data=f"# {book_name} - Summary\n\n{all_content}",
                                file_name=f"{book_name}_summary.md",
                                mime="text/markdown",
                                key=f"dl_{job_id}"
                            )

                    st.divider()

        # Auto-refresh for running jobs using Streamlit's native rerun
        if running_jobs:
            time.sleep(5)
            st.rerun()

if __name__ == "__main__":
    main()
