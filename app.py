"""
Ebook Summarizer - Streamlit GUI
A user-friendly interface for summarizing books chapter by chapter using local LLMs.
"""

import os
import json
import time
import re
import shutil
import tempfile
import traceback
import requests
import streamlit as st
from pathlib import Path
from typing import List, Tuple, Union
from datetime import datetime
from io import BytesIO

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
from config import (
    JOBS_DIR, PREFS_FILE, OLLAMA_API_BASE,
    DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR
)

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
    job_dir = JOBS_DIR / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)


def cancel_job(job_id: str):
    """Request cancellation of a running or pending job."""
    status_file = JOBS_DIR / job_id / "status.json"
    if status_file.exists():
        with open(status_file, 'r') as f:
            status = json.load(f)
        status["cancel_requested"] = True
        status["cancel_requested_at"] = datetime.now().isoformat()
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)


def clear_completed_jobs():
    """Delete all completed jobs."""
    jobs = get_all_jobs()
    count = 0
    for job in jobs:
        if job.get("state") == "completed":
            delete_job(job["job_id"])
            count += 1
    return count


def clear_all_jobs():
    """Delete all jobs regardless of state."""
    jobs = get_all_jobs()
    for job in jobs:
        delete_job(job["job_id"])
    return len(jobs)


def format_job_age(created_at: str) -> str:
    """Format job age as human-readable string (e.g., '2 hours ago')."""
    if not created_at:
        return ""
    try:
        created = datetime.fromisoformat(created_at)
        delta = datetime.now() - created
        seconds = delta.total_seconds()

        if seconds < 60:
            return "just now"
        elif seconds < 3600:
            mins = int(seconds / 60)
            return f"{mins}m ago"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            return f"{hours}h ago"
        else:
            days = int(seconds / 86400)
            return f"{days}d ago"
    except:
        return ""

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

def process_file(source: Union[str, "st.runtime.uploaded_file_manager.UploadedFile"],
                 work_dir: str) -> Tuple[List[dict], str]:
    """
    Process PDF/EPUB from path or UploadedFile object.
    Returns: (chapters_list, error_message)
    """
    chapters = []

    # Handle UploadedFile vs file path
    if hasattr(source, 'getbuffer'):  # UploadedFile
        input_path = os.path.join(work_dir, source.name)
        with open(input_path, 'wb') as f:
            f.write(source.getbuffer())
        file_name = Path(source.name).stem
        original_filename = source.name
    else:  # File path string
        input_path = source
        file_name = Path(source).stem
        original_filename = Path(source).name

    file_ext = Path(input_path).suffix.lower()
    file_name_clean = re.sub(r'[^\w\-_]', '', file_name.replace(" ", "-"))

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
                    'filename': original_filename
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
                      chunk_size: int, output_dir: str, fallback_style: str = None,
                      move_after_processing: bool = False,
                      progress_callback=None) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Create jobs for multiple files.
    Returns: (job_ids, failed_files) where failed_files is [(filename, error_message), ...]
    """
    job_ids = []
    failed_files = []

    for i, file_path in enumerate(files):
        # Report progress if callback provided
        if progress_callback:
            progress_callback(i, len(files), Path(file_path).name)

        work_dir = tempfile.mkdtemp()
        book_name = Path(file_path).stem

        try:
            chapters, error = process_file(file_path, work_dir)

            if error:
                failed_files.append((Path(file_path).name, error))
                continue

            if not chapters:
                failed_files.append((Path(file_path).name, "No chapters found in document"))
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
                "move_after_processing": move_after_processing,
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

        except Exception as e:
            failed_files.append((Path(file_path).name, str(e)))

    return job_ids, failed_files

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

    # Sidebar - minimal, just status
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
        st.caption("**Tip:** Jobs run in background - you can close this tab and check back later!")

    # Load user preferences
    prefs = load_prefs()

    # ===========================================
    # UNIFIED CONFIGURATION SECTION
    # ===========================================
    with st.expander("⚙️ Summarization Settings", expanded=True):
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

        prompt_styles = list(PROMPTS.keys())

        # Row 1: Model (full width)
        saved_model = prefs.get("model")
        default_idx = 0
        if saved_model and saved_model in sorted_models:
            default_idx = sorted_models.index(saved_model)

        selected_model = st.selectbox(
            "Model",
            sorted_models,
            index=default_idx,
            help="LLM to use for summarization"
        )

        if selected_model != saved_model:
            prefs["model"] = selected_model
            save_prefs(prefs)

        # Row 2: Primary Style | Fallback Style | Chunk Size
        col1, col2, col3 = st.columns(3)

        with col1:
            saved_style = prefs.get("style")
            style_idx = 0
            if saved_style and saved_style in prompt_styles:
                style_idx = prompt_styles.index(saved_style)

            primary_style = st.selectbox(
                "Primary Style",
                prompt_styles,
                index=style_idx,
                help="Main summarization style"
            )

            if primary_style != saved_style:
                prefs["style"] = primary_style
                save_prefs(prefs)

        with col2:
            fallback_options = [s for s in prompt_styles if s != primary_style]
            saved_fallback = prefs.get("fallback_style", fallback_options[0] if fallback_options else None)
            saved_use_fallback = prefs.get("use_fallback", primary_style == "Trading Setups")

            if saved_fallback not in fallback_options:
                saved_fallback = fallback_options[0] if fallback_options else None

            fallback_idx = fallback_options.index(saved_fallback) if saved_fallback in fallback_options else 0

            # Checkbox + selectbox combined
            use_fallback = st.checkbox(
                "Enable Fallback Style",
                value=saved_use_fallback,
                help="Use backup style if primary doesn't find relevant content"
            )

            if use_fallback != saved_use_fallback:
                prefs["use_fallback"] = use_fallback
                save_prefs(prefs)

            fallback_style = st.selectbox(
                "Fallback Style",
                fallback_options,
                index=fallback_idx,
                disabled=not use_fallback,
                label_visibility="collapsed"
            )

            if fallback_style != saved_fallback:
                prefs["fallback_style"] = fallback_style
                save_prefs(prefs)

        with col3:
            saved_chunk = prefs.get("chunk_size", 2000)
            chunk_size = st.number_input(
                "Chunk Size (tokens)",
                min_value=500,
                max_value=4000,
                value=saved_chunk,
                step=250,
                help="Larger = more context, slower"
            )

            if chunk_size != saved_chunk:
                prefs["chunk_size"] = chunk_size
                save_prefs(prefs)

        # Row 3: Output Directory (full width)
        saved_output_dir = prefs.get("output_dir", DEFAULT_OUTPUT_DIR)
        output_dir = st.text_input(
            "Output Directory",
            value=saved_output_dir,
            help="Where summaries will be saved"
        )

        if output_dir != saved_output_dir:
            prefs["output_dir"] = output_dir
            save_prefs(prefs)

        # Validate output directory
        if os.path.isdir(output_dir):
            st.caption(f"✓ Output directory exists")
        else:
            st.caption(f"⚠ Directory will be created when first job completes")

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
                    chapters, error = process_file(uploaded_file, work_dir)

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
                    # Create job with fallback support
                    job_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{st.session_state.book_name[:30].replace(' ', '_')}"
                    job_dir = JOBS_DIR / job_id
                    job_dir.mkdir(parents=True, exist_ok=True)

                    job_data = {
                        "book_name": st.session_state.book_name,
                        "chapters": chapters,
                        "model": selected_model,
                        "prompt": PROMPTS[primary_style]["prompt"],
                        "style_alias": PROMPTS[primary_style]["alias"],
                        "chunk_size": chunk_size,
                        "output_dir": output_dir,
                        "created_at": datetime.now().isoformat()
                    }

                    # Add fallback if enabled
                    if use_fallback and fallback_style:
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

                    st.success(f"✅ Job submitted! ID: `{job_id}`")
                    st.info("Switch to **Job Queue** tab to monitor progress.")
                    st.session_state.active_job = job_id

    with tab2:
        st.subheader("Batch Processing")
        st.caption("Process multiple documents from a directory")

        # Input directory configuration (batch-specific)
        saved_input_dir = prefs.get("input_dir", DEFAULT_INPUT_DIR)

        input_dir = st.text_input(
            "Input Directory",
            value=saved_input_dir,
            help="Directory containing PDF/EPUB files to process"
        )

        if input_dir != saved_input_dir:
            prefs["input_dir"] = input_dir
            save_prefs(prefs)

        # Check input directory
        input_valid = os.path.isdir(input_dir)
        if input_valid:
            files_count = len(get_files_in_directory(input_dir))
            st.caption(f"✓ Found {files_count} files")
        else:
            st.caption("✗ Directory not found")

        if input_valid:
            # Get files in input directory
            files = get_files_in_directory(input_dir)

            if not files:
                st.info("No PDF or EPUB files found in the input directory.")
            else:
                st.success(f"Found **{len(files)}** documents to process")

                # Preview files
                with st.expander("Preview Files", expanded=False):
                    for i, f in enumerate(files, 1):
                        size = os.path.getsize(f)
                        st.write(f"**{i}.** {os.path.basename(f)} ({size:,} bytes)")

                # Batch options
                st.divider()
                st.markdown("### Options")

                move_after = st.checkbox(
                    "Move files to 'Processed' folder after completion",
                    value=False,
                    help="After successful processing, move source files to a 'Processed' subfolder"
                )

                # Start batch button
                st.divider()
                if st.button("🚀 Start Batch Processing", type="primary", use_container_width=True):
                    if not is_worker_running():
                        st.error("Worker is not running! Start it first with: `systemctl --user start ebook-worker`")
                    else:
                        # Create output directory if needed
                        if not os.path.isdir(output_dir):
                            os.makedirs(output_dir, exist_ok=True)

                        # Progress placeholder for batch creation
                        progress_placeholder = st.empty()
                        status_placeholder = st.empty()

                        def update_progress(current, total, filename):
                            progress_placeholder.progress(current / total)
                            status_placeholder.caption(f"Processing {current}/{total}: {filename}")

                        job_ids, failed_files = create_batch_jobs(
                            files=files,
                            model=selected_model,
                            prompt=PROMPTS[primary_style]["prompt"],
                            style_alias=PROMPTS[primary_style]["alias"],
                            chunk_size=chunk_size,
                            output_dir=output_dir,
                            fallback_style=fallback_style if use_fallback else None,
                            move_after_processing=move_after,
                            progress_callback=update_progress
                        )

                        progress_placeholder.empty()
                        status_placeholder.empty()

                        if job_ids:
                            st.success(f"Created {len(job_ids)} jobs!")

                        if failed_files:
                            with st.expander(f"Failed files ({len(failed_files)})", expanded=True):
                                for filename, error in failed_files:
                                    st.error(f"**{filename}**: {error[:100]}...")

                        if job_ids:
                            st.info("Switch to **Job Queue** tab to monitor progress.")
                        elif not failed_files:
                            st.error("No jobs could be created. Check that files are valid PDF/EPUB documents.")

    with tab3:
        st.subheader("Job Queue")

        # Job management buttons
        col_refresh, col_clear_done, col_clear_all = st.columns([1, 1, 1])
        with col_refresh:
            if st.button("🔄 Refresh", key="refresh_jobs", use_container_width=True):
                st.rerun()
        with col_clear_done:
            if st.button("🧹 Clear Completed", key="clear_done", use_container_width=True):
                count = clear_completed_jobs()
                if count > 0:
                    st.toast(f"Cleared {count} completed jobs")
                    st.rerun()
        with col_clear_all:
            if st.button("🗑️ Clear All", key="clear_all", use_container_width=True):
                # Use session state for confirmation
                st.session_state.confirm_clear_all = True

        # Confirmation dialog for Clear All
        if st.session_state.get("confirm_clear_all"):
            st.warning("Are you sure you want to delete ALL jobs?")
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("Yes, delete all", type="primary", use_container_width=True):
                    count = clear_all_jobs()
                    st.session_state.confirm_clear_all = False
                    st.toast(f"Cleared {count} jobs")
                    st.rerun()
            with col_no:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.confirm_clear_all = False
                    st.rerun()

        jobs = get_all_jobs()
        running_jobs = [j for j in jobs if j.get("state") == "running"]
        pending_jobs = [j for j in jobs if j.get("state") == "pending"]

        if running_jobs or pending_jobs:
            st.caption("⏳ Auto-refreshing every 5 seconds while jobs are active...")

        if not jobs:
            st.info("No jobs yet. Upload a book and start a summarization!")
        else:
            for job in jobs:
                job_id = job["job_id"]
                state = job.get("state", "unknown")
                book_name = job.get("book_name", "Unknown")
                progress = job.get("progress_pct", 0)
                created_at = job.get("created_at", "")
                job_age = format_job_age(created_at)
                message = job.get("message", "")

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
                        elif state == "cancelled":
                            st.markdown(f"🚫 **{book_name}**")
                        elif state == "failed":
                            st.markdown(f"❌ **{book_name}**")
                        elif state == "interrupted":
                            st.markdown(f"⚠️ **{book_name}**")
                        else:
                            st.markdown(f"❓ **{book_name}**")

                    with col2:
                        if state == "running":
                            eta = job.get("eta", "calculating...")
                            st.caption(f"ETA: {eta}")
                        elif state == "completed":
                            st.caption(f"Done {job_age}" if job_age else "Done!")
                        elif state in ["failed", "cancelled", "interrupted"]:
                            st.caption(state.capitalize())
                        else:
                            st.caption(f"{state.capitalize()} {job_age}" if job_age else state.capitalize())

                    with col3:
                        # Cancel button for running/pending jobs
                        if state in ["running", "pending"]:
                            if st.button("⛔", key=f"cancel_{job_id}", help="Cancel job"):
                                cancel_job(job_id)
                                st.toast(f"Cancellation requested for {book_name}")
                                st.rerun()
                        # Delete button for finished jobs
                        elif state in ["completed", "failed", "cancelled", "interrupted"]:
                            if st.button("🗑️", key=f"del_{job_id}", help="Delete job"):
                                delete_job(job_id)
                                st.rerun()

                    # Show error message for failed/cancelled jobs
                    if state in ["failed", "cancelled", "interrupted"] and message:
                        st.error(message)

                    # Progress bar for active jobs
                    if state in ["running", "pending"]:
                        st.progress(progress / 100)
                        current_ch = job.get("current_chapter", 0)
                        total_ch = job.get("total_chapters", 0)
                        current_title = job.get("current_chapter_title", "")
                        if current_title:
                            st.caption(f"Chapter {current_ch}/{total_ch}: {current_title[:50]}...")
                        if job.get("cancel_requested"):
                            st.warning("Cancellation requested, will stop after current chunk...")

                    # Show results for completed jobs
                    if state == "completed":
                        output_path = job.get("output_path", "")
                        if output_path:
                            st.caption(f"📁 Saved to: `{output_path}`")

                        if job.get("used_fallback_dir"):
                            st.warning("⚠️ Output directory was not configured - saved to fallback location (~/Downloads)")

                        if job.get("used_fallback"):
                            st.caption("📝 Used fallback style (no trading content found)")

                        if job.get("moved_source_to"):
                            st.caption(f"📦 Source moved to: `{job['moved_source_to']}`")

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

        # Non-blocking auto-refresh using fragment decorator
        # Note: st.fragment requires Streamlit 1.33+, fallback to rerun if not available
        if running_jobs or pending_jobs:
            # Use a placeholder that will trigger refresh
            refresh_container = st.empty()
            with refresh_container:
                # This approach avoids blocking the UI
                import threading

                def delayed_rerun():
                    time.sleep(5)

                # Start background timer (non-blocking approach)
                # The actual refresh happens via the rerun at the end
                time.sleep(5)
                st.rerun()

if __name__ == "__main__":
    main()
