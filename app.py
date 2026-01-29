"""
Ebook Summarizer - Streamlit GUI
A user-friendly interface for summarizing books chapter by chapter using local LLMs.
"""

import os
import sys
import csv
import json
import time
import re
import shutil
import socket
import tempfile
import traceback
import requests
import streamlit as st
from pathlib import Path
from typing import Optional, List, Tuple, Generator

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
# User Preferences
# -----------------------------

PREFS_FILE = Path(__file__).parent / ".user_prefs.json"

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
# Configuration
# -----------------------------

OLLAMA_API_BASE = "http://localhost:11434/api"

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

def generate_summary(model: str, text: str, prompt: str) -> Tuple[str, float]:
    """Generate summary using Ollama API."""
    payload = {
        "model": model,
        "prompt": f"```{text}```\n\n{prompt}",
        "stream": False
    }

    start_time = time.time()
    try:
        response = requests.post(f"{OLLAMA_API_BASE}/generate", json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()
        output = result.get("response", "").strip()
        elapsed = time.time() - start_time
        return output, elapsed
    except Exception as e:
        return f"Error: {str(e)}", time.time() - start_time

def generate_title(model: str, text: str) -> str:
    """Generate a title for a chunk of text."""
    payload = {
        "model": model,
        "prompt": f"```{text[:500]}```\n\nWrite 8-11 words describing this text.",
        "stream": False
    }

    try:
        response = requests.post(f"{OLLAMA_API_BASE}/generate", json=payload, timeout=60)
        if response.status_code == 200:
            return response.json().get("response", "").strip().split('\n')[0]
    except:
        pass
    return text[:100].strip() + "..."

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

def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"

def get_summaries_dir() -> Path:
    """Get the summaries directory, trying Dropbox first, then Downloads."""
    dropbox_summaries = (
        Path.home() / "Dropbox" /
        "1. Kai Gao - Personal and Confidential - Dropbox" /
        "Personal" / "Trading" / "Readings" / "Summaries"
    )
    if dropbox_summaries.exists():
        return dropbox_summaries
    # Fallback to Downloads
    downloads = Path.home() / "Downloads"
    downloads.mkdir(exist_ok=True)
    return downloads

def save_summary_to_downloads(book_name: str, content: str, style_alias: str) -> str:
    """Save summary to Summaries folder (Dropbox if available, else Downloads)."""
    output_dir = get_summaries_dir()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{book_name}_{style_alias}_{timestamp}.md"
    filepath = output_dir / filename

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"# {book_name} - Summary\n\n")
        f.write(f"*Style: {style_alias} | Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n\n---\n\n")
        f.write(content)

    return str(filepath)

def chunk_text(text: str, max_tokens: int = 2000) -> List[str]:
    """Split text into chunks of approximately max_tokens."""
    # Rough estimate: 1 token ≈ 4 characters
    max_chars = max_tokens * 4

    if len(text) <= max_chars:
        return [text]

    chunks = []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks if chunks else [text]

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
        st.title(socket.gethostname().capitalize())
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

        # Prompt style
        prompt_style = st.selectbox(
            "Summary Style",
            list(PROMPTS.keys()),
            help="Choose how you want the summaries formatted"
        )

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
        st.caption("**Tip:** For 500+ page books, use smaller models (7B-13B) for faster processing.")

    # Main area - File upload
    uploaded_file = st.file_uploader(
        "Drag and drop your book here, or click to browse",
        type=['pdf', 'epub'],
        help="Supports PDF and EPUB formats",
        key="book_uploader"
    )

    if uploaded_file:
        st.info(f"File received: {uploaded_file.name} ({uploaded_file.size:,} bytes)")

        # Initialize session state
        if 'processed_file' not in st.session_state:
            st.session_state.processed_file = None
        if 'chapters' not in st.session_state:
            st.session_state.chapters = []
        if 'summaries' not in st.session_state:
            st.session_state.summaries = {}

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
                st.session_state.summaries = {}
                st.session_state.work_dir = work_dir

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

        # Summarize button
        col1, col2 = st.columns([1, 3])
        with col1:
            start_btn = st.button("🚀 Start Summarization", type="primary", use_container_width=True)

        if start_btn:
            prompt_text = PROMPTS[prompt_style]["prompt"]
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Results container
            results_container = st.container()

            total_chunks = 0
            processed_chunks = 0
            chunk_times = []

            # Count total chunks
            for ch in chapters:
                total_chunks += len(chunk_text(ch['text'], chunk_size))

            all_summaries = []

            for i, chapter in enumerate(chapters):
                chunks = chunk_text(chapter['text'], chunk_size)
                chapter_summaries = []

                for j, chunk in enumerate(chunks):
                    summary, elapsed = generate_summary(selected_model, chunk, prompt_text)
                    chapter_summaries.append(summary)
                    chunk_times.append(elapsed)
                    processed_chunks += 1

                    # Calculate progress and ETA
                    progress_pct = processed_chunks / total_chunks
                    progress_bar.progress(progress_pct)

                    avg_time = sum(chunk_times) / len(chunk_times)
                    remaining_chunks = total_chunks - processed_chunks
                    eta_seconds = avg_time * remaining_chunks

                    pct_display = int(progress_pct * 100)
                    eta_display = format_time(eta_seconds) if remaining_chunks > 0 else "almost done"

                    status_text.text(
                        f"Chapter {i+1}/{len(chapters)}: {chapter['title'][:40]}... | "
                        f"{pct_display}% | ETA: {eta_display}"
                    )

                # Combine chunk summaries
                combined_summary = "\n\n".join(chapter_summaries)
                st.session_state.summaries[chapter['title']] = combined_summary

                all_summaries.append(f"## {chapter['title']}\n\n{combined_summary}")

            progress_bar.progress(1.0)

            # Store final output
            st.session_state.final_output = "\n\n---\n\n".join(all_summaries)

            # Auto-save to Downloads
            book_name = Path(uploaded_file.name).stem
            style_alias = PROMPTS[prompt_style]["alias"]
            saved_path = save_summary_to_downloads(book_name, st.session_state.final_output, style_alias)
            st.session_state.saved_path = saved_path

            status_text.text(f"✅ Summarization complete! Saved to: {saved_path}")

    # Display results (outside of file upload block so it persists)
    if 'summaries' in st.session_state and st.session_state.summaries:
        st.divider()
        st.subheader("📝 Summaries")

        # Download button
        if 'final_output' in st.session_state and 'book_name' in st.session_state:
            book_name = st.session_state.book_name
            download_content = f"# {book_name} - Summary\n\n{st.session_state.final_output}"
            st.download_button(
                label="📥 Download All Summaries (Markdown)",
                data=download_content,
                file_name=f"{book_name}_summary.md",
                mime="text/markdown",
                key="download_summary"
            )
            if 'saved_path' in st.session_state:
                st.caption(f"Also saved to: {st.session_state.saved_path}")

        # Display each chapter summary
        for title, summary in st.session_state.summaries.items():
            with st.expander(f"📖 {title}", expanded=False):
                st.markdown(summary)

if __name__ == "__main__":
    main()
