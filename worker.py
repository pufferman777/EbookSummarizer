#!/usr/bin/env python3
"""
Background Worker for Ebook Summarizer
Processes summarization jobs independently from the Streamlit UI.
"""

import os
import json
import time
import signal
import sys
import requests
from pathlib import Path
from datetime import datetime

# Configuration
JOBS_DIR = Path(__file__).parent / "jobs"
OLLAMA_API_BASE = "http://localhost:11434/api"
POLL_INTERVAL = 2  # seconds

# Ensure jobs directory exists
JOBS_DIR.mkdir(exist_ok=True)

# Graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    print(f"\n[Worker] Shutdown requested (signal {signum})")
    shutdown_requested = True

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def log(msg: str):
    """Print timestamped log message."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def generate_summary(model: str, text: str, prompt: str) -> tuple[str, float]:
    """Generate summary using Ollama API."""
    payload = {
        "model": model,
        "prompt": f"```{text}```\n\n{prompt}",
        "stream": False
    }

    start_time = time.time()
    try:
        response = requests.post(f"{OLLAMA_API_BASE}/generate", json=payload, timeout=600)
        response.raise_for_status()
        result = response.json()
        output = result.get("response", "").strip()
        elapsed = time.time() - start_time
        return output, elapsed
    except Exception as e:
        return f"Error: {str(e)}", time.time() - start_time


def chunk_text(text: str, max_tokens: int = 2000) -> list[str]:
    """Split text into chunks of approximately max_tokens."""
    import re
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
    downloads = Path.home() / "Downloads"
    downloads.mkdir(exist_ok=True)
    return downloads


def update_job_status(job_id: str, **kwargs):
    """Update job status file."""
    status_file = JOBS_DIR / job_id / "status.json"

    if status_file.exists():
        with open(status_file, 'r') as f:
            status = json.load(f)
    else:
        status = {}

    status.update(kwargs)
    status["updated_at"] = datetime.now().isoformat()

    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)


def save_partial_result(job_id: str, chapter_title: str, summary: str):
    """Save partial result for a chapter."""
    results_file = JOBS_DIR / job_id / "results.json"

    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
    else:
        results = {"chapters": []}

    results["chapters"].append({
        "title": chapter_title,
        "summary": summary
    })

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)


def save_final_output(job_id: str, book_name: str, style_alias: str, custom_output_dir: str = None):
    """Compile and save final output to specified or default directory."""
    results_file = JOBS_DIR / job_id / "results.json"

    if not results_file.exists():
        return None

    with open(results_file, 'r') as f:
        results = json.load(f)

    # Compile all summaries
    all_summaries = []
    for ch in results.get("chapters", []):
        all_summaries.append(f"## {ch['title']}\n\n{ch['summary']}")

    content = "\n\n---\n\n".join(all_summaries)

    # Save to output directory (custom or default)
    if custom_output_dir:
        output_dir = Path(custom_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = get_summaries_dir()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{book_name}_{style_alias}_{timestamp}.md"
    filepath = output_dir / filename

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"# {book_name} - Summary\n\n")
        f.write(f"*Style: {style_alias} | Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n\n---\n\n")
        f.write(content)

    return str(filepath)


def has_trading_content(summaries: list) -> bool:
    """Check if summaries contain actual trading setups."""
    # Keywords that indicate trading setup content was found
    trading_keywords = [
        'entry', 'exit', 'stop loss', 'target', 'support', 'resistance',
        'breakout', 'setup', 'signal', 'indicator', 'pattern', 'moving average',
        'trend', 'long', 'short', 'buy', 'sell', 'position'
    ]

    all_text = ' '.join(summaries).lower()

    # Count keyword matches
    keyword_count = sum(1 for kw in trading_keywords if kw in all_text)

    # If we have at least 5 trading keywords, consider it valid
    return keyword_count >= 5


def process_job(job_id: str):
    """Process a single summarization job."""
    job_dir = JOBS_DIR / job_id
    job_file = job_dir / "job.json"

    if not job_file.exists():
        log(f"Job file not found: {job_file}")
        return

    with open(job_file, 'r') as f:
        job = json.load(f)

    log(f"Processing job: {job_id} - {job.get('book_name', 'Unknown')}")

    # Update status to running
    update_job_status(job_id,
        state="running",
        started_at=datetime.now().isoformat(),
        current_chapter=0,
        total_chapters=len(job.get("chapters", [])),
        processed_chunks=0,
        total_chunks=0,
        eta="calculating..."
    )

    chapters = job.get("chapters", [])
    model = job.get("model", "mistral")
    prompt = job.get("prompt", "Summarize this text.")
    chunk_size = job.get("chunk_size", 2000)
    book_name = job.get("book_name", "Unknown")
    style_alias = job.get("style_alias", "summary")
    custom_output_dir = job.get("output_dir")  # Custom output directory from batch job
    fallback_prompt = job.get("fallback_prompt")
    fallback_alias = job.get("fallback_alias")

    # Count total chunks
    total_chunks = 0
    for ch in chapters:
        total_chunks += len(chunk_text(ch.get("text", ""), chunk_size))

    update_job_status(job_id, total_chunks=total_chunks)

    processed_chunks = 0
    chunk_times = []
    all_chapter_summaries = []  # Keep track for fallback check

    for i, chapter in enumerate(chapters):
        if shutdown_requested:
            update_job_status(job_id, state="interrupted", message="Worker shutdown")
            return

        chapter_title = chapter.get("title", f"Chapter {i+1}")
        chapter_text = chapter.get("text", "")

        log(f"  Chapter {i+1}/{len(chapters)}: {chapter_title[:50]}...")

        update_job_status(job_id,
            current_chapter=i+1,
            current_chapter_title=chapter_title
        )

        chunks = chunk_text(chapter_text, chunk_size)
        chapter_summaries = []

        for j, chunk in enumerate(chunks):
            if shutdown_requested:
                update_job_status(job_id, state="interrupted", message="Worker shutdown")
                return

            summary, elapsed = generate_summary(model, chunk, prompt)
            chapter_summaries.append(summary)
            chunk_times.append(elapsed)
            processed_chunks += 1

            # Calculate ETA
            avg_time = sum(chunk_times) / len(chunk_times)
            remaining = total_chunks - processed_chunks
            eta_seconds = avg_time * remaining

            update_job_status(job_id,
                processed_chunks=processed_chunks,
                progress_pct=int((processed_chunks / total_chunks) * 100),
                eta=format_time(eta_seconds) if remaining > 0 else "almost done",
                last_chunk_time=f"{elapsed:.1f}s"
            )

        # Save chapter result immediately
        combined_summary = "\n\n".join(chapter_summaries)
        all_chapter_summaries.append(combined_summary)
        save_partial_result(job_id, chapter_title, combined_summary)
        log(f"    Saved chapter {i+1}")

    # Check if we need to use fallback (for trading style with no trading content)
    used_fallback = False
    if fallback_prompt and style_alias == "trading":
        if not has_trading_content(all_chapter_summaries):
            log(f"  No trading content found, using fallback style: {fallback_alias}")

            # Clear previous results
            results_file = job_dir / "results.json"
            if results_file.exists():
                results_file.unlink()

            # Re-process with fallback prompt
            update_job_status(job_id,
                message=f"Re-processing with {fallback_alias} style...",
                progress_pct=0,
                processed_chunks=0
            )

            processed_chunks = 0
            chunk_times = []

            for i, chapter in enumerate(chapters):
                if shutdown_requested:
                    update_job_status(job_id, state="interrupted", message="Worker shutdown")
                    return

                chapter_title = chapter.get("title", f"Chapter {i+1}")
                chapter_text = chapter.get("text", "")

                log(f"  [Fallback] Chapter {i+1}/{len(chapters)}: {chapter_title[:50]}...")

                update_job_status(job_id,
                    current_chapter=i+1,
                    current_chapter_title=f"[{fallback_alias}] {chapter_title}"
                )

                chunks = chunk_text(chapter_text, chunk_size)
                chapter_summaries = []

                for j, chunk in enumerate(chunks):
                    if shutdown_requested:
                        update_job_status(job_id, state="interrupted", message="Worker shutdown")
                        return

                    summary, elapsed = generate_summary(model, chunk, fallback_prompt)
                    chapter_summaries.append(summary)
                    chunk_times.append(elapsed)
                    processed_chunks += 1

                    avg_time = sum(chunk_times) / len(chunk_times)
                    remaining = total_chunks - processed_chunks
                    eta_seconds = avg_time * remaining

                    update_job_status(job_id,
                        processed_chunks=processed_chunks,
                        progress_pct=int((processed_chunks / total_chunks) * 100),
                        eta=format_time(eta_seconds) if remaining > 0 else "almost done",
                        last_chunk_time=f"{elapsed:.1f}s"
                    )

                combined_summary = "\n\n".join(chapter_summaries)
                save_partial_result(job_id, chapter_title, combined_summary)
                log(f"    Saved chapter {i+1}")

            style_alias = fallback_alias
            used_fallback = True

    # Save final output
    output_path = save_final_output(job_id, book_name, style_alias, custom_output_dir)

    update_job_status(job_id,
        state="completed",
        completed_at=datetime.now().isoformat(),
        progress_pct=100,
        eta="done",
        output_path=output_path,
        used_fallback=used_fallback
    )

    log(f"Job completed: {job_id} -> {output_path}")


def find_pending_jobs() -> list[str]:
    """Find all jobs with 'pending' state."""
    pending = []

    if not JOBS_DIR.exists():
        return pending

    for job_dir in JOBS_DIR.iterdir():
        if not job_dir.is_dir():
            continue

        status_file = job_dir / "status.json"
        if status_file.exists():
            try:
                with open(status_file, 'r') as f:
                    status = json.load(f)
                if status.get("state") == "pending":
                    pending.append(job_dir.name)
            except:
                pass

    return pending


def main():
    """Main worker loop."""
    log("Worker started")
    log(f"Jobs directory: {JOBS_DIR}")
    log(f"Polling interval: {POLL_INTERVAL}s")

    while not shutdown_requested:
        try:
            pending = find_pending_jobs()

            if pending:
                # Process oldest job first (by directory name which includes timestamp)
                pending.sort()
                job_id = pending[0]
                process_job(job_id)
            else:
                time.sleep(POLL_INTERVAL)

        except Exception as e:
            log(f"Error: {e}")
            time.sleep(POLL_INTERVAL)

    log("Worker stopped")


if __name__ == "__main__":
    main()
