"""
Generate Q&A pairs from Israeli legal documents using Gemini 2.0 Flash.

Usage:
    set GEMINI_API_KEY=your_key_here
    python generate_qa.py

Get a free API key at: https://aistudio.google.com/apikey
Auto-resumes if interrupted. Progress saved every 50 docs.
"""

import json
import os
import random
import sys
import time
import hashlib
from pathlib import Path

from google import genai

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# === Configuration ===
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
DATA_DIR = Path("data/filtered")
OUTPUT_FILE = Path("data/training/instructions.jsonl")
PROGRESS_FILE = Path("data/training/qa_progress.json")
SAMPLE_LIST_FILE = Path("data/training/sampled_docs.json")

# Sampling config
SAMPLE_COURT = 2000
SAMPLE_KOLZCHUT = 5347      # All of them
SAMPLE_WIKISOURCE = 1000
MIN_WORDS = 200
MAX_CHARS = 8000  # Truncate long docs to save tokens

# Rate limiting (free tier: 15 RPM, paid: up to 2000 RPM)
REQUESTS_PER_MINUTE = 15
DELAY = 60.0 / REQUESTS_PER_MINUTE

PROMPT_TEMPLATE = """אתה מומחה למשפט ישראלי. קרא את הטקסט המשפטי הבא וצור ממנו 3 זוגות של שאלה-תשובה בעברית.

כללים:
- השאלות צריכות להיות שאלות מעשיות שאזרח רגיל עשוי לשאול
- התשובות צריכות להיות מבוססות רק על הטקסט הנתון
- התשובות צריכות להיות ברורות, מפורטות ומדויקות
- כתוב בעברית בלבד

הטקסט:
{text}

החזר את התוצאה בפורמט JSON בלבד, ללא טקסט נוסף:
[
  {{"question": "השאלה כאן", "answer": "התשובה כאן"}},
  {{"question": "השאלה כאן", "answer": "התשובה כאן"}},
  {{"question": "השאלה כאן", "answer": "התשובה כאן"}}
]"""


def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f).get("done", []))
    return set()


def save_progress(done_set):
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({"done": list(done_set)}, f)


def file_hash(path):
    return hashlib.md5(str(path).encode()).hexdigest()[:12]


def build_sample_list():
    """Build and cache the sample list so it's fast on subsequent runs."""
    if SAMPLE_LIST_FILE.exists():
        print("Loading cached sample list...")
        with open(SAMPLE_LIST_FILE, "r", encoding="utf-8") as f:
            entries = json.load(f)
        print(f"  {len(entries)} documents in sample list")
        return entries

    print("Building sample list (first run only, takes a few minutes)...")
    random.seed(42)
    entries = []

    sources = {
        "court": SAMPLE_COURT,
        "kolzchut": SAMPLE_KOLZCHUT,
        "wikisource_laws": SAMPLE_WIKISOURCE,
    }

    for source, n in sources.items():
        source_dir = DATA_DIR / source
        if not source_dir.exists():
            print(f"  Warning: {source_dir} not found, skipping")
            continue

        # Use os.listdir (much faster than glob on Windows)
        all_files = [f for f in os.listdir(source_dir) if f.endswith(".json")]
        print(f"  {source}: {len(all_files)} files, sampling up to {n}...")

        # Oversample to account for short docs
        candidates = random.sample(all_files, min(n * 2, len(all_files)))

        count = 0
        for fname in candidates:
            if count >= n:
                break
            fpath = source_dir / fname
            try:
                with open(fpath, encoding="utf-8") as fh:
                    doc = json.load(fh)
                text = doc.get("text", "")
                if len(text.split()) >= MIN_WORDS:
                    entries.append({
                        "path": str(fpath),
                        "source": source,
                    })
                    count += 1
            except Exception:
                continue

        print(f"  {source}: {count} docs sampled")

    random.shuffle(entries)

    # Cache for future runs
    SAMPLE_LIST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SAMPLE_LIST_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f)
    print(f"  Sample list cached to {SAMPLE_LIST_FILE}")

    return entries


def parse_qa_response(response_text):
    text = response_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    if text.startswith("json"):
        text = text[4:].strip()

    pairs = json.loads(text)
    results = []
    for pair in pairs:
        q = pair.get("question", "").strip()
        a = pair.get("answer", "").strip()
        if q and a and len(a) > 20:
            results.append({
                "conversations": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ]
            })
    return results


def main():
    if not GEMINI_API_KEY:
        print("Error: Set GEMINI_API_KEY environment variable")
        print("  Get your key at: https://aistudio.google.com/apikey")
        print("  Then run:")
        print("    set GEMINI_API_KEY=your_key_here")
        print("    python generate_qa.py")
        sys.exit(1)

    client = genai.Client(api_key=GEMINI_API_KEY)

    # Build/load sample list
    entries = build_sample_list()
    print(f"\nTotal: {len(entries)} documents to process")

    # Load progress
    done = load_progress()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    total_qa = 0
    errors = 0

    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            total_qa = sum(1 for _ in f)

    remaining = sum(1 for e in entries if file_hash(e["path"]) not in done)
    print(f"Already done: {len(done)} docs, {total_qa} Q&A pairs")
    print(f"Remaining: {remaining} docs")
    print(f"Estimated time: {remaining * DELAY / 60:.0f} minutes\n")

    if remaining == 0:
        print("All documents processed!")
        return

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
        for entry in entries:
            fh = file_hash(entry["path"])
            if fh in done:
                continue

            # Read document
            try:
                with open(entry["path"], encoding="utf-8") as f:
                    doc = json.load(f)
                text = doc.get("text", "")
            except Exception:
                done.add(fh)
                continue

            if len(text) > MAX_CHARS:
                text = text[:MAX_CHARS] + "..."

            prompt = PROMPT_TEMPLATE.format(text=text)

            try:
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt,
                    config={"temperature": 0.7, "max_output_tokens": 2000},
                )
                qa_pairs = parse_qa_response(response.text)

                for pair in qa_pairs:
                    out.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    total_qa += 1

                done.add(fh)

                if len(done) % 50 == 0:
                    save_progress(done)
                    out.flush()

                print(f"  [{len(done)}/{len(entries)}] {entry['source']}: +{len(qa_pairs)} pairs (total: {total_qa})", end="\r")

            except json.JSONDecodeError:
                errors += 1
                done.add(fh)
            except Exception as e:
                errors += 1
                err_msg = str(e)
                if "429" in err_msg or "quota" in err_msg.lower():
                    print(f"\n  Rate limited, waiting 60s...")
                    save_progress(done)
                    time.sleep(60)
                elif "block" in err_msg.lower() or "safety" in err_msg.lower():
                    done.add(fh)
                else:
                    print(f"\n  Error: {err_msg[:100]}")

            time.sleep(DELAY)

    save_progress(done)

    print(f"\n\n{'='*50}")
    print(f"Done!")
    print(f"Total Q&A pairs: {total_qa}")
    print(f"Errors: {errors}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
