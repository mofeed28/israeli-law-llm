"""
Generate Q&A pairs from Israeli legal documents using Gemini 2.0 Flash.
Samples documents from filtered data, sends to Gemini, saves as instruction JSONL.
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

# Sampling config
SAMPLE_COURT = 2000
SAMPLE_KOLZCHUT = 5347      # All of them
SAMPLE_WIKISOURCE = 1000
MIN_WORDS = 200
MAX_CHARS = 8000  # Truncate long docs to save tokens

# Rate limiting
REQUESTS_PER_MINUTE = 15  # Free tier: 15 RPM
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
    """Load set of already-processed file hashes."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return set(data.get("done", []))
    return set()


def save_progress(done_set):
    """Save progress to disk."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({"done": list(done_set)}, f)


def file_hash(path):
    """Quick hash for tracking progress."""
    return hashlib.md5(str(path).encode()).hexdigest()[:12]


def sample_documents():
    """Sample documents from each source (fast: sample paths first, read later)."""
    random.seed(42)
    samples = []

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

        # Get file list fast, sample paths first
        files = list(source_dir.glob("*.json"))
        print(f"  {source}: {len(files)} files found, sampling {min(n, len(files))}...")
        selected_files = random.sample(files, min(n * 2, len(files)))  # Oversample to account for short docs

        count = 0
        for f in selected_files:
            if count >= n:
                break
            try:
                with open(f, encoding="utf-8") as fh:
                    doc = json.load(fh)
                text = doc.get("text", "")
                if len(text.split()) >= MIN_WORDS:
                    samples.append((f, source, text))
                    count += 1
            except Exception:
                continue

        print(f"  {source}: {count} docs sampled")

    random.shuffle(samples)
    return samples


def parse_qa_response(response_text):
    """Parse Gemini's JSON response into Q&A pairs."""
    text = response_text.strip()
    # Remove markdown code fences if present
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
        print("  Then run: set GEMINI_API_KEY=your_key_here")
        sys.exit(1)

    client = genai.Client(api_key=GEMINI_API_KEY)

    print("Sampling documents...")
    samples = sample_documents()
    print(f"Total: {len(samples)} documents to process\n")

    done = load_progress()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    total_qa = 0
    errors = 0
    skipped = 0

    # Count existing Q&A pairs
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            total_qa = sum(1 for _ in f)
        print(f"Resuming: {total_qa} Q&A pairs already generated, {len(done)} docs processed\n")

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
        for i, (filepath, source, text) in enumerate(samples):
            fh = file_hash(filepath)
            if fh in done:
                skipped += 1
                continue

            # Truncate long documents
            if len(text) > MAX_CHARS:
                text = text[:MAX_CHARS] + "..."

            prompt = PROMPT_TEMPLATE.format(text=text)

            try:
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt,
                    config={
                        "temperature": 0.7,
                        "max_output_tokens": 2000,
                    },
                )
                qa_pairs = parse_qa_response(response.text)

                for pair in qa_pairs:
                    out.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    total_qa += 1

                done.add(fh)

                if (len(done) - skipped) % 50 == 0:
                    save_progress(done)
                    out.flush()

                processed = len(done)
                print(f"  [{processed}/{len(samples)}] {source}: +{len(qa_pairs)} pairs (total: {total_qa})", end="\r")

            except json.JSONDecodeError:
                errors += 1
            except Exception as e:
                errors += 1
                err_msg = str(e)
                if "429" in err_msg or "quota" in err_msg.lower():
                    print(f"\n  Rate limited, waiting 60s...")
                    time.sleep(60)
                elif "block" in err_msg.lower() or "safety" in err_msg.lower():
                    done.add(fh)  # Skip blocked content
                else:
                    print(f"\n  Error: {err_msg[:100]}")

            time.sleep(DELAY)

    save_progress(done)

    print(f"\n\n{'='*50}")
    print(f"Done!")
    print(f"Total Q&A pairs: {total_qa}")
    print(f"Errors/skipped: {errors}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
