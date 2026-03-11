"""
Helper for Claude Code Q&A generation.
Claude Code reads docs, generates Q&A, and this script handles I/O and progress.

Usage from Claude Code:
    python qa_helper.py next 10       # Get next 10 unprocessed docs
    python qa_helper.py save          # Save Q&A pairs (reads from stdin)
    python qa_helper.py status        # Show progress
"""

import json
import sys
import hashlib
from pathlib import Path

OUTPUT_FILE = Path("data/training/instructions.jsonl")
PROGRESS_FILE = Path("data/training/qa_progress.json")
SAMPLE_LIST_FILE = Path("data/training/sampled_docs.json")

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stdin.reconfigure(encoding="utf-8", errors="replace")


def file_hash(path):
    return hashlib.md5(str(path).encode()).hexdigest()[:12]


def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f).get("done", []))
    return set()


def save_progress(done_set):
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({"done": list(done_set)}, f)


def cmd_status():
    if not SAMPLE_LIST_FILE.exists():
        print("Sample list not built yet. Run generate_qa.py first to build it.")
        return

    with open(SAMPLE_LIST_FILE, "r", encoding="utf-8") as f:
        entries = json.load(f)

    done = load_progress()
    total_qa = 0
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            total_qa = sum(1 for _ in f)

    remaining = sum(1 for e in entries if file_hash(e["path"]) not in done)
    print(f"Total docs: {len(entries)}")
    print(f"Processed: {len(done)}")
    print(f"Remaining: {remaining}")
    print(f"Q&A pairs generated: {total_qa}")


def cmd_next(n=10):
    with open(SAMPLE_LIST_FILE, "r", encoding="utf-8") as f:
        entries = json.load(f)

    done = load_progress()
    batch = []

    for entry in entries:
        if len(batch) >= n:
            break
        fh = file_hash(entry["path"])
        if fh in done:
            continue

        try:
            with open(entry["path"], encoding="utf-8") as f:
                doc = json.load(f)
            text = doc.get("text", "")
            if len(text) > 6000:
                text = text[:6000]
            batch.append({
                "path": entry["path"],
                "source": entry["source"],
                "title": doc.get("title", ""),
                "text": text,
            })
        except Exception:
            # Mark broken files as done
            done.add(fh)
            continue

    if not batch:
        print("ALL_DONE")
        return

    # Output as JSON for Claude Code to read
    print(json.dumps(batch, ensure_ascii=False, indent=2))


def cmd_save(qa_json_str):
    """Save Q&A pairs and mark docs as processed."""
    data = json.loads(qa_json_str)
    pairs = data.get("pairs", [])
    paths = data.get("processed_paths", [])

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    done = load_progress()
    for p in paths:
        done.add(file_hash(p))
    save_progress(done)

    total_qa = 0
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        total_qa = sum(1 for _ in f)

    print(f"Saved {len(pairs)} Q&A pairs from {len(paths)} docs (total: {total_qa})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python qa_helper.py [status|next N|save JSON]")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "status":
        cmd_status()
    elif cmd == "next":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        cmd_next(n)
    elif cmd == "save":
        qa_json = sys.argv[2] if len(sys.argv) > 2 else sys.stdin.read()
        cmd_save(qa_json)
    else:
        print(f"Unknown command: {cmd}")
