"""Run NER PII scrubbing in chunks to avoid OOM.

Usage: python run_ner_chunk.py [start_index] [chunk_size]
Processes chunk_size files starting from start_index in data/cleaned/court/.
Skips files that already have [שם] in their text.
"""

import gc
import json
import sys
from pathlib import Path

def main():
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    chunk_size = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    source = sys.argv[3] if len(sys.argv) > 3 else "court"

    base = Path(__file__).resolve().parent
    source_dir = base / "data" / "cleaned" / source

    files = sorted(source_dir.glob("*.json"))
    end = min(start + chunk_size, len(files))
    chunk_files = files[start:end]

    print(f"Processing {source} docs {start}-{end} ({len(chunk_files)} files)", flush=True)

    # Count already done
    to_process = []
    for f in chunk_files:
        try:
            with open(f, encoding="utf-8") as fh:
                doc = json.load(fh)
            if "[שם]" in doc["text"]:
                continue
            to_process.append((f, doc))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

    print(f"  {len(chunk_files) - len(to_process)} already done, {len(to_process)} to process", flush=True)

    if not to_process:
        print("  Nothing to do.", flush=True)
        return

    # Load NER model
    import torch
    from pipeline.pii import NERScrubber, scrub_regex

    scrubber = NERScrubber(device=0)
    processed = 0

    for f, doc in to_process:
        text = scrub_regex(doc["text"])
        text = scrubber.scrub_text(text)

        result = dict(doc)
        result["text"] = text
        with open(f, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False)

        processed += 1
        if processed % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()
        if processed % 500 == 0:
            print(f"  {processed}/{len(to_process)} done...", flush=True)

    print(f"  Finished: {processed} docs processed", flush=True)


if __name__ == "__main__":
    main()
