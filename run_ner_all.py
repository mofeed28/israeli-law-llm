"""Run NER in subprocess chunks to prevent OOM.

Each chunk runs as a separate Python process, so memory is fully
reclaimed between chunks.
"""

import subprocess
import sys

CHUNK_SIZE = 5000
TOTAL_COURT = 242463
SOURCES = [
    ("court", TOTAL_COURT),
    ("kolzchut", 10000),
    ("wikisource_laws", 10000),
]

for source, total in SOURCES:
    print(f"\n{'='*50}")
    print(f"NER scrubbing: {source}")
    print(f"{'='*50}", flush=True)

    start = 0
    while start < total:
        print(f"\n--- Chunk {start}-{start + CHUNK_SIZE} ---", flush=True)
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "run_ner_chunk.py",
             str(start), str(CHUNK_SIZE), source],
            cwd=str(__import__("pathlib").Path(__file__).parent),
        )
        if result.returncode != 0:
            print(f"  Chunk failed with code {result.returncode}", flush=True)
        start += CHUNK_SIZE

print("\n\nAll NER processing complete!", flush=True)
