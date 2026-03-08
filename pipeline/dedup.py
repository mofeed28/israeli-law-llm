"""Exact and near-duplicate removal."""

import hashlib
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
FILTERED_DIR = BASE_DIR / "data" / "filtered"

RE_WHITESPACE = re.compile(r"\s+")
MINHASH_THRESHOLD = 0.7
MINHASH_NUM_PERM = 128
NGRAM_SIZE = 5


def normalize_for_hash(text: str) -> str:
    """Collapse all whitespace for consistent hashing."""
    return RE_WHITESPACE.sub(" ", text).strip().lower()


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_for_hash(text).encode("utf-8")).hexdigest()


def get_shingles(text: str) -> set:
    """Character n-gram shingles for MinHash."""
    normalized = normalize_for_hash(text)
    if len(normalized) < NGRAM_SIZE:
        return {normalized}
    return {normalized[i:i + NGRAM_SIZE] for i in range(len(normalized) - NGRAM_SIZE + 1)}


def run_dedup():
    """Deduplicate filtered documents in place."""
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        print("  Warning: datasketch not installed. Skipping near-dedup.")
        print("  Install with: pip install datasketch")
        MinHash = None
        MinHashLSH = None

    stats = {}

    for source_dir in sorted(FILTERED_DIR.iterdir()):
        if not source_dir.is_dir():
            continue
        source = source_dir.name

        # Load all docs
        docs = []
        for f in sorted(source_dir.glob("*.json")):
            try:
                with open(f, encoding="utf-8") as fh:
                    doc = json.load(fh)
                docs.append((f, doc))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

        total = len(docs)

        # Phase 1: Exact dedup
        seen_hashes = {}
        exact_dups = []
        unique_docs = []

        for f, doc in docs:
            h = text_hash(doc["text"])
            if h in seen_hashes:
                exact_dups.append(f)
            else:
                seen_hashes[h] = f
                unique_docs.append((f, doc))

        # Delete exact duplicates
        for f in exact_dups:
            f.unlink()

        # Phase 2: Near-dedup with MinHash LSH
        near_dups = []
        if MinHash is not None and len(unique_docs) > 1:
            lsh = MinHashLSH(threshold=MINHASH_THRESHOLD, num_perm=MINHASH_NUM_PERM)
            minhashes = {}

            for f, doc in unique_docs:
                m = MinHash(num_perm=MINHASH_NUM_PERM)
                for shingle in get_shingles(doc["text"]):
                    m.update(shingle.encode("utf-8"))
                minhashes[f.name] = (f, m)

                result = lsh.query(m)
                if result:
                    near_dups.append(f)
                else:
                    lsh.insert(f.name, m)

            # Delete near-duplicates
            for f in near_dups:
                f.unlink()

        kept = total - len(exact_dups) - len(near_dups)
        stats[source] = {
            "total": total,
            "exact_dups": len(exact_dups),
            "near_dups": len(near_dups),
            "kept": kept,
        }
        print(f"  {source}: {kept}/{total} kept "
              f"(exact: -{len(exact_dups)}, near: -{len(near_dups)})")

    return stats
