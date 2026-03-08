"""Format cleaned/deduplicated docs for training with eval split and source balancing."""

import hashlib
import json
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
FILTERED_DIR = BASE_DIR / "data" / "filtered"
TRAINING_DIR = BASE_DIR / "data" / "training"

EVAL_RATIO = 0.02  # 2% held out for evaluation

# Upsample weights per source to balance court dominance.
# Court=1x, kolzchut=5x, wikisource=5x — brings law/rights texts closer
# to court volume without fully equalizing (court variety is still valuable).
SOURCE_UPSAMPLE = {
    "court": 1,
    "kolzchut": 5,
    "wikisource_laws": 5,
    "wikisource": 5,
}


def _deterministic_split(doc_id: str, eval_ratio: float = EVAL_RATIO) -> str:
    """Assign doc to train/eval deterministically based on hash.
    This ensures the same doc always lands in the same split,
    even if the pipeline is re-run with more data."""
    h = hashlib.md5(doc_id.encode()).hexdigest()
    # Use first 8 hex chars as fraction
    frac = int(h[:8], 16) / 0xFFFFFFFF
    return "eval" if frac < eval_ratio else "train"


def _load_all_docs() -> dict[str, list[dict]]:
    """Load all filtered docs grouped by source."""
    by_source = {}
    for source_dir in sorted(FILTERED_DIR.iterdir()):
        if not source_dir.is_dir():
            continue
        source = source_dir.name
        docs = []
        for f in sorted(source_dir.glob("*.json")):
            try:
                with open(f, encoding="utf-8") as fh:
                    doc = json.load(fh)
                doc["_id"] = f"{source}/{f.stem}"
                docs.append(doc)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
        by_source[source] = docs
    return by_source


def run_format():
    """Create train/eval JSONL with source balancing."""
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    train_path = TRAINING_DIR / "pretrain.jsonl"
    eval_path = TRAINING_DIR / "eval.jsonl"

    by_source = _load_all_docs()

    train_count = 0
    eval_count = 0
    source_stats = {}

    with open(train_path, "w", encoding="utf-8") as train_f, \
         open(eval_path, "w", encoding="utf-8") as eval_f:

        for source, docs in by_source.items():
            upsample = SOURCE_UPSAMPLE.get(source, 1)
            s_train = 0
            s_eval = 0

            for doc in docs:
                split = _deterministic_split(doc["_id"])
                line = json.dumps({"text": doc["text"]}, ensure_ascii=False) + "\n"

                if split == "eval":
                    eval_f.write(line)
                    eval_count += 1
                    s_eval += 1
                else:
                    for _ in range(upsample):
                        train_f.write(line)
                    train_count += upsample
                    s_train += 1

            source_stats[source] = {
                "raw_docs": len(docs),
                "train_docs": s_train,
                "train_rows": s_train * upsample,
                "eval_docs": s_eval,
                "upsample": upsample,
            }
            print(f"  {source}: {s_train} train (x{upsample}={s_train * upsample} rows), {s_eval} eval")

    # Shuffle train file (read, shuffle, rewrite)
    print("  Shuffling training data...")
    with open(train_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    random.seed(42)
    random.shuffle(lines)
    with open(train_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"\n  Train: {train_count:,} rows -> {train_path}")
    print(f"  Eval:  {eval_count:,} rows -> {eval_path}")
    return {"train": train_count, "eval": eval_count, "sources": source_stats}


def export_for_easy_dataset(output_dir: str | Path | None = None):
    """Export cleaned docs as individual .txt files for Easy Dataset import."""
    if output_dir is None:
        output_dir = TRAINING_DIR / "easy_dataset_export"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for source_dir in sorted(FILTERED_DIR.iterdir()):
        if not source_dir.is_dir():
            continue

        for f in sorted(source_dir.glob("*.json")):
            try:
                with open(f, encoding="utf-8") as fh:
                    doc = json.load(fh)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            title = doc.get("title", f.stem)
            safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)[:80]
            txt_name = f"{safe_title}_{f.stem[:8]}.txt"

            with open(output_dir / txt_name, "w", encoding="utf-8") as out:
                out.write(doc["text"])
            total += 1

    print(f"  Exported {total} .txt files to {output_dir}")
    return {"total": total, "path": str(output_dir)}
