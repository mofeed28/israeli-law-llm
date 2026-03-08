"""Corpus and tokenizer analysis."""

import json
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
FILTERED_DIR = BASE_DIR / "data" / "filtered"
TRAINING_DIR = BASE_DIR / "data" / "training"


def run_analysis(sample_size: int = 1000):
    """Analyze corpus stats and tokenization efficiency."""
    print("\n--- Corpus Statistics ---")
    source_stats = {}
    all_lengths = []

    for source_dir in sorted(FILTERED_DIR.iterdir()):
        if not source_dir.is_dir():
            continue
        source = source_dir.name
        files = list(source_dir.glob("*.json"))
        char_lengths = []
        word_lengths = []

        # Sample for efficiency
        sample_files = random.sample(files, min(sample_size, len(files)))
        for f in sample_files:
            try:
                with open(f, encoding="utf-8") as fh:
                    doc = json.load(fh)
                text = doc["text"]
                char_lengths.append(len(text))
                word_lengths.append(len(text.split()))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

        if char_lengths:
            char_lengths.sort()
            word_lengths.sort()
            mid = len(char_lengths) // 2
            source_stats[source] = {
                "total_files": len(files),
                "sampled": len(char_lengths),
                "chars_median": char_lengths[mid],
                "chars_mean": sum(char_lengths) // len(char_lengths),
                "chars_p10": char_lengths[len(char_lengths) // 10],
                "chars_p90": char_lengths[9 * len(char_lengths) // 10],
                "words_median": word_lengths[mid],
                "words_mean": sum(word_lengths) // len(word_lengths),
            }
            all_lengths.extend(char_lengths)

            s = source_stats[source]
            print(f"\n  {source} ({s['total_files']:,} docs, sampled {s['sampled']}):")
            print(f"    Chars — median: {s['chars_median']:,}, mean: {s['chars_mean']:,}, "
                  f"p10: {s['chars_p10']:,}, p90: {s['chars_p90']:,}")
            print(f"    Words — median: {s['words_median']:,}, mean: {s['words_mean']:,}")

    # Tokenizer analysis
    print("\n--- Tokenizer Analysis ---")
    _run_tokenizer_analysis(sample_size)

    return source_stats


def _run_tokenizer_analysis(sample_size: int = 500):
    """Test tokenization efficiency with common Hebrew-capable tokenizers."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("  transformers not installed, skipping tokenizer analysis")
        return

    # Collect sample texts
    texts = []
    for source_dir in sorted(FILTERED_DIR.iterdir()):
        if not source_dir.is_dir():
            continue
        files = list(source_dir.glob("*.json"))
        sample = random.sample(files, min(sample_size // 3, len(files)))
        for f in sample:
            try:
                with open(f, encoding="utf-8") as fh:
                    doc = json.load(fh)
                texts.append(doc["text"][:5000])  # cap per doc for speed
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

    if not texts:
        print("  No texts to analyze")
        return

    total_chars = sum(len(t) for t in texts)
    total_words = sum(len(t.split()) for t in texts)

    # Test common base models people fine-tune for Hebrew
    model_ids = [
        "meta-llama/Llama-2-7b-hf",
        "mistralai/Mistral-7B-v0.1",
        "google/gemma-2b",
        "dicta-il/dictalm2.0",
    ]

    for model_id in model_ids:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            total_tokens = sum(len(tokenizer.encode(t)) for t in texts)
            chars_per_token = total_chars / total_tokens
            words_per_token = total_words / total_tokens
            tokens_per_word = total_tokens / total_words

            print(f"\n  {model_id}:")
            print(f"    {total_tokens:,} tokens for {len(texts)} samples")
            print(f"    {chars_per_token:.1f} chars/token, {tokens_per_word:.1f} tokens/word")
            print(f"    Estimated total tokens for full corpus: "
                  f"~{int(total_tokens / len(texts) * _count_total_docs()):,}")
        except Exception as e:
            print(f"\n  {model_id}: skipped ({e})")


def _count_total_docs() -> int:
    """Count total docs across all filtered sources."""
    total = 0
    for source_dir in FILTERED_DIR.iterdir():
        if source_dir.is_dir():
            total += sum(1 for _ in source_dir.glob("*.json"))
    return total
