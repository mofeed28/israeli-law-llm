"""Quality filters for cleaned documents."""

import json
import re
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CLEANED_DIR = BASE_DIR / "data" / "cleaned"
FILTERED_DIR = BASE_DIR / "data" / "filtered"

MIN_WORDS = 100
MIN_ALPHA_RATIO = 0.4
MIN_HEBREW_RATIO = 0.5         # at least 50% of alphabetic chars must be Hebrew
MAX_WORD_REP_RATIO = 0.5       # fraction of words in repeated word 5-grams
MAX_DUP_LINE_RATIO = 0.3       # fraction of chars in duplicated lines
MAX_SHORT_LINE_RATIO = 0.80    # fraction of lines under 30 chars
SHORT_LINE_THRESHOLD = 30
MAX_BOILERPLATE_RATIO = 0.4    # max fraction of text matching boilerplate patterns

# Hebrew Unicode range
RE_HEBREW = re.compile(r"[\u0590-\u05FF]")
RE_ALPHA = re.compile(r"[a-zA-Z\u0590-\u05FF\u0600-\u06FF]")

# Common boilerplate patterns in Israeli court docs
BOILERPLATE_PATTERNS = [
    re.compile(r"^[-_=]{3,}$", re.MULTILINE),                    # separator lines
    re.compile(r"^\s*עמוד \d+ מתוך \d+\s*$", re.MULTILINE),     # page X of Y
    re.compile(r"^\s*\d+\s*$", re.MULTILINE),                    # lone page numbers
    re.compile(r"בבית (?:המשפט|הדין).*?(?:בשבתו|בהרכב)", re.DOTALL),  # court header
    re.compile(r"(?:הוגש|נחתם) (?:ביום|בתאריך) [\d.]+"),        # filing date stamps
    re.compile(r"(?:המזכירות|מזכירות בית) ת(?:שלח|מציא)"),       # clerk instructions
]


def word_count(text: str) -> int:
    return len(text.split())


def alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(c.isalpha() for c in text) / len(text)


def hebrew_ratio(text: str) -> float:
    """Fraction of alphabetic characters that are Hebrew."""
    alpha_chars = RE_ALPHA.findall(text)
    if not alpha_chars:
        return 0.0
    hebrew_chars = RE_HEBREW.findall(text)
    return len(hebrew_chars) / len(alpha_chars)


def word_ngram_repetition(text: str, n: int = 5) -> float:
    """Fraction of words that appear in repeated word n-grams."""
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(ngrams)
    repeated_words = sum((c - 1) * n for c in counts.values() if c > 1)
    return min(repeated_words / len(words), 1.0)


def duplicate_line_ratio(text: str) -> float:
    """Fraction of total chars that are in duplicated lines."""
    lines = text.split("\n")
    line_counts = Counter(lines)
    dup_chars = sum(len(line) * (count - 1) for line, count in line_counts.items() if count > 1 and line.strip())
    total_chars = sum(len(line) for line in lines)
    if total_chars == 0:
        return 0.0
    return dup_chars / total_chars


def short_line_ratio(text: str) -> float:
    """Fraction of non-empty lines that are under SHORT_LINE_THRESHOLD chars."""
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        return 1.0
    short = sum(1 for l in lines if len(l.strip()) < SHORT_LINE_THRESHOLD)
    return short / len(lines)


def boilerplate_ratio(text: str) -> float:
    """Fraction of text that matches known boilerplate patterns."""
    if not text:
        return 0.0
    boilerplate_chars = 0
    for pattern in BOILERPLATE_PATTERNS:
        for match in pattern.finditer(text):
            boilerplate_chars += match.end() - match.start()
    return min(boilerplate_chars / len(text), 1.0)


FILTERS = [
    ("min_length", lambda doc: word_count(doc["text"]) >= MIN_WORDS),
    ("alpha_ratio", lambda doc: alpha_ratio(doc["text"]) >= MIN_ALPHA_RATIO),
    ("hebrew_ratio", lambda doc: hebrew_ratio(doc["text"]) >= MIN_HEBREW_RATIO),
    ("word_repetition", lambda doc: word_ngram_repetition(doc["text"]) <= MAX_WORD_REP_RATIO),
    ("duplicate_lines", lambda doc: duplicate_line_ratio(doc["text"]) <= MAX_DUP_LINE_RATIO),
    ("short_lines", lambda doc: short_line_ratio(doc["text"]) <= MAX_SHORT_LINE_RATIO),
    ("boilerplate", lambda doc: boilerplate_ratio(doc["text"]) <= MAX_BOILERPLATE_RATIO),
]


def filter_document(doc: dict) -> str | None:
    """Return None if doc passes all filters, or the name of the first failed filter."""
    for name, check in FILTERS:
        if not check(doc):
            return name
    return None


def run_filtering():
    """Filter all cleaned documents and save to data/filtered/."""
    FILTERED_DIR.mkdir(parents=True, exist_ok=True)
    stats = {}

    for source_dir in sorted(CLEANED_DIR.iterdir()):
        if not source_dir.is_dir():
            continue
        source = source_dir.name

        out_dir = FILTERED_DIR / source
        out_dir.mkdir(parents=True, exist_ok=True)

        rejection_counts = {name: 0 for name, _ in FILTERS}
        total = 0
        kept = 0

        for f in sorted(source_dir.glob("*.json")):
            try:
                with open(f, encoding="utf-8") as fh:
                    doc = json.load(fh)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            total += 1
            failed = filter_document(doc)

            if failed is None:
                with open(out_dir / f.name, "w", encoding="utf-8") as fh:
                    json.dump(doc, fh, ensure_ascii=False)
                kept += 1
            else:
                rejection_counts[failed] += 1

        stats[source] = {
            "total": total,
            "kept": kept,
            "rejections": {k: v for k, v in rejection_counts.items() if v > 0},
        }
        print(f"  {source}: {kept}/{total} kept")
        for name, count in rejection_counts.items():
            if count > 0:
                print(f"    rejected by {name}: {count}")

    return stats
