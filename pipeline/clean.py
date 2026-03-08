"""Text cleaning functions for Israeli law documents."""

import json
import re
import unicodedata
from pathlib import Path

# Regex patterns compiled once
RE_NIQQUD = re.compile(r"[\u0591-\u05C7]")
RE_ZWCHARS = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]")
RE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
RE_MULTI_SPACES = re.compile(r"[^\S\n]+")  # multiple whitespace except newlines
RE_MULTI_NEWLINES = re.compile(r"\n{3,}")
RE_EMPTY_LINES = re.compile(r"\n[ \t]+\n")
RE_WIKI_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
RE_WIKI_TABLE = re.compile(r"\{\|.*?\|\}", re.DOTALL)
RE_WIKI_LINK = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]")
RE_EXT_LINK = re.compile(r"\[https?://[^\s\]]+(?: ([^\]]*))?\]")
RE_HTML_TAG = re.compile(r"<[^>]+>")
RE_HEBREW_LATIN_ADJ = re.compile(r"([\u0590-\u05FF])([a-zA-Z])")
RE_LATIN_HEBREW_ADJ = re.compile(r"([a-zA-Z])([\u0590-\u05FF])")

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
CLEANED_DIR = BASE_DIR / "data" / "cleaned"


def clean_hebrew_text(text: str) -> str:
    """General Hebrew text cleaning."""
    text = unicodedata.normalize("NFC", text)
    text = RE_NIQQUD.sub("", text)
    text = RE_ZWCHARS.sub("", text)
    text = RE_CONTROL.sub("", text)
    # Normalize Hebrew punctuation
    text = text.replace("\u05BE", "-")   # maqaf → hyphen
    text = text.replace("\u05F3", "'")   # geresh → apostrophe
    text = text.replace("\u05F4", '"')   # gershayim → quote
    # Space between adjacent Hebrew/Latin
    text = RE_HEBREW_LATIN_ADJ.sub(r"\1 \2", text)
    text = RE_LATIN_HEBREW_ADJ.sub(r"\1 \2", text)
    # Collapse whitespace
    text = RE_MULTI_SPACES.sub(" ", text)
    text = RE_EMPTY_LINES.sub("\n\n", text)
    text = RE_MULTI_NEWLINES.sub("\n\n", text)
    text = text.strip()
    return text


def clean_court_text(text: str) -> str:
    """Aggressive whitespace normalization for court documents."""
    # Strip each line, remove pure-whitespace lines
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped:
            lines.append(stripped)
        else:
            lines.append("")
    text = "\n".join(lines)
    # Collapse multiple blank lines
    text = RE_MULTI_NEWLINES.sub("\n\n", text)
    return text


def fix_wikitext(text: str) -> str:
    """Remove wikitext markup: templates, tables, links, HTML."""
    # Iteratively remove nested templates
    prev = None
    while prev != text:
        prev = text
        text = RE_WIKI_TEMPLATE.sub("", text)
    # Remove wiki tables
    text = RE_WIKI_TABLE.sub("", text)
    # Convert wiki links [[target|display]] → display
    text = RE_WIKI_LINK.sub(r"\1", text)
    # Convert external links [url display] → display
    text = RE_EXT_LINK.sub(lambda m: m.group(1) or "", text)
    # Strip HTML tags
    text = RE_HTML_TAG.sub("", text)
    # Remove leftover markup characters
    text = text.replace("'''", "").replace("''", "")
    text = text.replace("==", "")
    return text


def clean_document(doc: dict, source: str) -> dict | None:
    """Clean a single document based on its source. Returns None to skip."""
    if source == "knesset":
        return None

    title = doc.get("title", "")

    if source == "court":
        raw_text = doc.get("text", "")
        if not raw_text:
            return None
        text = clean_court_text(raw_text)
        text = clean_hebrew_text(text)
        meta = {
            k: doc.get(k)
            for k in ("case_id", "case_num", "date", "year", "type",
                       "judges", "parties", "court", "division")
            if doc.get(k) is not None
        }
    elif source in ("kolzchut", "wikisource_laws"):
        # Prefer wikitext field and re-strip it
        raw_text = doc.get("wikitext", "") or doc.get("text", "")
        if not raw_text:
            return None
        text = fix_wikitext(raw_text)
        text = clean_hebrew_text(text)
        meta = {
            "categories": doc.get("categories"),
            "url": doc.get("url"),
        }
        # Remap source name
        if source == "wikisource_laws":
            source = "wikisource"
    else:
        return None

    if not text:
        return None

    return {
        "text": text,
        "source": source,
        "title": title,
        "meta": meta,
    }


def load_docs(source_dir: Path):
    """Yield (filename, doc_dict) from a raw source directory."""
    for f in sorted(source_dir.glob("*.json")):
        try:
            with open(f, encoding="utf-8") as fh:
                yield f.name, json.load(fh)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue


def run_cleaning(batch_size: int = 10000):
    """Clean all raw sources and write to data/cleaned/."""
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    sources = ["court", "kolzchut", "wikisource_laws"]
    stats = {}

    for source in sources:
        source_dir = RAW_DIR / source
        if not source_dir.exists():
            print(f"  Skipping {source}: directory not found")
            continue

        out_dir = CLEANED_DIR / source
        out_dir.mkdir(parents=True, exist_ok=True)

        total = 0
        kept = 0
        batch = []

        for fname, doc in load_docs(source_dir):
            total += 1
            cleaned = clean_document(doc, source)
            if cleaned is not None:
                batch.append((fname, cleaned))
                kept += 1

            if len(batch) >= batch_size:
                _write_batch(batch, out_dir)
                batch = []

            if total % 50000 == 0:
                print(f"    {source}: processed {total}...")

        if batch:
            _write_batch(batch, out_dir)

        stats[source] = {"total": total, "kept": kept, "dropped": total - kept}
        print(f"  {source}: {kept}/{total} kept ({total - kept} dropped)")

    return stats


def _write_batch(batch: list, out_dir: Path):
    """Write a batch of cleaned docs to JSON files."""
    for fname, doc in batch:
        out_path = out_dir / fname
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False)
