"""PII detection and scrubbing for Israeli legal documents.

Two modes:
  - Regex: fast, catches ID numbers, phones, emails (always runs)
  - NER: Hebrew NER model to detect and replace person names (optional, GPU-accelerated)
"""

import gc
import json
import re
from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent.parent
CLEANED_DIR = BASE_DIR / "data" / "cleaned"

# === Regex patterns for Israeli PII ===

RE_TZ = re.compile(r"\b\d{9}\b")
RE_TZ_DASH = re.compile(r"\b\d{3}-\d{3}-\d{3}\b")
RE_PHONE_MOBILE = re.compile(r"\b0[5]\d[- ]?\d{3}[- ]?\d{4}\b")
RE_PHONE_LAND = re.compile(r"\b0[2-489][- ]?\d{7}\b")
RE_PHONE_INTL = re.compile(r"\b\+972[- ]?\d[- ]?\d{3}[- ]?\d{4}\b")
RE_EMAIL = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
RE_CC = re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b")

REGEX_PATTERNS = [
    (RE_TZ, "[מספר_זהות]"),
    (RE_TZ_DASH, "[מספר_זהות]"),
    (RE_PHONE_INTL, "[טלפון]"),
    (RE_PHONE_MOBILE, "[טלפון]"),
    (RE_PHONE_LAND, "[טלפון]"),
    (RE_EMAIL, "[אימייל]"),
    (RE_CC, "[מספר_כרטיס]"),
]

# NER config
NER_MODEL = "dicta-il/dictabert-ner"
BATCH_SIZE = 64
MAX_LENGTH = 512  # tokens


def scrub_regex(text: str) -> str:
    """Replace PII patterns with Hebrew placeholders."""
    for pattern, replacement in REGEX_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def scrub_document(doc: dict, use_ner: bool = False, ner_scrubber=None) -> dict:
    """Apply PII scrubbing to a document."""
    text = doc["text"]
    text = scrub_regex(text)
    if use_ner and ner_scrubber:
        text = ner_scrubber.scrub_text(text)
    doc = dict(doc)
    doc["text"] = text
    return doc


class NERScrubber:
    """Batched NER-based name scrubber using direct model inference."""

    def __init__(self, device: int = 0):
        from transformers import AutoTokenizer, AutoModelForTokenClassification

        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{device}")
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                if "5060" in name or "RTX" in name.upper():
                    self.device = torch.device(f"cuda:{i}")
                    break
            print(f"  NER using {self.device}: {torch.cuda.get_device_name(self.device.index)}")
        else:
            self.device = torch.device("cpu")
            print("  NER using CPU (no CUDA available)")

        self.tokenizer = AutoTokenizer.from_pretrained(NER_MODEL)
        self.model = AutoModelForTokenClassification.from_pretrained(NER_MODEL)
        self.model.to(self.device)
        self.model.eval()

        self.id2label = self.model.config.id2label

    def _find_person_spans(self, text: str, input_ids, offset_mapping, predictions) -> list:
        """Extract person name character spans from NER predictions."""
        spans = []
        current_start = None
        current_end = None

        for idx, (pred_id, offsets) in enumerate(zip(predictions, offset_mapping)):
            if offsets[0] == 0 and offsets[1] == 0:
                # Special token
                if current_start is not None:
                    spans.append((current_start, current_end))
                    current_start = None
                continue

            label = self.id2label[pred_id]

            if label in ("B-PER", "B-PERSON"):
                if current_start is not None:
                    spans.append((current_start, current_end))
                current_start = offsets[0]
                current_end = offsets[1]
            elif label in ("I-PER", "I-PERSON") and current_start is not None:
                current_end = offsets[1]
            else:
                if current_start is not None:
                    spans.append((current_start, current_end))
                    current_start = None

        if current_start is not None:
            spans.append((current_start, current_end))

        return spans

    def scrub_text(self, text: str) -> str:
        """Scrub person names from a single text using windowed tokenization."""
        if not text.strip():
            return text

        # Tokenize with offset mapping for the full text in windows
        all_spans = []
        stride = 400  # tokens overlap
        window = MAX_LENGTH - 2  # account for [CLS] and [SEP]

        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            return_tensors=None,
        )

        token_ids = encoding["input_ids"]
        offset_map = encoding["offset_mapping"]
        total_tokens = len(token_ids)

        start = 0
        while start < total_tokens:
            end = min(start + window, total_tokens)
            chunk_ids = [self.tokenizer.cls_token_id] + token_ids[start:end] + [self.tokenizer.sep_token_id]
            chunk_offsets = [(0, 0)] + offset_map[start:end] + [(0, 0)]

            input_tensor = torch.tensor([chunk_ids], device=self.device)

            with torch.no_grad():
                outputs = self.model(input_tensor)
                preds = outputs.logits.argmax(dim=-1)[0].cpu().tolist()

            spans = self._find_person_spans(text, chunk_ids, chunk_offsets, preds)
            all_spans.extend(spans)

            if end >= total_tokens:
                break
            start += window - stride

        # Deduplicate and merge overlapping spans
        if not all_spans:
            return text

        all_spans = list(set(all_spans))
        all_spans.sort()
        merged = [all_spans[0]]
        for s, e in all_spans[1:]:
            if s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

        # Replace in reverse order
        for s, e in reversed(merged):
            name = text[s:e].strip()
            if len(name) >= 2:  # skip single-char false positives
                text = text[:s] + "[שם]" + text[e:]

        return text

    def scrub_batch(self, texts: list[str]) -> list[str]:
        """Scrub a batch of texts with memory cleanup."""
        results = [self.scrub_text(t) for t in texts]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return results


def run_pii_scrub(use_ner: bool = True, gpu_device: int = 0):
    """Scrub PII from all cleaned documents in place."""
    ner_scrubber = None
    if use_ner:
        try:
            ner_scrubber = NERScrubber(device=gpu_device)
        except Exception as e:
            print(f"  Failed to load NER model: {e}")
            print("  Falling back to regex-only mode")
            use_ner = False

    stats = {}

    for source_dir in sorted(CLEANED_DIR.iterdir()):
        if not source_dir.is_dir():
            continue
        source = source_dir.name

        files = sorted(source_dir.glob("*.json"))
        total = len(files)
        scrubbed = 0

        # Process in batches for NER
        batch_files = []
        batch_docs = []
        skipped = 0

        for i, f in enumerate(files):
            try:
                with open(f, encoding="utf-8") as fh:
                    doc = json.load(fh)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            # Skip docs already processed by NER
            if use_ner and "[שם]" in doc["text"]:
                skipped += 1
                continue

            # Always regex first
            text = scrub_regex(doc["text"])
            doc_copy = dict(doc)
            doc_copy["text"] = text

            batch_files.append(f)
            batch_docs.append((doc, doc_copy))

            if len(batch_docs) >= BATCH_SIZE:
                scrubbed += _flush_ner_batch(batch_files, batch_docs, ner_scrubber, use_ner)
                batch_files = []
                batch_docs = []

            if (i + 1) % 2000 == 0:
                # Aggressively free memory
                if use_ner and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                print(f"    {source}: {i + 1}/{total} (skipped {skipped} already done)...", flush=True)

            # Reload model every 10K processed docs to prevent memory leak
            if use_ner and (i + 1) % 10000 == 0 and ner_scrubber is not None:
                del ner_scrubber.model
                del ner_scrubber.tokenizer
                del ner_scrubber
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                ner_scrubber = NERScrubber(device=gpu_device)

        if batch_docs:
            scrubbed += _flush_ner_batch(batch_files, batch_docs, ner_scrubber, use_ner)

        stats[source] = {"total": total, "modified": scrubbed}
        print(f"  {source}: {scrubbed}/{total} docs had PII replaced")

    return stats


def _flush_ner_batch(files, docs, ner_scrubber, use_ner):
    """Process and write a batch of docs."""
    scrubbed = 0

    if use_ner and ner_scrubber:
        texts = [d[1]["text"] for d in docs]
        scrubbed_texts = ner_scrubber.scrub_batch(texts)

        for f, (orig_doc, regex_doc), final_text in zip(files, docs, scrubbed_texts):
            result = dict(orig_doc)
            result["text"] = final_text
            if final_text != orig_doc["text"]:
                scrubbed += 1
            with open(f, "w", encoding="utf-8") as fh:
                json.dump(result, fh, ensure_ascii=False)
    else:
        for f, (orig_doc, regex_doc) in zip(files, docs):
            if regex_doc["text"] != orig_doc["text"]:
                scrubbed += 1
            with open(f, "w", encoding="utf-8") as fh:
                json.dump(regex_doc, fh, ensure_ascii=False)

    return scrubbed
