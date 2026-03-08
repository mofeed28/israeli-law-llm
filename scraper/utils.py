import json
import time
import hashlib
from datetime import date, datetime
from pathlib import Path
from functools import wraps

from scraper.config import REQUEST_DELAY


class _JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)


def save_document(directory: Path, doc_id: str, data: dict) -> Path:
    """Save a document as JSON to the specified directory."""
    directory.mkdir(parents=True, exist_ok=True)
    safe_name = hashlib.md5(str(doc_id).encode()).hexdigest()
    filepath = directory / f"{safe_name}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, cls=_JSONEncoder)
    return filepath


def retry(max_retries=3, backoff=2.0):
    """Retry decorator with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    wait = backoff * (2 ** attempt)
                    print(f"  Retry {attempt + 1}/{max_retries} after {wait}s: {e}")
                    time.sleep(wait)
            raise last_exception
        return wrapper
    return decorator


def throttle():
    """Sleep for the configured delay between requests."""
    time.sleep(REQUEST_DELAY)
