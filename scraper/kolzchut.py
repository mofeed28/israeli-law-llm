"""Kol-Zchut (כל-זכות) scraper using MediaWiki API with Scrapling Fetcher.

Handles aggressive rate limiting by:
- Saving progress (page list + downloaded pages) to disk
- Using longer delays between requests
- Cooling down between batches to avoid IP blocks
"""

import json
import re
import time
from pathlib import Path
from urllib.parse import urlencode
from scrapling import Fetcher
from tqdm import tqdm

from scraper.config import KOLZCHUT_API_URL, KOLZCHUT_DIR
from scraper.utils import save_document, retry


_fetcher = None
PROGRESS_FILE = KOLZCHUT_DIR / "_progress.json"
BATCH_SIZE = 15  # requests before a cooldown
DELAY = 3  # seconds between requests
COOLDOWN = 60  # seconds between batches


def get_fetcher():
    global _fetcher
    if _fetcher is None:
        _fetcher = Fetcher()
    return _fetcher


def load_progress() -> dict:
    """Load saved progress from disk."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"titles": [], "continue_token": None, "listing_done": False, "downloaded": []}


def save_progress(progress: dict):
    """Save progress to disk."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


class RateLimitError(Exception):
    pass


@retry(max_retries=3, backoff=30.0)
def api_request(params: dict) -> dict:
    """Make a request to the Kol-Zchut MediaWiki API."""
    params.setdefault("format", "json")
    url = f"{KOLZCHUT_API_URL}?{urlencode(params)}"
    response = get_fetcher().get(url)
    status = response.status
    if status in (403, 429):
        raise RateLimitError(f"Rate limited (HTTP {status})")
    if status != 200:
        raise Exception(f"HTTP {status}")
    return response.json()


def _fetch_page_range(params: dict, progress: dict, titles: list[str]) -> bool:
    """Fetch pages for a given param set. Returns True if completed, False if blocked."""
    continue_token = None
    request_count = 0
    consecutive_failures = 0

    while True:
        if continue_token:
            params["apcontinue"] = continue_token

        try:
            data = api_request(params)
            consecutive_failures = 0
        except RateLimitError:
            consecutive_failures += 1
            if consecutive_failures >= 2:
                print(f"\n  Persistent block. Saving {len(titles)} pages and moving on.")
                progress["titles"] = titles
                save_progress(progress)
                return False
            print(f"\n  Rate limited. Cooling down {COOLDOWN}s...")
            progress["titles"] = titles
            save_progress(progress)
            time.sleep(COOLDOWN)
            continue

        pages = data.get("query", {}).get("allpages", [])
        titles.extend(p["title"] for p in pages)
        request_count += 1
        print(f"  Enumerated {len(titles)} pages ({request_count} requests)")

        if "continue" in data:
            continue_token = data["continue"].get("apcontinue")
        else:
            return True

        if request_count % 5 == 0:
            progress["titles"] = titles
            progress["continue_token"] = continue_token
            save_progress(progress)

        # Cooldown between batches
        if request_count % BATCH_SIZE == 0:
            print(f"  Batch complete. Cooling down {COOLDOWN}s...")
            progress["titles"] = titles
            progress["continue_token"] = continue_token
            save_progress(progress)
            time.sleep(COOLDOWN)
        else:
            time.sleep(DELAY)

    return True


def get_all_pages(progress: dict) -> list[str]:
    """Get all page titles using multiple ranges to work around WAF blocks."""
    if progress["listing_done"]:
        print(f"  Page list already complete: {len(progress['titles'])} pages")
        return progress["titles"]

    titles = progress["titles"]

    # Hebrew letter ranges to try — if one range gets blocked, skip to next
    # Each range starts from a different Hebrew letter
    start_letters = [None, "ש", "ת"]  # None = start from beginning

    for start in start_letters:
        params = {
            "action": "query",
            "list": "allpages",
            "aplimit": "500",
            "apnamespace": "0",
        }
        if start:
            params["apfrom"] = start

        # Skip ranges we already covered
        if start is None and len(titles) > 0:
            print(f"  Skipping initial range (already have {len(titles)} pages)")
            continue

        print(f"  Fetching pages from: {start.encode('utf-8') if start else 'beginning'}...")
        completed = _fetch_page_range(params, progress, titles)
        if completed:
            break  # Got to the end
        time.sleep(COOLDOWN)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for t in titles:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    titles = unique

    progress["titles"] = titles
    progress["listing_done"] = True
    save_progress(progress)
    print(f"  Page listing complete: {len(titles)} unique pages")
    return titles


@retry(max_retries=2, backoff=5.0)
def get_page_content(title: str) -> dict | None:
    """Get the full wikitext content of a page."""
    data = api_request({
        "action": "parse",
        "page": title.replace(" ", "_"),
        "prop": "wikitext|categories",
    })
    parse = data.get("parse")
    if not parse:
        return None

    wikitext = parse.get("wikitext", {}).get("*", "")
    categories = [c["*"] for c in parse.get("categories", [])]
    clean_text = strip_wikitext(wikitext)

    return {
        "title": title,
        "wikitext": wikitext,
        "text": clean_text,
        "categories": categories,
        "source": "kolzchut",
        "url": f"https://www.kolzchut.org.il/he/{title.replace(' ', '_')}",
    }


def strip_wikitext(text: str) -> str:
    """Basic wikitext to plain text conversion."""
    text = re.sub(r"\{\{[^}]*\}\}", "", text)
    text = re.sub(r"\[\[[^|\]]*\|([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def scrape():
    """Main entry point: scrape all Kol-Zchut legal articles."""
    progress = load_progress()
    downloaded_set = set(progress["downloaded"])

    print("Fetching Kol-Zchut page list...")
    titles = get_all_pages(progress)
    print(f"Found {len(titles)} pages total, {len(downloaded_set)} already downloaded")

    # Filter out titles with quotes — they trigger WAF 403 blocks
    remaining = [t for t in titles if t not in downloaded_set and '"' not in t]
    skipped = len(titles) - len(downloaded_set) - len(remaining)
    if skipped:
        print(f"  Skipping {skipped} pages with quotes in titles (WAF issue)")
    print(f"Downloading {len(remaining)} remaining pages...")

    saved = len(downloaded_set)
    request_count = 0

    for title in tqdm(remaining, desc="Kol-Zchut"):
        try:
            doc = get_page_content(title)
            if doc and len(doc["text"]) > 50:
                save_document(KOLZCHUT_DIR, title, doc)
                saved += 1
            downloaded_set.add(title)
            request_count += 1

            # Save progress every 10 pages
            if request_count % 10 == 0:
                progress["downloaded"] = list(downloaded_set)
                save_progress(progress)

            # Cooldown between batches
            if request_count % BATCH_SIZE == 0:
                progress["downloaded"] = list(downloaded_set)
                save_progress(progress)
                print(f"\n  Batch done ({saved} saved). Cooling down {COOLDOWN}s...")
                time.sleep(COOLDOWN)
            else:
                time.sleep(DELAY)

        except RateLimitError:
            print(f"\n  Rate limited. Saving progress and cooling down {COOLDOWN}s...")
            progress["downloaded"] = list(downloaded_set)
            save_progress(progress)
            time.sleep(COOLDOWN)
        except Exception as e:
            print(f"\n  Error on '{title}': {e}")
            time.sleep(DELAY)

    progress["downloaded"] = list(downloaded_set)
    save_progress(progress)
    print(f"Saved {saved}/{len(titles)} Kol-Zchut documents")
