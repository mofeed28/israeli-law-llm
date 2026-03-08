"""Hebrew Wikisource scraper for full Israeli law texts.

Scrapes the "Open Book of Laws" (ספר החוקים הפתוח) project which contains
5,945+ full law and regulation texts maintained by Hasadna/Wikimedia Israel.
Uses the MediaWiki API (same approach as Kol-Zchut but much less aggressive
rate limiting needed).
"""

import hashlib
import re
import sys
import time
from urllib.parse import urlencode

from scrapling import Fetcher
from tqdm import tqdm

from scraper.config import WIKISOURCE_API_URL, WIKISOURCE_DIR
from scraper.utils import save_document, retry

# Fix Windows console encoding for Hebrew output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


_fetcher = None
DELAY = 1  # Wikisource is more permissive than Kolzchut


def get_fetcher():
    global _fetcher
    if _fetcher is None:
        _fetcher = Fetcher()
    return _fetcher


@retry(max_retries=3, backoff=5.0)
def api_request(params: dict) -> dict:
    """Make a request to the Hebrew Wikisource MediaWiki API."""
    params.setdefault("format", "json")
    url = f"{WIKISOURCE_API_URL}?{urlencode(params)}"
    response = get_fetcher().get(url)
    if response.status != 200:
        raise Exception(f"HTTP {response.status}")
    return response.json()


def get_category_pages(category: str) -> list[str]:
    """Get all page titles in a category (and subcategories)."""
    titles = []
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmlimit": "500",
        "cmtype": "page",
    }
    continue_token = None

    while True:
        if continue_token:
            params["cmcontinue"] = continue_token

        data = api_request(params)
        members = data.get("query", {}).get("categorymembers", [])
        titles.extend(m["title"] for m in members)

        if "continue" in data:
            continue_token = data["continue"].get("cmcontinue")
        else:
            break
        time.sleep(0.5)

    return titles


def get_all_law_pages() -> list[str]:
    """Get all law page titles from Wikisource categories."""
    print("  Enumerating law pages from Wikisource categories...")

    # Main categories for Israeli laws on Hebrew Wikisource
    categories = [
        "ספר_החוקים_הפתוח",
        "חוקי_מדינת_ישראל",
        "תקנות_מדינת_ישראל",
        "חוקי_יסוד",
        "פקודות_מנדטוריות",
    ]

    all_titles = set()
    for cat in categories:
        try:
            pages = get_category_pages(cat)
            print(f"    {cat}: {len(pages)} pages")
            all_titles.update(pages)
        except Exception as e:
            print(f"    {cat}: error - {e}")
        time.sleep(DELAY)

    # Also try allpages in the main namespace with common law prefixes
    for prefix in ["חוק ", "תקנות ", "פקודת ", "צו "]:
        try:
            params = {
                "action": "query",
                "list": "allpages",
                "apprefix": prefix,
                "aplimit": "500",
                "apnamespace": "0",
            }
            cont = None
            while True:
                if cont:
                    params["apcontinue"] = cont
                data = api_request(params)
                pages = data.get("query", {}).get("allpages", [])
                new = [p["title"] for p in pages]
                all_titles.update(new)
                if "continue" in data:
                    cont = data["continue"].get("apcontinue")
                else:
                    break
                time.sleep(0.5)
            print(f"    prefix '{prefix}': found pages (total now: {len(all_titles)})")
        except Exception as e:
            print(f"    prefix '{prefix}': error - {e}")

    return sorted(all_titles)


@retry(max_retries=2, backoff=3.0)
def get_page_content(title: str) -> dict | None:
    """Get the full wikitext content of a law page."""
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

    if len(clean_text) < 50:
        return None

    return {
        "title": title,
        "wikitext": wikitext,
        "text": clean_text,
        "categories": categories,
        "source": "wikisource",
        "url": f"https://he.wikisource.org/wiki/{title.replace(' ', '_')}",
    }


def strip_wikitext(text: str) -> str:
    """Convert wikitext to plain text."""
    text = re.sub(r"\{\{[^}]*\}\}", "", text)
    text = re.sub(r"\[\[[^|\]]*\|([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _already_downloaded(title: str) -> bool:
    """Check if a page was already downloaded."""
    safe_name = hashlib.md5(title.encode()).hexdigest()
    return (WIKISOURCE_DIR / f"{safe_name}.json").exists()


def scrape():
    """Main entry point: scrape all Israeli law texts from Hebrew Wikisource."""
    titles = get_all_law_pages()
    already = sum(1 for t in titles if _already_downloaded(t))
    remaining = [t for t in titles if not _already_downloaded(t)]
    print(f"  Found {len(titles)} unique law pages, {already} already downloaded")
    print(f"  Downloading {len(remaining)} remaining pages...")

    saved = 0
    for title in tqdm(remaining, desc="Wikisource laws"):
        try:
            doc = get_page_content(title)
            if doc:
                save_document(WIKISOURCE_DIR, title, doc)
                saved += 1
        except Exception as e:
            print(f"\n  Error on '{title}': {e}")
        time.sleep(DELAY)

    print(f"Saved {saved}/{len(remaining)} new law texts from Wikisource "
          f"(total: {already + saved})")
