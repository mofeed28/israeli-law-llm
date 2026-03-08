"""Knesset legislation scraper using OData API with Scrapling Fetcher."""

from scrapling import Fetcher
from tqdm import tqdm

from scraper.config import KNESSET_LAWS_ENDPOINT, KNESSET_DIR, USER_AGENT
from scraper.utils import save_document, retry, throttle


_fetcher = None


def get_fetcher():
    global _fetcher
    if _fetcher is None:
        _fetcher = Fetcher()
    return _fetcher

ODATA_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json",
}


@retry()
def fetch_laws_page(skip: int = 0, top: int = 100) -> list[dict]:
    """Fetch a page of laws from the Knesset OData API."""
    url = f"{KNESSET_LAWS_ENDPOINT}?$top={top}&$skip={skip}&$format=json"
    response = get_fetcher().get(url, headers=ODATA_HEADERS)
    data = response.json()
    return data.get("value", data.get("d", {}).get("results", []))


def get_all_laws() -> list[dict]:
    """Paginate through all laws in the Knesset OData API."""
    all_laws = []
    skip = 0
    top = 100

    print("Fetching Knesset laws...")
    while True:
        page = fetch_laws_page(skip=skip, top=top)
        if not page:
            break
        all_laws.extend(page)
        print(f"  Fetched {len(all_laws)} laws so far...")
        skip += top
        throttle()

    return all_laws


def extract_law_document(law: dict) -> dict:
    """Extract and normalize a law record into our document format."""
    law_id = str(law.get("LawID", law.get("Id", "")))
    name = law.get("Name", law.get("KNS_LawName", ""))
    type_desc = law.get("TypeDesc", law.get("SubTypeDesc", ""))
    pub_date = law.get("PublicationDate", law.get("LastUpdatedDate", ""))

    # Build a text representation
    text_parts = []
    if name:
        text_parts.append(f"שם החוק: {name}")
    if type_desc:
        text_parts.append(f"סוג: {type_desc}")
    if pub_date:
        text_parts.append(f"תאריך פרסום: {pub_date}")

    # Include any additional text fields
    for key in ("LawDesc", "Description", "Text", "SubTypeDesc", "StatusDesc"):
        val = law.get(key)
        if val and str(val) not in text_parts:
            text_parts.append(f"{key}: {val}")

    return {
        "law_id": law_id,
        "title": name,
        "type": type_desc,
        "publication_date": pub_date,
        "text": "\n".join(text_parts),
        "raw_fields": law,
        "source": "knesset",
    }


def scrape():
    """Main entry point: scrape all Knesset legislation."""
    laws = get_all_laws()
    print(f"Found {len(laws)} laws total")

    saved = 0
    for law in tqdm(laws, desc="Knesset"):
        try:
            doc = extract_law_document(law)
            if doc["title"]:
                save_document(KNESSET_DIR, doc["law_id"] or doc["title"], doc)
                saved += 1
        except Exception as e:
            print(f"\n  Error processing law: {e}")

    print(f"Saved {saved}/{len(laws)} Knesset laws")
