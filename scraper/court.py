"""Israeli court decisions from HuggingFace datasets.

Sources:
1. guychuk/case-law-israel — 10,558 judgments from various courts
2. LevMuchnik/SupremeCourtOfIsrael — 751,194 Supreme Court documents (5.3GB)

The Israeli court websites have aggressive anti-bot protection that blocks
all scraping attempts, so we use pre-collected datasets instead.
"""

from datasets import load_dataset
from tqdm import tqdm

from scraper.config import COURT_DIR
from scraper.utils import save_document


def scrape_guychuk():
    """Download court decisions from guychuk/case-law-israel (10K judgments)."""
    print("  Dataset: guychuk/case-law-israel")
    ds = load_dataset("guychuk/case-law-israel", split="judgments")
    print(f"  Loaded {len(ds)} court judgments")

    saved = 0
    for row in tqdm(ds, desc="guychuk court"):
        text = row.get("document_text", "")
        if not text or len(text) < 100:
            continue

        doc = {
            "judgment_id": row.get("judgment_id", ""),
            "title": row.get("title", ""),
            "case_number": row.get("name_number", ""),
            "date": row.get("doc_create_date", ""),
            "court_type": row.get("court_type_label", ""),
            "district": row.get("district_label", ""),
            "subject": row.get("publication_subject_label", ""),
            "judges": row.get("judges_str", ""),
            "text": text,
            "source": "court_guychuk",
        }

        doc_id = row.get("judgment_id") or row.get("name_number") or str(saved)
        save_document(COURT_DIR, f"gc_{doc_id}", doc)
        saved += 1

    return saved, len(ds)


def scrape_supreme():
    """Download Supreme Court documents from LevMuchnik/SupremeCourtOfIsrael (751K docs)."""
    print("  Dataset: LevMuchnik/SupremeCourtOfIsrael")
    print("  This is a large dataset (5.3GB) — download may take a while...")

    ds = load_dataset("LevMuchnik/SupremeCourtOfIsrael", split="train", streaming=True)

    saved = 0
    skipped = 0
    for row in tqdm(ds, desc="Supreme Court", total=751194):
        text = row.get("text", "")
        if not text or len(text) < 200:
            skipped += 1
            continue

        doc = {
            "case_id": row.get("case_id", ""),
            "case_num": row.get("CaseNum", row.get("case_nbr", "")),
            "case_name": row.get("CaseName", row.get("meta_case_nm", "")),
            "title": row.get("html_title", ""),
            "date": row.get("VerdictDt", row.get("meta_verdict_dt", "")),
            "year": row.get("Year", ""),
            "type": row.get("Type", ""),
            "judges": row.get("meta_judge", []),
            "parties": row.get("meta_side_nm", []),
            "lawyers": row.get("meta_lawyer_nm", []),
            "court": row.get("meta_court_nm", ""),
            "division": row.get("meta_mador_nm", ""),
            "text": text,
            "source": "court_supreme",
        }

        doc_id = row.get("case_id") or row.get("CaseNum") or str(saved)
        save_document(COURT_DIR, f"sc_{doc_id}", doc)
        saved += 1

        # Progress save every 10,000
        if saved % 10000 == 0:
            print(f"\n  Saved {saved} documents so far (skipped {skipped})...")

    return saved, saved + skipped


def scrape():
    """Main entry point: download all court decision datasets."""
    print("Downloading Israeli court decisions from HuggingFace...")

    # Dataset 1: guychuk (smaller, various courts)
    saved1, total1 = scrape_guychuk()
    print(f"  guychuk: saved {saved1}/{total1}")

    # Dataset 2: LevMuchnik Supreme Court (large, 751K docs)
    saved2, total2 = scrape_supreme()
    print(f"  Supreme Court: saved {saved2}/{total2}")

    print(f"\nTotal court documents saved: {saved1 + saved2}")
