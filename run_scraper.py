"""CLI entry point for Israeli law scrapers."""

import argparse
import sys

from scraper import kolzchut, court, knesset, wikisource


SCRAPERS = {
    "kolzchut": kolzchut.scrape,
    "court": court.scrape,
    "knesset": knesset.scrape,
    "wikisource": wikisource.scrape,
}


def main():
    parser = argparse.ArgumentParser(description="Israeli Law Data Scraper")
    parser.add_argument(
        "--source",
        choices=["kolzchut", "court", "knesset", "wikisource", "all"],
        required=True,
        help="Which source to scrape",
    )
    args = parser.parse_args()

    sources = list(SCRAPERS.keys()) if args.source == "all" else [args.source]

    for source in sources:
        print(f"\n{'='*50}")
        print(f"Scraping: {source}")
        print(f"{'='*50}")
        try:
            SCRAPERS[source]()
        except KeyboardInterrupt:
            print(f"\nInterrupted during {source}. Exiting.")
            sys.exit(1)
        except Exception as e:
            print(f"Error in {source} scraper: {e}")
            continue

    print("\nDone!")


if __name__ == "__main__":
    main()
