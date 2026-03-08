"""CLI entry point for the data cleaning pipeline."""

import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
CLEANED_DIR = BASE_DIR / "data" / "cleaned"
FILTERED_DIR = BASE_DIR / "data" / "filtered"
TRAINING_DIR = BASE_DIR / "data" / "training"


def print_stats():
    """Print corpus statistics at each stage."""
    print("\n=== Raw Data ===")
    if RAW_DIR.exists():
        for d in sorted(RAW_DIR.iterdir()):
            if d.is_dir():
                count = sum(1 for _ in d.glob("*.json"))
                print(f"  {d.name}: {count:,} files")
            elif d.is_file():
                size_mb = d.stat().st_size / (1024 * 1024)
                print(f"  {d.name}: {size_mb:.1f} MB")

    print("\n=== Cleaned Data ===")
    if CLEANED_DIR.exists():
        for d in sorted(CLEANED_DIR.iterdir()):
            if d.is_dir():
                count = sum(1 for _ in d.glob("*.json"))
                print(f"  {d.name}: {count:,} files")
    else:
        print("  (not yet generated)")

    print("\n=== Filtered Data ===")
    if FILTERED_DIR.exists():
        for d in sorted(FILTERED_DIR.iterdir()):
            if d.is_dir():
                count = sum(1 for _ in d.glob("*.json"))
                print(f"  {d.name}: {count:,} files")
    else:
        print("  (not yet generated)")

    print("\n=== Training Data ===")
    for name in ("pretrain.jsonl", "eval.jsonl"):
        p = TRAINING_DIR / name
        if p.exists():
            with open(p, encoding="utf-8") as f:
                lines = sum(1 for _ in f)
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  {name}: {lines:,} rows, {size_mb:.1f} MB")
    if not (TRAINING_DIR / "pretrain.jsonl").exists():
        print("  (not yet generated)")


def main():
    parser = argparse.ArgumentParser(description="Israeli Law LLM Data Pipeline")
    parser.add_argument(
        "--step",
        choices=["clean", "pii", "filter", "dedup", "format", "analyze", "stats", "all"],
        default="all",
        help="Pipeline step to run (default: all)",
    )
    parser.add_argument(
        "--no-ner",
        action="store_true",
        help="Skip NER-based name detection (regex-only PII scrubbing)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device index for NER model (default: 0, auto-detects RTX)",
    )
    args = parser.parse_args()

    if args.step == "stats":
        print_stats()
        return

    if args.step == "analyze":
        from pipeline.analyze import run_analysis
        run_analysis()
        return

    steps = ["clean", "pii", "filter", "dedup", "format"] if args.step == "all" else [args.step]

    for step in steps:
        print(f"\n{'='*50}")
        print(f"Running: {step}")
        print(f"{'='*50}")

        if step == "clean":
            from pipeline.clean import run_cleaning
            run_cleaning()

        elif step == "pii":
            from pipeline.pii import run_pii_scrub
            run_pii_scrub(use_ner=not args.no_ner, gpu_device=args.gpu)

        elif step == "filter":
            from pipeline.filter import run_filtering
            run_filtering()

        elif step == "dedup":
            from pipeline.dedup import run_dedup
            run_dedup()

        elif step == "format":
            from pipeline.format import run_format
            run_format()

    print("\nDone!")


if __name__ == "__main__":
    main()
