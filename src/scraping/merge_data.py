"""
merge_data.py
-------------
Combines all chunk CSV files from data/raw/chunks/ into a single
mobile_data_raw.csv file, deduplicates records, and reports stats.

Usage:
    python src/scraping/merge_data.py
"""

import csv
import glob
import logging
import os
import sys

import pandas as pd

# ─── Configuration ────────────────────────────────────────────────────────────
CHUNK_DIR   = os.path.join("data", "raw", "chunks")
OUTPUT_FILE = os.path.join("data", "raw", "mobile_data_raw.csv")
LOG_DIR     = "logs"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "merge_data.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def merge_chunks(chunk_dir: str = CHUNK_DIR, output_file: str = OUTPUT_FILE) -> int:
    """
    Merge all chunk_*.csv files in chunk_dir into a single CSV.

    Returns
    -------
    Number of records written to output_file.
    """
    pattern = os.path.join(chunk_dir, "chunk_*.csv")
    chunk_files = sorted(glob.glob(pattern))

    if not chunk_files:
        logger.error(f"No chunk files found in '{chunk_dir}'. Run mobile_scraper.py first.")
        sys.exit(1)

    logger.info(f"Found {len(chunk_files)} chunk files in '{chunk_dir}'")

    dfs = []
    total_raw = 0
    for fp in chunk_files:
        try:
            df = pd.read_csv(fp, dtype=str)
            total_raw += len(df)
            dfs.append(df)
            logger.info(f"  Loaded {len(df):>5} rows from {os.path.basename(fp)}")
        except Exception as e:
            logger.warning(f"  Could not read {fp}: {e}")

    if not dfs:
        logger.error("All chunk files failed to load. Aborting.")
        sys.exit(1)

    merged = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total rows before deduplication: {total_raw:,}")

    # Deduplicate by URL (same listing scraped twice)
    before = len(merged)
    merged = merged.drop_duplicates(subset=["url"], keep="first")
    duplicates_removed = before - len(merged)

    # Re-assign sequential IDs
    merged["id"] = range(1, len(merged) + 1)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    merged.to_csv(output_file, index=False, encoding="utf-8")

    logger.info(f"Duplicates removed : {duplicates_removed:,}")
    logger.info(f"Final record count : {len(merged):,}")
    logger.info(f"Output saved to    : {output_file}")

    return len(merged)


if __name__ == "__main__":
    count = merge_chunks()
    print(f"\nMerged dataset saved -> '{OUTPUT_FILE}'  ({count:,} records)")
    print("  Run 'python src/preprocessing/preprocess_data.py' to clean the data.")
