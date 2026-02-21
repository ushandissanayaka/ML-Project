"""
preprocess_data.py
------------------
Cleans and preprocesses the raw scraped mobile phone data.
Reads  : data/raw/mobile_data_raw.csv
Writes : data/processed/mobile_data_processed.csv

Usage:
    python src/preprocessing/preprocess_data.py
"""

import logging
import os
import re
import sys
from typing import Optional

import pandas as pd

# ─── Configuration ────────────────────────────────────────────────────────────
RAW_FILE       = os.path.join("data", "raw", "mobile_data_raw.csv")
PROCESSED_FILE = os.path.join("data", "processed", "mobile_data_processed.csv")
LOG_DIR        = "logs"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(PROCESSED_FILE), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(LOG_DIR, "preprocess_data.log"), encoding="utf-8"
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

KNOWN_BRANDS = [
    "Samsung", "Apple", "Huawei", "Xiaomi", "Oppo", "Vivo", "Realme",
    "Nokia", "Sony", "Motorola", "LG", "OnePlus", "Honor", "Tecno",
    "Infinix", "Itel", "ZTE", "Lenovo", "Asus", "Google", "Other",
]


# ─── Cleaning functions ───────────────────────────────────────────────────────

def clean_price(df: pd.DataFrame) -> pd.DataFrame:
    """Convert 'price' column from string to float. Nullify unparseable values."""
    def to_numeric(val):
        if pd.isna(val) or str(val).strip() in ("N/A", ""):
            return None
        cleaned = re.sub(r"[^\d.]", "", str(val))
        try:
            return float(cleaned) if cleaned else None
        except ValueError:
            return None

    df["price"] = df["price"].apply(to_numeric)
    return df


def clean_title(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and remove non-printable characters from title."""
    df["title"] = (
        df["title"]
        .fillna("Unknown")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    return df


def clean_location(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise location field."""
    df["location"] = (
        df["location"]
        .fillna("Unknown")
        .str.strip()
        .str.title()
        .replace("N/A", "Unknown")
    )
    return df


def clean_condition(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise condition values."""
    valid = {"Brand New", "Used", "Unknown"}
    df["condition"] = df["condition"].where(df["condition"].isin(valid), other="Unknown")
    return df


def clean_brand(df: pd.DataFrame) -> pd.DataFrame:
    """
    Re-infer brand from title if current value is 'Other'.
    Ensures brand is in the known-brands list.
    """
    def infer_from_title(row):
        if row["brand"] not in KNOWN_BRANDS:
            return "Other"
        return row["brand"]

    # Try to re-detect brand from title for rows labelled 'Other'
    def reclassify(row):
        if row["brand"] != "Other":
            return row["brand"]
        title_lower = str(row["title"]).lower()
        for b in KNOWN_BRANDS[:-1]:  # skip 'Other'
            if b.lower() in title_lower:
                return b
        return "Other"

    df["brand"] = df.apply(reclassify, axis=1)
    return df


def clean_brand_version(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract a simple model version from the title.
    e.g., 'iPhone 13' -> 'iPhone 13', 'Galaxy S21' -> 'S21'
    """
    def extract_version(row):
        title = str(row["title"])
        brand = str(row["brand"])
        
        # Remove brand from title to see what's left
        remaining = re.sub(brand, "", title, flags=re.IGNORECASE).strip()
        
        # Look for common model patterns (word + number or just numbers)
        # e.g., "13 Pro", "S22", "A52", "Pixel 6"
        match = re.search(r"\b([A-Za-z]?\d+[A-Za-z]*)\b", remaining)
        if match:
            return match.group(1).upper()
        
        # Fallback to the first word after brand
        words = remaining.split()
        if words:
            return words[0].upper()
            
        return "Generic"

    df["brand_version"] = df.apply(extract_version, axis=1)
    return df


def extract_ram(title: str) -> Optional[float]:
    """Extract RAM in GB from title (e.g., '8GB RAM', '12 GB RAM')."""
    match = re.search(r"\b(\d+)\s*GB\s*RAM\b", title, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def extract_storage(title: str) -> Optional[float]:
    """Extract storage in GB from title (e.g., '128GB', '256 GB', '1TB')."""
    # Look for TB first
    tb_match = re.search(r"\b(\d+)\s*TB\b", title, re.IGNORECASE)
    if tb_match:
        return float(tb_match.group(1)) * 1024
    
    # Look for GB not followed by RAM
    gb_match = re.search(r"\b(\d+)\s*GB\b(?!\s*RAM)", title, re.IGNORECASE)
    if gb_match:
        return float(gb_match.group(1))
    return None


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived feature columns useful for ML."""
    # Price bucket
    bins   = [0, 10_000, 30_000, 60_000, 100_000, 200_000, float("inf")]
    labels = ["<10k", "10k-30k", "30k-60k", "60k-100k", "100k-200k", ">200k"]
    df["price_range"] = pd.cut(
        df["price"], bins=bins, labels=labels, right=False
    )

    # Is brand new flag
    df["is_brand_new"] = (df["condition"] == "Brand New").astype(int)

    # Title length
    df["title_length"] = df["title"].str.len()

    # RAM and Storage extraction
    df["ram_gb"] = df["title"].apply(extract_ram)
    df["storage_gb"] = df["title"].apply(extract_storage)

    return df


# ─── Pipeline ─────────────────────────────────────────────────────────────────

def preprocess(raw_file: str = RAW_FILE, output_file: str = PROCESSED_FILE) -> pd.DataFrame:
    if not os.path.exists(raw_file):
        logger.error(f"Raw data file not found: '{raw_file}'")
        logger.error("Run merge_data.py first to generate the raw CSV.")
        sys.exit(1)

    logger.info(f"Loading raw data from '{raw_file}' ...")
    df = pd.read_csv(raw_file, dtype=str)
    logger.info(f"  Loaded {len(df):,} rows, {df.shape[1]} columns")

    # ── Step 1: Remove duplicate URLs ────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["url"], keep="first")
    logger.info(f"  Duplicates removed: {before - len(df):,}")

    # ── Step 2: Clean individual columns ─────────────────────────────────────
    df = clean_title(df)
    df = clean_price(df)
    df = clean_location(df)
    df = clean_condition(df)
    df = clean_brand(df)
    df = clean_brand_version(df)

    # ── Step 3: Drop rows with no price (critical feature) ────────────────────
    before = len(df)
    df = df.dropna(subset=["price"])
    logger.info(f"  Rows without price dropped: {before - len(df):,}")

    # ── Step 4: Enforce price sanity (> 0 and < 5,000,000 LKR) ───────────────
    before = len(df)
    df["price"] = df["price"].astype(float)
    df = df[(df["price"] > 0) & (df["price"] < 5_000_000)]
    logger.info(f"  Out-of-range price rows dropped: {before - len(df):,}")

    # ── Step 5: Feature engineering ───────────────────────────────────────────
    df = engineer_features(df)

    # ── Step 6: Reset index & re-assign IDs ───────────────────────────────────
    df = df.reset_index(drop=True)
    df["id"] = df.index + 1

    # ── Step 7: Save ──────────────────────────────────────────────────────────
    df.to_csv(output_file, index=False, encoding="utf-8")

    logger.info(f"Preprocessing complete.")
    logger.info(f"  Final record count : {len(df):,}")
    logger.info(f"  Columns            : {list(df.columns)}")
    logger.info(f"  Saved to           : '{output_file}'")

    # Print summary stats
    print("\n" + "=" * 55)
    print("  PREPROCESSING SUMMARY")
    print("=" * 55)
    print(f"  Records             : {len(df):,}")
    print(f"  Price range (LKR)   : {df['price'].min():,.0f} – {df['price'].max():,.0f}")
    print(f"  Median price (LKR)  : {df['price'].median():,.0f}")
    print(f"\n  Brand distribution:")
    for brand, cnt in df["brand"].value_counts().head(5).items():
        print(f"    {brand:<15} {cnt:>6,}")
    print(f"\n  Top Brand Versions:")
    for ver, cnt in df["brand_version"].value_counts().head(10).items():
        print(f"    {ver:<15} {cnt:>6,}")
    print(f"\n  Condition split:")
    for cond, cnt in df["condition"].value_counts().items():
        print(f"    {cond:<15} {cnt:>6,}")
    print("=" * 55)

    return df


if __name__ == "__main__":
    preprocess()
    print(f"\nProcessed data saved -> '{PROCESSED_FILE}'")
    print("  Run 'python src/modeling/train_model.py' to train the price model.")
