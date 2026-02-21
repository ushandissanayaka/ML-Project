"""
mobile_scraper.py
-----------------
Main scraper for ikman.lk mobile phone listings.
Scrapes up to MAX_RECORDS listings and saves them as chunked CSVs.

CSS selectors verified against live page (2026-02-21):
  - Cards      : a[data-testid="ad-card-link"]
  - Title      : h2[class*="title"]
  - Price      : div[class*="price"]
  - Description: div[class*="description"]  -> "Location, Category"

Usage:
    python src/scraping/mobile_scraper.py
    python src/scraping/mobile_scraper.py --max-records 100
"""

import argparse
import csv
import os
import re
import sys
import time
import random
import logging
from datetime import datetime
from typing import Optional

from bs4 import BeautifulSoup

# Ensure project root is on path so sibling modules can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from fetch_page import BASE_URL, build_session, fetch_page

# ── Configuration ─────────────────────────────────────────────────────────────
MAX_RECORDS = 10000
CHUNK_SIZE  = 100
DELAY_MIN   = 1.8
DELAY_MAX   = 3.5
CHUNK_DIR   = os.path.join("data", "raw", "chunks")
LOG_DIR     = "logs"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(CHUNK_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(LOG_DIR, "mobile_scraper.log"), encoding="utf-8"
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

CSV_FIELDS = [
    "id", "title", "price", "currency", "location",
    "condition", "brand", "date_posted", "url", "scraped_at",
]

KNOWN_BRANDS = [
    "Samsung", "Apple", "iPhone", "Huawei", "Xiaomi", "Oppo", "Vivo",
    "Realme", "Nokia", "Sony", "Motorola", "LG", "OnePlus", "Honor",
    "Tecno", "Infinix", "Itel", "ZTE", "Lenovo", "Asus", "Google",
    "Poco", "iQOO", "Nothing", "Redmi",
]

# ── Parsing helpers ───────────────────────────────────────────────────────────

def parse_listings(html: str, base_domain: str = "https://ikman.lk") -> list[dict]:
    """
    Parse the HTML of a mobile listings page and return a list of record dicts.
    ikman.lk cards are <a data-testid="ad-card-link"> elements.
    """
    soup = BeautifulSoup(html, "html.parser")
    cards = soup.find_all("a", attrs={"data-testid": "ad-card-link"})

    if not cards:
        logger.debug("No ad-card-link elements found – trying fallback selectors")
        cards = soup.select("li.normal--2QYVk a[href*='/en/ad/']")

    records = []
    for card in cards:
        try:
            rec = _extract_card(card, base_domain)
            if rec:
                records.append(rec)
        except Exception as e:
            logger.debug(f"Skipping card: {e}")
    return records


def _extract_card(card, base_domain: str) -> Optional[dict]:
    """Extract all fields from a single <a data-testid='ad-card-link'> element."""

    # ── URL ──────────────────────────────────────────────────────────────────
    href = card.get("href", "")
    if not href:
        return None
    ad_url = href if href.startswith("http") else base_domain + href

    # ── Title ─────────────────────────────────────────────────────────────────
    title_tag = (
        card.find("h2")
        or card.find(class_=re.compile(r"title", re.I))
    )
    # Fallback: use the <a title="..."> attribute
    title = (
        title_tag.get_text(strip=True)
        if title_tag
        else card.get("title", "N/A").replace(" for sale", "").strip()
    )

    # ── Price ──────────────────────────────────────────────────────────────────
    price_tag = card.find(class_=re.compile(r"price", re.I))
    price_raw = price_tag.get_text(strip=True) if price_tag else "N/A"
    price, currency = _parse_price(price_raw)

    # ── Location ──────────────────────────────────────────────────────────────
    # ikman.lk puts "Colombo, Mobile Phones" in div[class*="description"]
    desc_tag = card.find(class_=re.compile(r"description", re.I))
    if desc_tag:
        desc_text = desc_tag.get_text(strip=True)
        # "Colombo, Mobile Phones" → split on comma, take first part
        location = desc_text.split(",")[0].strip()
    else:
        location = "N/A"

    # ── Date posted ───────────────────────────────────────────────────────────
    date_tag = card.find("time") or card.find(class_=re.compile(r"date|time", re.I))
    if date_tag:
        date_posted = date_tag.get("datetime") or date_tag.get_text(strip=True)
    else:
        date_posted = "N/A"

    # ── Condition ─────────────────────────────────────────────────────────────
    condition = _infer_condition(title)

    # ── Brand ─────────────────────────────────────────────────────────────────
    brand = _infer_brand(title)

    return {
        "id": None,
        "title": title,
        "price": price,
        "currency": currency,
        "location": location,
        "condition": condition,
        "brand": brand,
        "date_posted": date_posted,
        "url": ad_url,
        "scraped_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _parse_price(raw: str) -> tuple[str, str]:
    """Return (numeric_price_str, currency)."""
    if not raw or raw == "N/A":
        return ("N/A", "LKR")
    cleaned = raw.replace(",", "").replace(" ", "")
    match = re.search(r"\d+(?:\.\d+)?", cleaned)
    price = match.group() if match else "N/A"
    currency = "LKR" if ("Rs" in raw or "\u20a8" in raw or "\u0dbb\u0dd4" in raw) else "LKR"
    return (price, currency)


def _infer_brand(title: str) -> str:
    t_lower = title.lower()
    for brand in KNOWN_BRANDS:
        if brand.lower() in t_lower:
            return "Apple" if brand == "iPhone" else brand
    return "Other"


def _infer_condition(title: str) -> str:
    t_lower = title.lower()
    if "brand new" in t_lower:
        return "Brand New"
    if "used" in t_lower or "second hand" in t_lower or "recondition" in t_lower:
        return "Used"
    return "Unknown"


# ── Chunk saving ──────────────────────────────────────────────────────────────

def save_chunk(records: list[dict], chunk_index: int) -> str:
    filename = os.path.join(CHUNK_DIR, f"chunk_{chunk_index:04d}.csv")
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    logger.info(f"Saved chunk {chunk_index} -> {filename} ({len(records)} records)")
    return filename


# ── Main scraping loop ────────────────────────────────────────────────────────

def scrape(max_records: int = MAX_RECORDS):
    session = build_session()
    all_records: list[dict] = []
    chunk_index = 66
    page_num = 235
    global_id = 6501
    consecutive_empty = 0

    logger.info(f"Starting scrape - target: {max_records} records")

    while len(all_records) < max_records:
        html = fetch_page(BASE_URL, page_num=page_num, session=session)

        if html is None:
            consecutive_empty += 1
            logger.warning(f"No HTML from page {page_num} (empty streak: {consecutive_empty})")
            if consecutive_empty >= 5:
                logger.error("5 consecutive empty pages - stopping scrape")
                break
            time.sleep(random.uniform(5, 10))
            page_num += 1
            continue

        listings = parse_listings(html)

        if not listings:
            consecutive_empty += 1
            logger.warning(f"No listings from page {page_num} (empty streak: {consecutive_empty})")
            if consecutive_empty >= 5:
                logger.error("5 consecutive pages with no listings - stopping")
                break
            page_num += 1
            time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
            continue

        consecutive_empty = 0

        for rec in listings:
            if len(all_records) >= max_records:
                break
            rec["id"] = global_id
            all_records.append(rec)
            global_id += 1

        logger.info(
            f"Page {page_num}: +{len(listings)} listings | "
            f"Total: {len(all_records)}/{max_records}"
        )

        # Save a chunk whenever we cross a CHUNK_SIZE boundary
        while chunk_index * CHUNK_SIZE <= len(all_records):
            chunk_records = all_records[(chunk_index - 1) * CHUNK_SIZE: chunk_index * CHUNK_SIZE]
            save_chunk(chunk_records, chunk_index)
            chunk_index += 1

        page_num += 1
        time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

    # Save any remaining records
    remaining_start = (chunk_index - 1) * CHUNK_SIZE
    if remaining_start < len(all_records):
        save_chunk(all_records[remaining_start:], chunk_index)

    logger.info(
        f"Scraping complete. {len(all_records)} records across {chunk_index} chunks in '{CHUNK_DIR}'"
    )
    return all_records


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape mobile phone data from ikman.lk")
    parser.add_argument(
        "--max-records",
        type=int,
        default=MAX_RECORDS,
        help=f"Maximum number of records to scrape (default: {MAX_RECORDS})",
    )
    args = parser.parse_args()

    records = scrape(max_records=args.max_records)
    print(f"\nDone. {len(records)} records saved to '{CHUNK_DIR}'")
    print("  Run: python src/scraping/merge_data.py")
