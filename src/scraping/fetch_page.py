"""
fetch_page.py
-------------
Utility to fetch a single HTML page from ikman.lk.
Handles headers, retries, and saves a debug snapshot.
"""

import requests
import time
import random
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs", "fetch_page.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

BASE_URL = "https://ikman.lk/en/ads/sri-lanka/mobiles"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_2) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
]


def build_session() -> requests.Session:
    """Creates a configured requests session with browser-like headers."""
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Referer": "https://ikman.lk/",
        }
    )
    return session


def fetch_page(
    url: str,
    page_num: int = 1,
    session: requests.Session = None,
    retries: int = 5,
    save_debug: bool = False,
) -> str | None:
    """
    Fetch HTML content of a URL with retry logic.

    Parameters
    ----------
    url       : Full URL to fetch (page query param appended here)
    page_num  : Page number to append as query parameter
    session   : Requests session to reuse; creates one if None
    retries   : Number of retry attempts on failure
    save_debug: If True, saves HTML snapshot to page_source.html

    Returns
    -------
    HTML string on success, None on failure.
    """
    if session is None:
        session = build_session()

    paginated_url = f"{url}?page={page_num}" if page_num > 1 else url
    wait_times = [2, 4, 8, 16, 32]  # exponential back-off

    for attempt in range(retries):
        try:
            session.headers["User-Agent"] = random.choice(USER_AGENTS)
            logger.info(f"Fetching page {page_num}: {paginated_url} (attempt {attempt + 1})")
            response = session.get(paginated_url, timeout=30)

            if response.status_code == 200:
                html = response.text
                if save_debug:
                    with open("page_source.html", "w", encoding="utf-8") as f:
                        f.write(html)
                    logger.info("Debug snapshot saved to page_source.html")
                return html

            elif response.status_code == 403:
                wait = wait_times[min(attempt, len(wait_times) - 1)] + random.uniform(1, 3)
                logger.warning(f"403 Forbidden – Cloudflare block. Waiting {wait:.1f}s ...")
                time.sleep(wait)

            elif response.status_code == 429:
                wait = wait_times[min(attempt, len(wait_times) - 1)] * 2
                logger.warning(f"429 Rate-limited. Waiting {wait:.1f}s ...")
                time.sleep(wait)

            else:
                logger.warning(f"HTTP {response.status_code} on page {page_num}")
                time.sleep(wait_times[min(attempt, len(wait_times) - 1)])

        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error on page {page_num}: {e}")
            time.sleep(wait_times[min(attempt, len(wait_times) - 1)])
        except requests.exceptions.Timeout:
            logger.error(f"Timeout on page {page_num}")
            time.sleep(wait_times[min(attempt, len(wait_times) - 1)])
        except Exception as e:
            logger.error(f"Unexpected error on page {page_num}: {e}")
            time.sleep(2)

    logger.error(f"All {retries} attempts failed for page {page_num}")
    return None


if __name__ == "__main__":
    # Quick test: fetch page 1 and save a debug snapshot
    session = build_session()
    html = fetch_page(BASE_URL, page_num=1, session=session, save_debug=True)
    if html:
        print(f"SUCCESS – fetched {len(html):,} characters from page 1")
    else:
        print("FAILED – could not fetch page 1")
