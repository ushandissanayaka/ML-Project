# Mobile Phone Price Scraper & ML Pipeline

Scrapes **5,500+ mobile phone listings** from [ikman.lk](https://ikman.lk/en/ads/sri-lanka/mobiles) and builds a price prediction model.

## Project Structure

```
ML assignment/
├── data/
│   ├── raw/
│   │   ├── chunks/                  ← incremental chunk CSVs (auto-created)
│   │   └── mobile_data_raw.csv      ← merged raw data
│   └── processed/
│       └── mobile_data_processed.csv
├── outputs/
│   ├── metrics/model_results.txt
│   └── plots/
│       ├── actual_vs_predicted.png
│       ├── feature_importance.png
│       └── residuals.png
├── src/
│   ├── scraping/
│   │   ├── fetch_page.py            ← single-page fetcher utility
│   │   ├── mobile_scraper.py        ← main scraper (5,500 records)
│   │   └── merge_data.py            ← chunk merger
│   ├── preprocessing/
│   │   └── preprocess_data.py
│   └── modeling/
│       └── train_model.py
├── logs/                            ← runtime log files
└── README.md
```

## Quick Start

### 1 — Install dependencies
```bash
pip install requests beautifulsoup4 pandas scikit-learn matplotlib
```

### 2 — Scrape mobile phone listings (5,500 records)
```bash
python src/scraping/mobile_scraper.py
```
> Chunks are saved every 100 records to `data/raw/chunks/` so no data is lost on interruption.

### 3 — Merge chunks into one CSV
```bash
python src/scraping/merge_data.py
```

### 4 — Preprocess the raw data
```bash
python src/preprocessing/preprocess_data.py
```

### 5 — Train price prediction model
```bash
python src/modeling/train_model.py
```

### Quick test (50 records only)
```bash
python src/scraping/mobile_scraper.py --max-records 50
python src/scraping/merge_data.py
python src/preprocessing/preprocess_data.py
```

## Scraped Fields

| Column       | Description                           |
|--------------|---------------------------------------|
| `id`         | Sequential record ID                  |
| `title`      | Listing title                         |
| `price`      | Price (numeric, LKR)                  |
| `currency`   | Currency (LKR)                        |
| `location`   | City / district                       |
| `condition`  | Brand New / Used / Unknown            |
| `brand`      | Detected brand (Samsung, Apple, etc.) |
| `date_posted`| Date the ad was posted                |
| `url`        | Direct link to the listing            |
| `scraped_at` | Timestamp of scraping                 |

## Notes

- The scraper uses **polite delays** (1.8–3.5 s per page) and rotating User-Agent headers.
- Full 5,500-record scrape takes approximately **15–30 minutes**.
- Data is saved in chunks — safe to interrupt and resume.
