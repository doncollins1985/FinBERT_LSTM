import os
import time
import requests
import pandas as pd
import logging
import json
import csv  # Import csv module
from tqdm import tqdm
from datetime import datetime
from typing import List
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ==============================
# Configuration and Setup
# ==============================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("collect_news_data.log"),
        logging.StreamHandler()
    ]
)

# Load API Key from environment variable
API_KEY = os.getenv("NYT_API_KEY")
if not API_KEY:
    logging.error("NYT_API_KEY environment variable not set.")
    raise EnvironmentError("Please set the NYT_API_KEY environment variable.")

# Constants
BASE_URL = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
QUERY = "finance business technology"
MAX_RETRIES = 2
BACKOFF_FACTOR = 5  # For exponential backoff
MAX_PAGES = 10        # Maximum pages per date to fetch

# ==============================
# Helper Functions
# ==============================


def create_session() -> requests.Session:
    """
    Create a requests Session with a retry strategy.
    """
    session = requests.Session()
    retries = Retry(
        total=MAX_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch_articles_for_date(session: requests.Session, date: str, max_pages: int = MAX_PAGES) -> List[str]:
    """
    Fetch articles for a specific date, handling pagination.

    Args:
        session: requests.Session object.
        date: Date string in 'YYYY-MM-DD' format.
        max_pages: Maximum number of pages to fetch.

    Returns:
        List of article headlines.
    """
    articles = []
    base_date = date.replace("-", "")
    for page in range(max_pages):
        params = {
            'q': QUERY,
            'begin_date': base_date,
            'end_date': base_date,
            'api-key': API_KEY,
            'page': page
        }
        try:
            response = session.get(BASE_URL, params=params, timeout=10)
            if response.status_code == 429:
                # Rate limit exceeded, wait and retry
                retry_after = int(response.headers.get("Retry-After", 60))
                logging.warning(f"Rate limit exceeded. Retrying after {
                                retry_after} seconds.")
                time.sleep(retry_after)
                continue
            response.raise_for_status()
            data = response.json()
            docs = data.get('response', {}).get('docs', [])
            if not docs:
                break  # No more articles
            for doc in docs:
                headline = doc.get('headline', {}).get('main')
                if headline:
                    articles.append(headline)
            # If less than 10 articles are returned, it's likely the last page
            if len(docs) < 10:
                break
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed for date {
                          date} on page {page}: {e}")
            break
    return articles


def load_dates_from_stock_data(stock_file: str) -> List[str]:
    """
    Load dates from a stock prices CSV file.

    Args:
        stock_file: Path to the stock prices CSV file.

    Returns:
        List of date strings in 'YYYY-MM-DD' format.
    """
    try:
        stock_data = pd.read_csv(stock_file)
        if 'Date' not in stock_data.columns:
            logging.error("Stock data CSV does not contain 'Date' column.")
            raise KeyError("Missing 'Date' column.")
        # Ensure dates are in string format and sorted
        dates = stock_data['Date'].astype(str).dropna().unique().tolist()
        dates = [datetime.strptime(date, "%Y-%m-%d").strftime("%Y-%m-%d")
                 for date in dates if is_valid_date(date)]
        dates.sort()
        return dates
    except Exception as e:
        logging.error(f"Failed to load dates from {stock_file}: {e}")
        raise


def is_valid_date(date_str: str) -> bool:
    """
    Validate if the provided string is a valid date in 'YYYY-MM-DD' format.

    Args:
        date_str: Date string.

    Returns:
        True if valid, False otherwise.
    """
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def initialize_output_file(output_file: str) -> None:
    """
    Initialize the output CSV file by creating it with headers if it doesn't exist.

    Args:
        output_file: Path to the output CSV file.
    """
    if not os.path.exists(output_file):
        # Create the file with headers using csv module
        try:
            with open(output_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Date', 'Articles'])
            logging.info(
                f"Created new output file with headers: {output_file}")
        except Exception as e:
            logging.error(f"Failed to initialize output file {
                          output_file}: {e}")
            raise


def append_to_csv(output_file: str, date: str, articles: List[str]) -> None:
    """
    Append a single row of data to the CSV file.

    Args:
        output_file: Path to the output CSV file.
        date: Date string in 'YYYY-MM-DD' format.
        articles: List of article headlines.
    """
    # Serialize the articles list as a JSON string
    articles_str = json.dumps(articles, ensure_ascii=False)
    # Append to the CSV file using csv.writer to handle quoting
    try:
        with open(output_file, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([date, articles_str])
        logging.info(f"Appended {len(articles)} articles for date {
                     date} to {output_file}.")
    except Exception as e:
        logging.error(f"Failed to append data for date {date}: {e}")


def get_existing_dates(output_file: str) -> set:
    """
    Retrieve a set of dates already present in the output CSV file.

    Args:
        output_file: Path to the output CSV file.

    Returns:
        Set of date strings in 'YYYY-MM-DD' format.
    """
    existing_dates = set()
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    date = row.get('Date', '').strip()
                    if is_valid_date(date):
                        existing_dates.add(date)
            logging.info(f"Loaded {len(existing_dates)
                                   } existing dates from {output_file}.")
        except Exception as e:
            logging.error(f"Failed to read existing output file {
                          output_file}: {e}")
            # Decide whether to proceed or abort. Here, we'll proceed with all dates.
    return existing_dates


def fetch_news_data(dates: List[str], output_file: str, max_pages_per_date: int = MAX_PAGES) -> None:
    """
    Fetch news articles for a list of dates and save to a CSV file incrementally.

    Args:
        dates: List of date strings in 'YYYY-MM-DD' format.
        output_file: Path to the output CSV file.
        max_pages_per_date: Maximum number of pages to fetch per date.
    """
    session = create_session()

    # Initialize the output file if it doesn't exist
    initialize_output_file(output_file)

    # Retrieve existing dates to avoid re-downloading
    existing_dates = get_existing_dates(output_file)

    # Filter dates to only include those not already processed
    filtered_dates = [date for date in dates if date not in existing_dates]
    logging.info(f"Total dates to process: {len(filtered_dates)}")

    if not filtered_dates:
        logging.info("No new dates to process. Exiting.")
        return

    for date in tqdm(filtered_dates, desc="Fetching news data"):
        # Date format has been validated during loading
        articles = fetch_articles_for_date(
            session, date, max_pages=max_pages_per_date)
        if articles:
            append_to_csv(output_file, date, articles)
        else:
            logging.warning(f"No articles found for date {date}.")

        # Short sleep to avoid hitting rate limits too quickly
        time.sleep(7)  # Adjust based on NYT API rate limits


# ==============================
# Main Execution
# ==============================


def main():
    """
    Main function to execute the news data fetching process.
    """
    STOCK_DATA_FILE = "data/stock_prices.csv"
    OUTPUT_FILE = "data/news_data.csv"

    try:
        dates = load_dates_from_stock_data(STOCK_DATA_FILE)
        fetch_news_data(dates, OUTPUT_FILE, max_pages_per_date=MAX_PAGES)
    except Exception as e:
        logging.critical(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
