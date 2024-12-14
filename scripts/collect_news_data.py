import requests
import pandas as pd

API_KEY = "your-nyt-api-key"  # Replace with your NY Times API Key
url = "https://api.nytimes.com/svc/search/v2/articlesearch.json"

def fetch_news_data(dates, output_file):
    news_data = []
    for date in dates:
        params = {
            'q': 'finance business technology',
            'begin_date': date.replace("-", ""),
            'end_date': date.replace("-", ""),
            'api-key': API_KEY
        }
        response = requests.get(url, params=params).json()
        articles = [doc['headline']['main'] for doc in response['response']['docs']]
        news_data.append({'Date': date, 'Articles': articles})
    
    pd.DataFrame(news_data).to_csv(output_file, index=False)
    print(f"News data saved to {output_file}")

# Example usage
if __name__ == "__main__":
    # Example: Read dates from stock data
    stock_data = pd.read_csv("data/stock_prices.csv")
    dates = stock_data['Date'].tolist()
    fetch_news_data(dates, "data/news_data.csv")

