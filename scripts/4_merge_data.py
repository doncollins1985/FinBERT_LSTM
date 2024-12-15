import pandas as pd

# Load stock price data
stock_prices = pd.read_csv("data/stock_prices.csv")  # Columns: Date, Close

# Load sentiment data
# Columns: Date, Positive, Negative, Neutral
news_sentiment = pd.read_csv("data/news_sentiment.csv")

# Merge data on 'Date'
# Use 'inner' join to keep matching dates
merged_data = pd.merge(stock_prices, news_sentiment, on="Date", how="inner")

# Save merged data
merged_data.to_csv("data/merged_data.csv", index=False)

print("Merged data saved to data/merged_data.csv")
