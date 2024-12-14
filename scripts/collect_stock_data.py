import yfinance as yf

def fetch_stock_data(ticker, start_date, end_date, output_file):
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data[['Close']]  # Keep only closing prices
    data.reset_index(inplace=True)
    data.columns = ['Date', 'Close']
    data['Date'] = data['Date'].astype(str)  # Convert dates to strings for merging
    data.to_csv(output_file, index=False)
    print(f"Stock data saved to {output_file}")

# Example usage
if __name__ == "__main__":
    fetch_stock_data("NDX", "2020-10-01", "2022-09-30", "data/stock_prices.csv")

