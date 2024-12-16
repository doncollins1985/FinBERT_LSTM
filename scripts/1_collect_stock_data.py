import yfinance as yf


def fetch_stock_data(ticker, start_date, end_date, output_file):
    """
    Fetch stock data for a given ticker and date range, then save the closing prices to a CSV file.

    Parameters:
        ticker (str): Stock ticker symbol.
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        output_file (str): Path to the output CSV file.
    """
    try:
        # Fetch stock data using yfinance
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            print(f"No data found for {ticker} between {
                  start_date} and {end_date}.")
            return

        # Keep only closing prices and reset index
        data = data[['Close']]
        data.reset_index(inplace=True)
        data.columns = ['Date', 'Close']

        # Convert dates to strings for compatibility
        data['Date'] = data['Date'].astype(str)

        # Save to CSV file
        data.to_csv(output_file, index=False)
        print(f"Stock data saved to {output_file}")
    except Exception as e:
        print(f"An error occurred while fetching stock data: {e}")


# Example usage
if __name__ == "__main__":
    output_path = "data/stock_prices.csv"
    fetch_stock_data("^GSPC", "2010-01-01", "2024-12-01", output_path)
