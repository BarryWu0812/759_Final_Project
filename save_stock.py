import numpy as np
import yfinance as yf
import csv

# Function to get stock data
def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Close'].values

# Save data to CSV file
def save_to_csv(X, X_filename):
    # Save features (X)
    with open(X_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in X:
            writer.writerow(row)


def main():
    # Parameters
    ticker = "AAPL"   # Stock ticker symbol
    start_date = "2020-01-01"
    end_date = "2024-11-28"
    n_days = 5        # Use last 5 days' prices as features

    # Load and prepare stock data
    data = get_stock_data(ticker, start_date, end_date)

    # Save the data to CSV files
    save_to_csv(data, ticker + "_" + end_date + ".csv")

    print(f"Features saved to X_data.csv")

# Prepare data for training

if __name__ == "__main__":
    main()
