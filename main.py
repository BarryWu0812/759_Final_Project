# stock_predict.py

import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from svr_model import StockSVR, prepare_data
from sklearn.preprocessing import StandardScaler

# Fetch stock data
def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Close'].values

# Main function for stock prediction
def main():
    # Parameters
    ticker = "AAPL"   # Stock ticker symbol
    start_date = "2020-01-01"
    end_date = "2024-11-08"
    n_days = 5        # Use last 5 days' prices as features

    # Load and prepare stock data
    data = get_stock_data(ticker, start_date, end_date)
    X, y = prepare_data(data, n_days)

    # Split data into training and testing sets
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    # Train the SVR model
    svr = StockSVR(learning_rate=0.001, C=1.0, n_iters=1000, epsilon=0.1)
    svr.fit(X_train, y_train)

    # Make predictions
    predictions = svr.predict(X_test)
    predictions = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()

    y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Evaluate model performance
    mse = np.mean((predictions - y_test) ** 2)
    print(f"Mean Squared Error (MSE): {mse:.4f}")

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(y_test)), y_test, label="True Prices", color="blue")
    plt.plot(range(len(predictions)), predictions, label="Predicted Prices", color="red")
    plt.xlabel("Days")
    plt.ylabel("Stock Price")
    plt.title(f"{ticker} Stock Price Prediction using Custom SVR")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
