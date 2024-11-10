import yfinance as yf
import numpy as np
import pandas as pd
from math import floor

def load_stock_data(stock_symbol, period, interval):
    data = yf.Ticker(stock_symbol).history(period=period, interval=interval)
    print(data);
    return data

def preprocess_data(data):

    # Feature engineering
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['Volatility'] = data['Close'].rolling(window=20).std()
    data['Daily Return'] = data['Close'].pct_change()
    data.dropna(inplace=True)  # Drop rows with NaN values
    
    # Define features and target
    X = data[['Close', 'MA5', 'MA20', 'Volatility', 'Daily Return']].values
    y = data['Close'].shift(-1).dropna().values  # Predicting the next day's close
    X = X[:-1]  # Align X with y by dropping the last row
    
    return X, y

def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    indices = np.random.choice(n_samples, size=floor(n_samples*0.7), replace=True)
    return X[indices], y[indices]

