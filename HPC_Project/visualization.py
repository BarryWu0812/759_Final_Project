# visualization.py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def plot_actual_vs_predicted(y_test, y_pred, title="Actual vs Predicted Prices"):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual Prices', color='blue')
    plt.plot(y_pred, label='Predicted Prices', color='orange', linestyle='--')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

def plot_error(y_test, y_pred, title="Prediction Error Over Time"):
    errors = np.abs(y_test - y_pred)
    plt.figure(figsize=(12, 6))
    plt.plot(errors, label='Absolute Error', color='red')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Absolute Error")
    plt.legend()
    plt.show()

def plot_error_distribution(y_test, y_pred, title="Prediction Error Distribution"):
    errors = np.abs(y_test - y_pred)
    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=20, color='purple', alpha=0.7)
    plt.title(title)
    plt.xlabel("Absolute Error")
    plt.ylabel("Frequency")
    plt.show()

def print_error_metrics(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

