import matplotlib.pyplot as plt
import pandas as pd

def plot_stock_prediction(y_test, predictions):
    """
    Plots true stock prices and predicted prices.

    Parameters:
    - y_test: List or array of true stock prices.
    - predictions: List or array of predicted stock prices.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label="True Prices", color="blue")
    plt.plot(predictions, label="Predicted Prices", color="red")
    plt.xlabel("Days")
    plt.ylabel("Stock Price")
    plt.title("Stock Price Prediction")
    plt.legend()
    plt.grid(True)
    plt.show()

def load_csv(filename):
    """
    Loads data from a CSV file.

    Parameters:
    - filename: Path to the CSV file.

    Returns:
    - List of values from the CSV.
    """
    return pd.read_csv(filename, header=None).values.flatten()

# Main function to test plotting
def main():
    # Load data from CSV files (replace with your filenames)
    y_test = load_csv("y_test.csv")         # True prices
    predictions = load_csv("predictions.csv")  # Predicted prices

    # Plot the stock prediction results
    plot_stock_prediction(y_test, predictions)

if __name__ == "__main__":
    main()
