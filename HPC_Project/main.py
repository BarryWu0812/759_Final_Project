# main.py
from data_processing import load_stock_data, preprocess_data
from model import RandomForest
from visualization import plot_actual_vs_predicted, plot_error, plot_error_distribution, print_error_metrics
from sklearn.model_selection import train_test_split

# Load and preprocess data

stock_symbol="ZETA"
period="6mo"
interval = "1d"
data = load_stock_data(stock_symbol,period, interval)
X, y = preprocess_data(data)
print(X)



# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train the Random Forest model
rf = RandomForest(n_trees=100, max_depth=20, min_samples_split=2)
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Visualize results
plot_actual_vs_predicted(y_test, y_pred, title=f"{stock_symbol} Actual vs Predicted Prices")
plot_error(y_test, y_pred, title="Prediction Error Over Time")
plot_error_distribution(y_test, y_pred, title="Prediction Error Distribution")
print_error_metrics(y_test, y_pred)

