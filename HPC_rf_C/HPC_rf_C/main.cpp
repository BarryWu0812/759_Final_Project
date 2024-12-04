#include <iostream>
#include "data_processing.hpp"
#include "model.hpp"

int main() {
    // Load data
    auto data = load_stock_data("ZETA", "6mo", "1d");
    auto [X, y] = preprocess_data(data);

    // Train-test split
    size_t train_size = static_cast<size_t>(X.size() * 0.8);
    std::vector<std::vector<double>> X_train(X.begin(), X.begin() + train_size);
    std::vector<std::vector<double>> X_test(X.begin() + train_size, X.end());
    std::vector<double> y_train(y.begin(), y.begin() + train_size);
    std::vector<double> y_test(y.begin() + train_size, y.end());

    // Train Random Forest
    RandomForest rf(100, 20, 2);
    rf.fit(X_train, y_train);

    // Predict and evaluate
    auto y_pred = rf.predict(X_test);

    // Print results
    for (size_t i = 0; i < y_test.size(); ++i) {
        std::cout << "Actual: " << y_test[i] << ", Predicted: " << y_pred[i] << std::endl;
    }

    return 0;
}

