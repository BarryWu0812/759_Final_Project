#include <iostream>
#include <stddef.h>
#include <stdexcept>
#include <numeric>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>

using std::vector;

// StockSVR class for support vector regression
double dot_product(const std::vector<double>& v1, const std::vector<double>& v2)
{
    double result = 0.0;
    for (size_t i = 0; i < v1.size(); i++)
    {
        result += v1[i] * v2[i];
    }
    return result;
}

class StockSVR {
private:
    double lr;       // Learning rate
    double C;        // Regularization penalty
    int n_iters;     // Number of iterations for training
    double epsilon;  // Epsilon-insensitive margin
    std::vector<double> w;  // Weights
    double b;        // Bias

public:
    StockSVR(double learning_rate, double penalty, int iterations, double eps)
        : lr(learning_rate), C(penalty), n_iters(iterations), epsilon(eps), b(0.0) {}

    // Train the SVR model
    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
        int n_samples = X.size();
        int n_features = X[0].size();

        // Initialize weights and bias
        w.resize(n_features, 0.0);

        // Gradient Descent
        for (int iter = 0; iter < n_iters; iter++)
        {
            for (int i = 0; i < n_samples; i++)
            {
                // Compute the predicted value
                double y_pred = dot_product(X[i], w) + b;
                double loss = y[i] - y_pred;

                // Update weights and bias based on loss
                if (std::abs(loss) > epsilon)
                {
                    for (int j = 0; j < n_features; ++j)
                    {
                        w[j] -= lr * (w[j] - C * X[i][j] * sign(loss));
                    }
                    b -= lr * (-C * sign(loss));
                }
                else
                {
                    for (int j = 0; j < n_features; ++j)
                    {
                        w[j] -= lr * w[j];
                    }
                }
            }
        }
    }

    // Predict values for new data
    vector<double> predict(const vector<vector<double>>& X)
    {
        std::vector<double> predictions;
        for (const auto& x : X)
        {
            predictions.push_back(dot_product(x, w) + b);
        }
        return predictions;
    }

private:
    // Helper function: Compute the sign of a value
    int sign(double value)
    {
        return (value > 0) - (value < 0);
    }
};

class StandardScaler {
private:
    std::vector<double> means_;
    std::vector<double> std_devs_;
    bool is_fitted_;

    double compute_mean(const std::vector<double>& data) {
        return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    }

    double compute_std_dev(const std::vector<double>& data, double mean) {
        double variance = 0.0;
        for (const auto& value : data) {
            variance += (value - mean) * (value - mean);
        }
        return std::sqrt(variance / data.size());
    }

public:
    StandardScaler() : is_fitted_(false) {}

    // Fit for 1D or 2D data
    void fit(const std::vector<std::vector<double>>& X) {
        if (X.empty() || X[0].empty()) {
            throw std::invalid_argument("Input matrix is empty");
        }

        size_t n_features = X[0].size();
        means_.resize(n_features, 0);
        std_devs_.resize(n_features, 0);

        for (size_t j = 0; j < n_features; j++) {
            std::vector<double> column;
            for (const auto& row : X) {
                column.push_back(row[j]);
            }
            means_[j] = compute_mean(column);
            std_devs_[j] = compute_std_dev(column, means_[j]);
        }
        is_fitted_ = true;
    }

    void fit(const std::vector<double>& X) {
        means_.resize(1, compute_mean(X));
        std_devs_.resize(1, compute_std_dev(X, means_[0]));
        is_fitted_ = true;
    }

    // Transform for 1D or 2D data
    std::vector<double> transform(const std::vector<double>& X) const {
        if (!is_fitted_ || means_.size() != 1 || std_devs_.size() != 1) {
            throw std::logic_error("Scaler has not been fitted for 1D data");
        }

        std::vector<double> transformed(X.size());
        for (size_t i = 0; i < X.size(); i++) {
            transformed[i] = (X[i] - means_[0]) / std_devs_[0];
        }
        return transformed;
    }

    vector<vector<double>> transform(const vector<vector<double>>& X) const {
        if (!is_fitted_) {
            throw std::logic_error("Scaler has not been fitted for 2D data");
        }

        std::vector<std::vector<double>> transformed(X.size(), std::vector<double>(X[0].size(), 0));
        for (size_t i = 0; i < X.size(); i++) {
            for (size_t j = 0; j < X[0].size(); j++) {
                transformed[i][j] = (X[i][j] - means_[j]) / std_devs_[j];
            }
        }
        return transformed;
    }

    // Inverse Transform for 1D or 2D data
    std::vector<double> inverse_transform(const std::vector<double>& X) const {
        if (!is_fitted_ || means_.size() != 1 || std_devs_.size() != 1) {
            throw std::logic_error("Scaler has not been fitted for 1D data");
        }

        std::vector<double> original(X.size());
        for (size_t i = 0; i < X.size(); i++) {
            original[i] = X[i] * std_devs_[0] + means_[0];
        }
        return original;
    }

    std::vector<std::vector<double>> inverse_transform(const std::vector<std::vector<double>>& X) const {
        if (!is_fitted_) {
            throw std::logic_error("Scaler has not been fitted for 2D data");
        }

        std::vector<std::vector<double>> original(X.size(), std::vector<double>(X[0].size(), 0));
        for (size_t i = 0; i < X.size(); i++) {
            for (size_t j = 0; j < X[0].size(); j++) {
                original[i][j] = X[i][j] * std_devs_[j] + means_[j];
            }
        }
        return original;
    }
};


// Prepare data for training (rolling window)
void prepare_data(const vector<double>& data, int n_days, vector<vector<double>>& X, vector<double>& y)
{
    int n_samples = data.size() - n_days;

    // Resize X and y
    X.resize(n_samples, std::vector<double>(n_days));
    y.resize(n_samples);

    // Populate X and y
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_days; j++) {
            X[i][j] = data[i + j];  // Create rolling window features
        }
        y[i] = data[i + n_days];    // Target value
    }
}

// Normalize the data
// void normalize(std::vector<std::vector<double>>& X, std::vector<double>& y)
// {
//     // Normalize features (column-wise)
//     for (size_t j = 0; j < X[0].size(); j++)
//     {
//         double mean = 0.0, stddev = 0.0;
//         for (size_t i = 0; i < X.size(); i++)
//         {
//             mean += X[i][j];
//         }
//         mean /= X.size();
//         for (size_t i = 0; i < X.size(); i++)
//         {
//             stddev += (X[i][j] - mean) * (X[i][j] - mean);
//         }
//         stddev = std::sqrt(stddev / X.size());
//         for (size_t i = 0; i < X.size(); i++)
//         {
//             X[i][j] = (X[i][j] - mean) / (stddev + 1e-8);
//         }
//     }

//     // Normalize target
//     double mean_y = 0.0, stddev_y = 0.0;
//     for (size_t i = 0; i < y.size(); i++)
//     {
//         mean_y += y[i];
//     }
//     mean_y /= y.size();
//     for (size_t i = 0; i < y.size(); ++i) {
//         stddev_y += (y[i] - mean_y) * (y[i] - mean_y);
//     }
//     stddev_y = std::sqrt(stddev_y / y.size());
//     for (size_t i = 0; i < y.size(); ++i) {
//         y[i] = (y[i] - mean_y) / (stddev_y + 1e-8);
//     }
// }

vector<double> read_data(const std::string& filename)
{
    std::vector<double> data;
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        // Convert line to double and store it
        data.push_back(std::stod(line));
    }

    return data;
}

// Save data to CSV for plotting in Python
void save_to_csv(const std::string& filename, const std::vector<double>& data)
{
    std::ofstream file(filename);
    for (const auto& value : data) {
        file << value << "\n";
    }
    file.close();
}

// Main function
int main() {
    std::string data_filename = "AAPL_2024-11-28.csv";
    vector<double> data = read_data(data_filename);
    int n_days = 5; // Number of days to use as features

    // Prepare the data
    vector<vector<double>> X;
    vector<double> y;
    prepare_data(data, n_days, X, y);

    // Split into training and testing sets
    int split_idx = static_cast<int>(0.8 * X.size());
    std::vector<std::vector<double>> X_train(X.begin(), X.begin() + split_idx);
    std::vector<std::vector<double>> X_test(X.begin() + split_idx, X.end());
    std::vector<double> y_train(y.begin(), y.begin() + split_idx);
    std::vector<double> y_test(y.begin() + split_idx, y.end());

    // Fit scalers
    StandardScaler scaler_X, scaler_y;
    scaler_X.fit(X_train);
    scaler_y.fit(y_train);

    vector<vector<double>> X_scaled = scaler_X.transform(X_train);
    vector<double> y_scaled = scaler_y.transform(y_train);

    // Normalize the data
    // normalize(X_train, y_train);

    // Train SVR model
    StockSVR svr(0.001, 1.0, 1000, 0.1);
    svr.fit(X_scaled, y_scaled);

    // Make predictions
    X_test = scaler_X.transform(X_test);
    vector<double> predictions = svr.predict(X_test);
    predictions = scaler_y.inverse_transform(predictions);
    // y_test = scaler_y.inverse_transform(y_test);

    // Print results
    std::cout << "True vs Predicted:" << std::endl;
    for (size_t i = 0; i < y_test.size(); ++i) {
        std::cout << "True: " << y_test[i] << ", Predicted: " << predictions[i] << std::endl;
    }

    save_to_csv("y_test.csv", y_test);
    save_to_csv("predictions.csv", predictions);

    std::cout << "Data saved to CSV files for plotting in Python." << std::endl;


    return 0;
}