#include "data_processing.hpp"
#include <vector>
#include <tuple>
#include <random>
#include <cmath>
#include <iostream>

std::vector<std::vector<double>> load_stock_data(const std::string& stock_symbol, const std::string& period, const std::string& interval) {
    // Placeholder implementation. Use a stock data API or local file loading.
    std::vector<std::vector<double>> data = {
        {100, 105, 110}, // Day 1
            {105, 110, 115}, // Day 2
            {110, 115, 120}, // Day 3
            {115, 120, 125}, // Day 4
            {120, 125, 130}, // Day 5
            {125, 130, 135}, // Day 6
            {130, 135, 140}, // Day 7
            {135, 140, 145}, // Day 8
            {140, 145, 150}, // Day 9
            {145, 150, 155}, // Day 10
            {150, 155, 160}, // Day 11
            {155, 160, 165}, // Day 12
            {160, 165, 170}, // Day 13
            {165, 170, 175}, // Day 14
            {170, 175, 180}, // Day 15
            {175, 180, 185}, // Day 16
            {180, 185, 190}, // Day 17
            {185, 190, 195}, // Day 18
            {190, 195, 200}, // Day 19
            {195, 200, 205}, // Day 20
            {200, 205, 210}, // Day 21
            {205, 210, 215}, // Day 22
            {210, 215, 220}, // Day 23
            {215, 220, 225}, // Day 24
            {220, 225, 230}, // Day 25
            {225, 230, 235}, // Day 26
            {230, 235, 240}, // Day 27
            {235, 240, 245}, // Day 28
            {240, 245, 250}, // Day 29
            {245, 250, 255}  // Day 30
    };
    std::cout << "Stock data loaded (stub): " << data.size() << " rows.\n";
    return data;
}

std::tuple<std::vector<std::vector<double>>, std::vector<double>> preprocess_data(const std::vector<std::vector<double>>& data) {
    std::vector<std::vector<double>> X; // Features
    std::vector<double> y;             // Target
    for (size_t i = 1; i < data.size(); ++i) {
        // Feature engineering: MA5, MA20, Volatility, Daily Return
        std::vector<double> row = data[i - 1]; // Using previous day as features
        X.push_back(row);
        y.push_back(data[i][0]);  // Next day's price (Close)
    }
    return {X, y};
}

std::tuple<std::vector<std::vector<double>>, std::vector<double>> bootstrap_sample(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
    size_t n_samples = static_cast<size_t>(X.size() * 0.7);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, X.size() - 1);

    std::vector<std::vector<double>> X_sample;
    std::vector<double> y_sample;
    for (size_t i = 0; i < n_samples; ++i) {
        int idx = dis(gen);
        X_sample.push_back(X[idx]);
        y_sample.push_back(y[idx]);
    }
    return {X_sample, y_sample};
}

