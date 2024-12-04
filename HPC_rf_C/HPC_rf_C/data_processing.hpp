#ifndef DATA_PROCESSING_H
#define DATA_PROCESSING_H

#include <vector>
#include <tuple>
#include <string>

// Load stock data (requires implementation using a stock API or local data)
std::vector<std::vector<double>> load_stock_data(const std::string& stock_symbol, const std::string& period, const std::string& interval);

// Preprocess stock data
std::tuple<std::vector<std::vector<double>>, std::vector<double>> preprocess_data(const std::vector<std::vector<double>>& data);

// Bootstrap sampling
std::tuple<std::vector<std::vector<double>>, std::vector<double>> bootstrap_sample(const std::vector<std::vector<double>>& X, const std::vector<double>& y);

#endif

