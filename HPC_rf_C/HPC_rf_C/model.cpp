#include "model.hpp"
#include "data_processing.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

DecisionTree::DecisionTree(int max_depth, int min_samples_split)
    : max_depth(max_depth), min_samples_split(min_samples_split), root(nullptr) {}

void DecisionTree::fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y, int depth) {
    root = build_tree(X, y, depth);
}

double DecisionTree::calculate_mse(const std::vector<double>& y_left, const std::vector<double>& y_right) {
    auto mse = [](const std::vector<double>& y) -> double {
        if (y.empty()) return 0.0;
        double mean = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
        double variance = std::accumulate(y.begin(), y.end(), 0.0, [mean](double acc, double val) {
            return acc + (val - mean) * (val - mean);
        });
        return variance / y.size();
    };
    double mse_left = mse(y_left) * y_left.size();
    double mse_right = mse(y_right) * y_right.size();
    return (mse_left + mse_right) / (y_left.size() + y_right.size());
}

std::pair<int, double> DecisionTree::find_best_split(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
    int best_feature = -1;
    double best_threshold = 0.0;
    double best_mse = std::numeric_limits<double>::infinity();

    for (size_t feature = 0; feature < X[0].size(); ++feature) {
        // Extract unique thresholds for the current feature
        std::vector<double> thresholds;
        for (const auto& row : X) thresholds.push_back(row[feature]);
        std::sort(thresholds.begin(), thresholds.end());
        thresholds.erase(std::unique(thresholds.begin(), thresholds.end()), thresholds.end());

        for (double threshold : thresholds) {
            // Split data into left and right subsets based on the threshold
            std::vector<double> y_left, y_right;
            for (size_t i = 0; i < X.size(); ++i) {
                if (X[i][feature] <= threshold) {
                    y_left.push_back(y[i]);
                } else {
                    y_right.push_back(y[i]);
                }
            }

            // Skip invalid splits
            if (y_left.empty() || y_right.empty()) continue;

            // Calculate the MSE for the split
            double mse = calculate_mse(y_left, y_right);
            if (mse < best_mse) {
                best_mse = mse;
                best_feature = feature;
                best_threshold = threshold;
            }
        }
        
    }

    return {best_feature, best_threshold};
}


DecisionTree::Node* DecisionTree::build_tree(const std::vector<std::vector<double>>& X, const std::vector<double>& y, int depth) {
    if (depth == max_depth || y.size() < min_samples_split || std::all_of(y.begin(), y.end(), [&](double val) { return val == y[0]; })) {
        double leaf_value = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
        
        return new Node{0, 0.0, leaf_value, nullptr, nullptr}; // Leaf node
        
    }

    // Find the best feature and threshold to split
    auto [best_feature, best_threshold] = find_best_split(X, y);
    if (best_feature == -1) {
        // If no valid split is found, create a leaf node
        double leaf_value = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
        return new Node{-1, 0.0, leaf_value, nullptr, nullptr};
    }

    std::vector<std::vector<double>> left_X, right_X;
    std::vector<double> left_y, right_y;

    // Partition data into left and right subsets (implement your split logic here)
    for (size_t i = 0; i < X.size(); ++i) {
        if (X[i][best_feature] <= best_threshold) {
            left_X.push_back(X[i]);
            left_y.push_back(y[i]);
        } else {
            right_X.push_back(X[i]);
            right_y.push_back(y[i]);
        }
    }

    Node* left_child = build_tree(left_X, left_y, depth + 1);
    Node* right_child = build_tree(right_X, right_y, depth + 1);
    return new Node{best_feature, best_threshold, 0.0, left_child, right_child};
}


double DecisionTree::predict(const std::vector<double>& x) const {
    return predict_sample(root, x);
}

double DecisionTree::predict_sample(Node* node, const std::vector<double>& x) const {
    if (!node) {
        throw std::runtime_error("Node is nullptr during prediction.");
    }
    if (!node->left && !node->right) return node->value; // Leaf node
    if (x[node->feature] <= node->threshold) {
        return predict_sample(node->left, x); // Go to left child
    }
    return predict_sample(node->right, x); // Go to right child
}


RandomForest::RandomForest(int n_trees, int max_depth, int min_samples_split)
    : n_trees(n_trees), max_depth(max_depth), min_samples_split(min_samples_split) {}

void RandomForest::fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
    for (int i = 0; i < n_trees; ++i) {
        auto [X_sample, y_sample] = bootstrap_sample(X, y);
        DecisionTree tree(max_depth, min_samples_split);
        tree.fit(X_sample, y_sample);
        trees.push_back(tree);
    }
}

std::vector<double> RandomForest::predict(const std::vector<std::vector<double>>& X) const {
    std::vector<double> predictions(X.size(), 0.0);
    for (const auto& tree : trees) {
        for (size_t i = 0; i < X.size(); ++i) {
            predictions[i] += tree.predict(X[i]);
        }
    }
    for (auto& pred : predictions) pred /= n_trees;
    return predictions;
}


