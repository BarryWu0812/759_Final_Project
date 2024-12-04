#ifndef MODEL_H
#define MODEL_H

#include <vector>

class DecisionTree {
public:
    DecisionTree(int max_depth = 5, int min_samples_split = 10);
    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y, int depth = 0);
    double predict(const std::vector<double>& x) const;

private:
    struct Node {
        int feature;
        double threshold;
        double value; // Leaf node value
        Node* left = nullptr;
        Node* right = nullptr;
    };

    Node* root;
    int max_depth;
    int min_samples_split;
    
    // Helper functions for tree building
    std::pair<int, double> find_best_split(const std::vector<std::vector<double>>& X, const std::vector<double>& y);
    double calculate_mse(const std::vector<double>& y_left, const std::vector<double>& y_right);


    Node* build_tree(const std::vector<std::vector<double>>& X, const std::vector<double>& y, int depth);
    double predict_sample(Node* node, const std::vector<double>& x) const;
};

class RandomForest {
public:
    RandomForest(int n_trees = 10, int max_depth = 5, int min_samples_split = 10);
    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y);
    std::vector<double> predict(const std::vector<std::vector<double>>& X) const;

private:
    std::vector<DecisionTree> trees;
    int n_trees;
    int max_depth;
    int min_samples_split;
};

#endif

