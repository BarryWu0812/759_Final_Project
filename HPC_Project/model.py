# model.py
import numpy as np
from data_processing import bootstrap_sample  # Ensure this import is correct


class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y, depth=0):
        if depth == self.max_depth or len(y) < self.min_samples_split or len(set(y)) == 1:
            # Leaf node: store mean value
            self.tree = np.mean(y)
        else:
            best_split = self._find_best_split(X, y)
            if best_split:
                feature, threshold = best_split
                indices_left = X[:, feature] <= threshold
                indices_right = X[:, feature] > threshold
                left_tree = DecisionTree(self.max_depth, self.min_samples_split)
                right_tree = DecisionTree(self.max_depth, self.min_samples_split)
                left_tree.fit(X[indices_left], y[indices_left], depth + 1)
                right_tree.fit(X[indices_right], y[indices_right], depth + 1)
                self.tree = (feature, threshold, left_tree, right_tree)
            else:
                self.tree = np.mean(y)

    def _find_best_split(self, X, y):
        best_mse = float("inf")
        best_split = None
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                mse = self._calculate_mse(X, y, feature, threshold)
                if mse < best_mse:
                    best_mse = mse
                    best_split = (feature, threshold)
        return best_split

    def _calculate_mse(self, X, y, feature, threshold):
        left = y[X[:, feature] <= threshold]
        right = y[X[:, feature] > threshold]
        if len(left) == 0 or len(right) == 0:
            return float("inf")
        mse_left = np.var(left) * len(left)
        mse_right = np.var(right) * len(right)
        return (mse_left + mse_right) / len(y)

    def predict(self, X):
        if isinstance(self.tree, tuple):
            feature, threshold, left_tree, right_tree = self.tree
            left_indices = X[:, feature] <= threshold
            right_indices = X[:, feature] > threshold
            predictions = np.zeros(X.shape[0])
            predictions[left_indices] = left_tree.predict(X[left_indices])
            predictions[right_indices] = right_tree.predict(X[right_indices])
            return predictions
        else:
            return np.full(X.shape[0], self.tree)
            
class RandomForest:
    def __init__(self, n_trees=10, max_depth=5, min_samples_split=10):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_trees):
            X_sample, y_sample = bootstrap_sample(X, y)
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(predictions, axis=0)
