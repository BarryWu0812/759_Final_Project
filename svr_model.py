# svr_model.py

import numpy as np

class StockSVR:
    def __init__(self, learning_rate=0.001, C=1.0, n_iters=1000, epsilon=0.1):
        self.lr = learning_rate
        self.C = C  # Penalty parameter
        self.n_iters = n_iters
        self.epsilon = epsilon
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0

        # Gradient Descent
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Calculate the margin loss
                y_pred = np.dot(x_i, self.w) + self.b
                loss = y[idx] - y_pred

                # Compute the gradient for w and b
                if abs(loss) > self.epsilon:
                    # Hinge loss applies with penalty C
                    dw = self.w - self.C * x_i * np.sign(loss)
                    db = -self.C * np.sign(loss)
                else:
                    # No hinge loss, only regularization term
                    dw = self.w
                    db = 0

                # Update weights and bias
                self.w -= self.lr * dw
                self.b -= self.lr * db

    def predict(self, X):
        # Linear regression prediction
        return np.dot(X, self.w) + self.b


# Helper function to prepare stock data for training
def prepare_data(data, n_days):
    X, y = [], []
    for i in range(len(data) - n_days):
        X.append(data[i:i + n_days])  # Last n_days as features
        y.append(data[i + n_days])    # Next day's price as target

    # Convert to 2D numpy arrays
    X = np.array(X).reshape(-1, n_days)  # Reshape to (n_samples, n_days)
    y = np.array(y)
    
    return X, y