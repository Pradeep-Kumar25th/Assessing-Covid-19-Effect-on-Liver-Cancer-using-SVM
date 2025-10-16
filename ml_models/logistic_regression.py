"""
Logistic Regression implemented from scratch using NumPy
"""
import numpy as np


class LogisticRegression:
    """
    Logistic Regression using Gradient Descent
    
    Parameters:
    -----------
    learning_rate : float, default=0.01
        Learning rate for gradient descent
    n_iterations : int, default=1000
        Number of iterations for gradient descent
    regularization : str, default=None
        Type of regularization ('l1', 'l2', or None)
    lambda_reg : float, default=0.01
        Regularization parameter
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, regularization=None, lambda_reg=0.01):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None
        self.losses = []
    
    def _sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X, y):
        """
        Fit the logistic regression model
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values (binary: 0 or 1)
        """
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Linear combination
            linear_model = np.dot(X, self.weights) + self.bias
            # Apply sigmoid
            y_pred = self._sigmoid(linear_model)
            
            # Calculate gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Apply regularization
            if self.regularization == 'l2':
                dw += (self.lambda_reg / n_samples) * self.weights
            elif self.regularization == 'l1':
                dw += (self.lambda_reg / n_samples) * np.sign(self.weights)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Calculate loss (Binary Cross-Entropy)
            epsilon = 1e-15
            y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
            loss = -np.mean(y * np.log(y_pred_clipped) + (1 - y) * np.log(1 - y_pred_clipped))
            
            if self.regularization == 'l2':
                loss += (self.lambda_reg / (2 * n_samples)) * np.sum(self.weights ** 2)
            elif self.regularization == 'l1':
                loss += (self.lambda_reg / n_samples) * np.sum(np.abs(self.weights))
            
            self.losses.append(loss)
        
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test data
            
        Returns:
        --------
        proba : array, shape (n_samples,)
            Probability of the positive class
        """
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)
    
    def predict(self, X, threshold=0.5):
        """
        Make binary predictions
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test data
        threshold : float, default=0.5
            Classification threshold
            
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Predicted class labels (0 or 1)
        """
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)
    
    def score(self, X, y):
        """
        Calculate accuracy score
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test data
        y : array-like, shape (n_samples,)
            True values
            
        Returns:
        --------
        accuracy : float
            Accuracy score
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
