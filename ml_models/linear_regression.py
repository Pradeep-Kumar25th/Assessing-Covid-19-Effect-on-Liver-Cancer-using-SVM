"""
Linear Regression implemented from scratch using NumPy
"""
import numpy as np


class LinearRegression:
    """
    Linear Regression using Gradient Descent
    
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
    
    def fit(self, X, y):
        """
        Fit the linear regression model
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        """
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Predictions
            y_pred = np.dot(X, self.weights) + self.bias
            
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
            
            # Calculate loss (MSE)
            loss = np.mean((y_pred - y) ** 2)
            if self.regularization == 'l2':
                loss += (self.lambda_reg / (2 * n_samples)) * np.sum(self.weights ** 2)
            elif self.regularization == 'l1':
                loss += (self.lambda_reg / n_samples) * np.sum(np.abs(self.weights))
            
            self.losses.append(loss)
        
        return self
    
    def predict(self, X):
        """
        Make predictions using the linear model
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test data
            
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Predicted values
        """
        return np.dot(X, self.weights) + self.bias
    
    def score(self, X, y):
        """
        Calculate R² score
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test data
        y : array-like, shape (n_samples,)
            True values
            
        Returns:
        --------
        score : float
            R² score
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
