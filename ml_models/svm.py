"""
Support Vector Machine implemented from scratch using NumPy
"""
import numpy as np


class SupportVectorMachine:
    """
    Support Vector Machine (SVM) Classifier using gradient descent
    
    Parameters:
    -----------
    learning_rate : float, default=0.001
        Learning rate for gradient descent
    lambda_param : float, default=0.01
        Regularization parameter
    n_iterations : int, default=1000
        Number of iterations for gradient descent
    """
    
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []
    
    def fit(self, X, y):
        """
        Fit the SVM model
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values (should be -1 or 1)
        """
        n_samples, n_features = X.shape
        
        # Convert labels to -1 and 1 if they are 0 and 1
        y_ = np.where(y <= 0, -1, 1)
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for iteration in range(self.n_iterations):
            loss = 0
            
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) + self.bias) >= 1
                
                if condition:
                    # No misclassification
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    # Misclassification
                    self.weights -= self.learning_rate * (
                        2 * self.lambda_param * self.weights - np.dot(x_i, y_[idx])
                    )
                    self.bias -= self.learning_rate * y_[idx]
                    loss += 1 - y_[idx] * (np.dot(x_i, self.weights) + self.bias)
            
            # Calculate hinge loss
            avg_loss = loss / n_samples
            self.losses.append(avg_loss)
        
        return self
    
    def predict(self, X):
        """
        Make predictions using the SVM model
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test data
            
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Predicted class labels (0 or 1)
        """
        linear_output = np.dot(X, self.weights) + self.bias
        predictions = np.sign(linear_output)
        # Convert back to 0 and 1
        return np.where(predictions <= 0, 0, 1)
    
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
