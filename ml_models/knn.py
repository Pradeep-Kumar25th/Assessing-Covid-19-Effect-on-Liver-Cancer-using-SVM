"""
K-Nearest Neighbors implemented from scratch using NumPy
"""
import numpy as np
from collections import Counter


class KNearestNeighbors:
    """
    K-Nearest Neighbors Classifier
    
    Parameters:
    -----------
    k : int, default=5
        Number of neighbors to consider
    metric : str, default='euclidean'
        Distance metric ('euclidean' or 'manhattan')
    """
    
    def __init__(self, k=5, metric='euclidean'):
        self.k = k
        self.metric = metric
        self.X_train = None
        self.y_train = None
    
    def _euclidean_distance(self, x1, x2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))
    
    def _manhattan_distance(self, x1, x2):
        """Calculate Manhattan distance between two points"""
        return np.sum(np.abs(x1 - x2), axis=1)
    
    def fit(self, X, y):
        """
        Fit the KNN model (store training data)
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        """
        self.X_train = X
        self.y_train = y
        return self
    
    def predict(self, X):
        """
        Make predictions for test data
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test data
            
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Predicted class labels
        """
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)
    
    def _predict_single(self, x):
        """Predict class label for a single sample"""
        # Calculate distances to all training samples
        if self.metric == 'euclidean':
            distances = self._euclidean_distance(self.X_train, x)
        elif self.metric == 'manhattan':
            distances = self._manhattan_distance(self.X_train, x)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        
        # Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Get labels of k nearest neighbors
        k_nearest_labels = self.y_train[k_indices]
        
        # Return most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
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
