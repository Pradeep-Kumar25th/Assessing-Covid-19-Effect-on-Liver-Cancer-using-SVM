"""
Random Forest implemented from scratch using NumPy
"""
import numpy as np
from collections import Counter
from .decision_tree import DecisionTree


class RandomForest:
    """
    Random Forest Classifier
    
    Parameters:
    -----------
    n_trees : int, default=10
        Number of trees in the forest
    max_depth : int, default=10
        Maximum depth of each tree
    min_samples_split : int, default=2
        Minimum number of samples required to split a node
    n_features : int, default=None
        Number of features to consider for best split
    """
    
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []
    
    def fit(self, X, y):
        """
        Build a random forest of decision trees
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        """
        self.trees = []
        
        for _ in range(self.n_trees):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features
            )
            
            # Bootstrap sampling
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        
        return self
    
    def _bootstrap_samples(self, X, y):
        """Create a bootstrap sample from the dataset"""
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]
    
    def predict(self, X):
        """
        Make predictions using the random forest
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test data
            
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Predicted class labels
        """
        # Get predictions from all trees
        predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Transpose to get predictions for each sample
        tree_preds = np.swapaxes(predictions, 0, 1)
        
        # Take majority vote for each sample
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions
    
    def _most_common_label(self, y):
        """Return the most common class label"""
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def score(self, X, y):
        """Calculate accuracy score"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
