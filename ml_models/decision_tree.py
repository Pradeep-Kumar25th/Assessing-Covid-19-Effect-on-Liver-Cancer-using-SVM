"""
Decision Tree implemented from scratch using NumPy
"""
import numpy as np
from collections import Counter


class Node:
    """Node class for the decision tree"""
    
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # Feature index to split on
        self.threshold = threshold  # Threshold value for split
        self.left = left           # Left child node
        self.right = right         # Right child node
        self.value = value         # Value if leaf node


class DecisionTree:
    """
    Decision Tree Classifier using CART algorithm
    
    Parameters:
    -----------
    max_depth : int, default=10
        Maximum depth of the tree
    min_samples_split : int, default=2
        Minimum number of samples required to split a node
    n_features : int, default=None
        Number of features to consider for best split (for Random Forest)
    """
    
    def __init__(self, max_depth=10, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.root = None
    
    def fit(self, X, y):
        """
        Build the decision tree
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        """
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)
        return self
    
    def _grow_tree(self, X, y, depth=0):
        """Recursively grow the decision tree"""
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))
        
        # Stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Select random features
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        
        # Find best split
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)
        
        # Create child nodes
        left_idxs = X[:, best_feature] < best_thresh
        right_idxs = ~left_idxs
        
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        
        return Node(best_feature, best_thresh, left, right)
    
    def _best_split(self, X, y, feat_idxs):
        """Find the best feature and threshold to split on"""
        best_gain = -1
        split_idx, split_threshold = None, None
        
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            
            for thr in thresholds:
                # Calculate information gain
                gain = self._information_gain(y, X_column, thr)
                
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr
        
        return split_idx, split_threshold
    
    def _information_gain(self, y, X_column, threshold):
        """Calculate information gain of a split"""
        # Parent entropy
        parent_entropy = self._entropy(y)
        
        # Create children
        left_idxs = X_column < threshold
        right_idxs = ~left_idxs
        
        if np.sum(left_idxs) == 0 or np.sum(right_idxs) == 0:
            return 0
        
        # Calculate weighted average entropy of children
        n = len(y)
        n_l, n_r = np.sum(left_idxs), np.sum(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        
        # Calculate information gain
        information_gain = parent_entropy - child_entropy
        return information_gain
    
    def _entropy(self, y):
        """Calculate entropy of a node"""
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])
    
    def _most_common_label(self, y):
        """Return the most common class label"""
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
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
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        """Traverse the tree to make a prediction for a single sample"""
        if node.value is not None:
            return node.value
        
        if x[node.feature] < node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    def score(self, X, y):
        """Calculate accuracy score"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
