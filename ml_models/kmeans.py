"""
K-Means Clustering implemented from scratch using NumPy
"""
import numpy as np


class KMeans:
    """
    K-Means Clustering Algorithm
    
    Parameters:
    -----------
    n_clusters : int, default=3
        Number of clusters
    max_iterations : int, default=100
        Maximum number of iterations
    tolerance : float, default=1e-4
        Convergence tolerance
    random_state : int, default=None
        Random seed for reproducibility
    """
    
    def __init__(self, n_clusters=3, max_iterations=100, tolerance=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
    
    def fit(self, X):
        """
        Fit the K-Means model
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        
        # Initialize centroids randomly from data points
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_indices]
        
        for iteration in range(self.max_iterations):
            # Assign samples to closest centroid
            distances = self._compute_distances(X)
            labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.array([
                X[labels == k].mean(axis=0) if np.sum(labels == k) > 0 else self.centroids[k]
                for k in range(self.n_clusters)
            ])
            
            # Check convergence
            centroid_shift = np.sqrt(np.sum((new_centroids - self.centroids) ** 2))
            if centroid_shift < self.tolerance:
                break
            
            self.centroids = new_centroids
        
        # Final assignments
        distances = self._compute_distances(X)
        self.labels_ = np.argmin(distances, axis=1)
        
        # Calculate inertia (within-cluster sum of squares)
        self.inertia_ = np.sum([
            distances[i, self.labels_[i]] ** 2 for i in range(n_samples)
        ])
        
        return self
    
    def _compute_distances(self, X):
        """
        Compute distances between samples and centroids
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data points
            
        Returns:
        --------
        distances : array, shape (n_samples, n_clusters)
            Distances from each sample to each centroid
        """
        distances = np.zeros((X.shape[0], self.n_clusters))
        
        for k in range(self.n_clusters):
            distances[:, k] = np.sqrt(np.sum((X - self.centroids[k]) ** 2, axis=1))
        
        return distances
    
    def predict(self, X):
        """
        Predict cluster labels for samples
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test data
            
        Returns:
        --------
        labels : array, shape (n_samples,)
            Cluster labels
        """
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)
    
    def fit_predict(self, X):
        """
        Fit the model and predict cluster labels
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        labels : array, shape (n_samples,)
            Cluster labels
        """
        self.fit(X)
        return self.labels_
