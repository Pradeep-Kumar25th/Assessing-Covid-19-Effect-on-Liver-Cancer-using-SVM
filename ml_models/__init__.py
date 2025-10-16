"""
Machine Learning Models from Scratch
=====================================
A comprehensive library of machine learning algorithms implemented from scratch using only NumPy.
"""

from .linear_regression import LinearRegression
from .logistic_regression import LogisticRegression
from .knn import KNearestNeighbors
from .decision_tree import DecisionTree
from .random_forest import RandomForest
from .svm import SupportVectorMachine
from .neural_network import NeuralNetwork
from .kmeans import KMeans

__all__ = [
    'LinearRegression',
    'LogisticRegression',
    'KNearestNeighbors',
    'DecisionTree',
    'RandomForest',
    'SupportVectorMachine',
    'NeuralNetwork',
    'KMeans'
]

__version__ = '1.0.0'
