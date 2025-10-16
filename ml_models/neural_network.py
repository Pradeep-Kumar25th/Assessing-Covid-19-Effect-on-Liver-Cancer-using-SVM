"""
Neural Network implemented from scratch using NumPy
"""
import numpy as np


class NeuralNetwork:
    """
    Feedforward Neural Network with backpropagation
    
    Parameters:
    -----------
    hidden_layers : list of int, default=[64, 32]
        Number of neurons in each hidden layer
    learning_rate : float, default=0.01
        Learning rate for gradient descent
    n_iterations : int, default=1000
        Number of iterations for training
    activation : str, default='relu'
        Activation function ('relu', 'sigmoid', or 'tanh')
    """
    
    def __init__(self, hidden_layers=[64, 32], learning_rate=0.01, n_iterations=1000, activation='relu'):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.activation = activation
        self.weights = []
        self.biases = []
        self.losses = []
    
    def _initialize_weights(self, input_size, output_size):
        """Initialize weights and biases"""
        self.weights = []
        self.biases = []
        
        # Input to first hidden layer
        layer_sizes = [input_size] + self.hidden_layers + [output_size]
        
        for i in range(len(layer_sizes) - 1):
            # He initialization for ReLU, Xavier for others
            if self.activation == 'relu':
                w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            else:
                w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(1.0 / layer_sizes[i])
            
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def _relu(self, z):
        """ReLU activation function"""
        return np.maximum(0, z)
    
    def _relu_derivative(self, z):
        """Derivative of ReLU"""
        return (z > 0).astype(float)
    
    def _sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def _sigmoid_derivative(self, z):
        """Derivative of sigmoid"""
        s = self._sigmoid(z)
        return s * (1 - s)
    
    def _tanh(self, z):
        """Tanh activation function"""
        return np.tanh(z)
    
    def _tanh_derivative(self, z):
        """Derivative of tanh"""
        return 1 - np.tanh(z) ** 2
    
    def _activate(self, z):
        """Apply activation function"""
        if self.activation == 'relu':
            return self._relu(z)
        elif self.activation == 'sigmoid':
            return self._sigmoid(z)
        elif self.activation == 'tanh':
            return self._tanh(z)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def _activate_derivative(self, z):
        """Apply derivative of activation function"""
        if self.activation == 'relu':
            return self._relu_derivative(z)
        elif self.activation == 'sigmoid':
            return self._sigmoid_derivative(z)
        elif self.activation == 'tanh':
            return self._tanh_derivative(z)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def _forward_pass(self, X):
        """Perform forward pass through the network"""
        activations = [X]
        z_values = []
        
        for i in range(len(self.weights) - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            z_values.append(z)
            a = self._activate(z)
            activations.append(a)
        
        # Output layer (sigmoid for binary classification)
        z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        z_values.append(z)
        output = self._sigmoid(z)
        activations.append(output)
        
        return activations, z_values
    
    def _backward_pass(self, X, y, activations, z_values):
        """Perform backward pass (backpropagation)"""
        n_samples = X.shape[0]
        deltas = [None] * len(self.weights)
        
        # Output layer error
        output_error = activations[-1] - y.reshape(-1, 1)
        deltas[-1] = output_error
        
        # Hidden layers errors
        for i in range(len(self.weights) - 2, -1, -1):
            error = np.dot(deltas[i+1], self.weights[i+1].T)
            delta = error * self._activate_derivative(z_values[i])
            deltas[i] = delta
        
        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * np.dot(activations[i].T, deltas[i]) / n_samples
            self.biases[i] -= self.learning_rate * np.sum(deltas[i], axis=0, keepdims=True) / n_samples
    
    def fit(self, X, y):
        """
        Train the neural network
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values (binary: 0 or 1)
        """
        n_samples, n_features = X.shape
        n_outputs = 1  # Binary classification
        
        # Initialize weights
        self._initialize_weights(n_features, n_outputs)
        
        # Training loop
        for iteration in range(self.n_iterations):
            # Forward pass
            activations, z_values = self._forward_pass(X)
            
            # Calculate loss (Binary Cross-Entropy)
            epsilon = 1e-15
            y_pred = np.clip(activations[-1], epsilon, 1 - epsilon)
            loss = -np.mean(y * np.log(y_pred) + (1 - y.reshape(-1, 1)) * np.log(1 - y_pred))
            self.losses.append(loss)
            
            # Backward pass
            self._backward_pass(X, y, activations, z_values)
        
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
        activations, _ = self._forward_pass(X)
        return activations[-1].flatten()
    
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
