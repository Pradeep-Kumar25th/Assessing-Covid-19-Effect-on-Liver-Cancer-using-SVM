# Machine Learning Models from Scratch

A comprehensive collection of machine learning algorithms implemented from scratch using only NumPy. This project demonstrates the fundamental workings of various ML algorithms without relying on high-level libraries like scikit-learn.

## 🎯 Models Implemented

### Supervised Learning - Classification
1. **Logistic Regression** - Binary classification with gradient descent
2. **K-Nearest Neighbors (KNN)** - Instance-based learning algorithm
3. **Decision Tree** - CART algorithm with information gain
4. **Random Forest** - Ensemble of decision trees with bootstrap sampling
5. **Support Vector Machine (SVM)** - Maximum margin classifier with gradient descent
6. **Neural Network** - Feedforward network with backpropagation

### Supervised Learning - Regression
7. **Linear Regression** - Ordinary least squares with gradient descent

### Unsupervised Learning
8. **K-Means Clustering** - Centroid-based clustering algorithm

## 📁 Project Structure

```
.
├── ml_models/                  # Model implementations
│   ├── __init__.py
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   ├── knn.py
│   ├── decision_tree.py
│   ├── random_forest.py
│   ├── svm.py
│   ├── neural_network.py
│   └── kmeans.py
├── ML_Models_Complete_Example.ipynb  # Comprehensive examples
├── covid-liver.csv                   # Sample dataset
├── requirements.txt                  # Dependencies
└── README.md                         # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>

# Install dependencies
pip install -r requirements.txt
```

### Usage

```python
import numpy as np
from ml_models import LogisticRegression, DecisionTree, NeuralNetwork

# Load your data
X_train, X_test, y_train, y_test = # your data here

# Logistic Regression
lr = LogisticRegression(learning_rate=0.1, n_iterations=1000)
lr.fit(X_train, y_train)
accuracy = lr.score(X_test, y_test)
print(f"Accuracy: {accuracy}")

# Decision Tree
dt = DecisionTree(max_depth=10, min_samples_split=5)
dt.fit(X_train, y_train)
predictions = dt.predict(X_test)

# Neural Network
nn = NeuralNetwork(hidden_layers=[32, 16], learning_rate=0.1, n_iterations=1000)
nn.fit(X_train, y_train)
predictions = nn.predict(X_test)
```

## 📊 Model Details

### 1. Linear Regression
- **Algorithm**: Gradient Descent
- **Features**: L1/L2 Regularization
- **Use Case**: Continuous target prediction

```python
from ml_models import LinearRegression

model = LinearRegression(
    learning_rate=0.01,
    n_iterations=1000,
    regularization='l2',  # 'l1', 'l2', or None
    lambda_reg=0.01
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
r2_score = model.score(X_test, y_test)
```

### 2. Logistic Regression
- **Algorithm**: Gradient Descent with Sigmoid
- **Features**: Binary classification, probability estimates
- **Loss Function**: Binary Cross-Entropy

```python
from ml_models import LogisticRegression

model = LogisticRegression(
    learning_rate=0.1,
    n_iterations=1000,
    regularization='l2',
    lambda_reg=0.01
)
model.fit(X_train, y_train)
probabilities = model.predict_proba(X_test)
predictions = model.predict(X_test, threshold=0.5)
```

### 3. K-Nearest Neighbors
- **Algorithm**: Distance-based voting
- **Distance Metrics**: Euclidean, Manhattan
- **Non-parametric**: No training phase

```python
from ml_models import KNearestNeighbors

model = KNearestNeighbors(
    k=5,
    metric='euclidean'  # 'euclidean' or 'manhattan'
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### 4. Decision Tree
- **Algorithm**: CART (Classification and Regression Trees)
- **Split Criterion**: Information Gain (Entropy)
- **Features**: Max depth control, min samples split

```python
from ml_models import DecisionTree

model = DecisionTree(
    max_depth=10,
    min_samples_split=2,
    n_features=None  # None uses all features
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### 5. Random Forest
- **Algorithm**: Ensemble of Decision Trees
- **Features**: Bootstrap sampling, feature randomness
- **Prediction**: Majority voting

```python
from ml_models import RandomForest

model = RandomForest(
    n_trees=10,
    max_depth=10,
    min_samples_split=2,
    n_features=None  # sqrt(n_features) recommended
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### 6. Support Vector Machine
- **Algorithm**: Gradient Descent for SVM
- **Features**: Hinge loss, regularization
- **Use Case**: Binary classification

```python
from ml_models import SupportVectorMachine

model = SupportVectorMachine(
    learning_rate=0.001,
    lambda_param=0.01,
    n_iterations=1000
)
model.fit(X_train, y_train)  # y should be 0/1 or -1/1
predictions = model.predict(X_test)
```

### 7. Neural Network
- **Algorithm**: Backpropagation
- **Activation Functions**: ReLU, Sigmoid, Tanh
- **Features**: Customizable architecture

```python
from ml_models import NeuralNetwork

model = NeuralNetwork(
    hidden_layers=[64, 32],  # List of hidden layer sizes
    learning_rate=0.01,
    n_iterations=1000,
    activation='relu'  # 'relu', 'sigmoid', or 'tanh'
)
model.fit(X_train, y_train)
probabilities = model.predict_proba(X_test)
predictions = model.predict(X_test)
```

### 8. K-Means Clustering
- **Algorithm**: Centroid-based clustering
- **Features**: Configurable clusters, convergence tolerance
- **Use Case**: Unsupervised learning

```python
from ml_models import KMeans

model = KMeans(
    n_clusters=3,
    max_iterations=100,
    tolerance=1e-4,
    random_state=42
)
cluster_labels = model.fit_predict(X_train)
print(f"Inertia: {model.inertia_}")
```

## 📓 Example Notebook

Check out `ML_Models_Complete_Example.ipynb` for a comprehensive demonstration of all models on the COVID-19 liver cancer dataset. The notebook includes:

- Data preprocessing and exploration
- Training all models
- Model evaluation and comparison
- Visualization of results
- Decision boundary plots

## 🎓 Educational Purpose

This project is designed for:

- **Learning**: Understand how ML algorithms work under the hood
- **Teaching**: Use as educational material for ML courses
- **Research**: Modify and experiment with algorithm implementations
- **Interviews**: Demonstrate understanding of ML fundamentals

## 🔧 Implementation Details

### Key Features:
- ✅ Pure NumPy implementation (no sklearn models)
- ✅ Well-documented code with docstrings
- ✅ Consistent API across all models (fit, predict, score)
- ✅ Training history tracking (losses)
- ✅ Regularization support where applicable
- ✅ Efficient vectorized operations

### Data Requirements:
- Input: NumPy arrays (n_samples, n_features)
- Binary classification targets: 0/1 labels
- Feature scaling recommended for gradient-based methods

## 📈 Performance Considerations

| Model | Training Speed | Prediction Speed | Memory | Best Use Case |
|-------|---------------|------------------|--------|---------------|
| Linear Regression | Fast | Very Fast | Low | Linear relationships |
| Logistic Regression | Fast | Very Fast | Low | Binary classification |
| KNN | Very Fast | Slow | High | Small datasets |
| Decision Tree | Medium | Fast | Medium | Non-linear, interpretable |
| Random Forest | Slow | Medium | High | Complex patterns, ensembles |
| SVM | Medium | Fast | Medium | Binary classification |
| Neural Network | Slow | Fast | High | Complex non-linear patterns |
| K-Means | Fast | Fast | Low | Clustering, unsupervised |

## 🧪 Testing

To verify all models work correctly, run the test suite:

```bash
# Run the complete example notebook
jupyter notebook ML_Models_Complete_Example.ipynb

# Or run a quick test script
python test_models.py
```

## 📝 Dataset Information

The included `covid-liver.csv` dataset contains information about COVID-19 patients with liver conditions. Features include:

- Age, Gender
- Cancer type and stage
- Treatment methods
- Survival outcomes
- Clinical measurements

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Add more models (Naive Bayes, Gradient Boosting, etc.)
- Implement multiclass classification
- Add cross-validation support
- Optimize performance
- Add more comprehensive tests

## 📄 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- Algorithms based on standard ML textbooks and research papers
- Dataset from COVID-19 liver cancer research study
- Inspired by the need to understand ML fundamentals

## 📚 References

- "Pattern Recognition and Machine Learning" by Christopher Bishop
- "The Elements of Statistical Learning" by Hastie, Tibshirani, Friedman
- "Machine Learning" by Tom Mitchell
- Scikit-learn documentation (for API design inspiration)

---

**Note**: These implementations are for educational purposes. For production use, consider using optimized libraries like scikit-learn, TensorFlow, or PyTorch.
