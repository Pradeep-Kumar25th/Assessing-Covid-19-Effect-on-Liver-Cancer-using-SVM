# Quick Start Guide - ML Models from Scratch

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Tests
```bash
python3 test_models.py
```

### Step 3: Explore Examples
```bash
jupyter notebook ML_Models_Complete_Example.ipynb
```

## 📖 Basic Usage

### Import Models
```python
from ml_models import (
    LinearRegression,
    LogisticRegression,
    KNearestNeighbors,
    DecisionTree,
    RandomForest,
    SupportVectorMachine,
    NeuralNetwork,
    KMeans
)
```

### Classification Example
```python
# Prepare your data
X_train, X_test, y_train, y_test = # your data

# Train a model
model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
```

### Regression Example
```python
# Train Linear Regression
model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# R² Score
r2 = model.score(X_test, y_test)
print(f"R² Score: {r2:.4f}")
```

### Clustering Example
```python
# Train K-Means
model = KMeans(n_clusters=3, random_state=42)
cluster_labels = model.fit_predict(X)

print(f"Inertia: {model.inertia_}")
print(f"Centroids shape: {model.centroids.shape}")
```

## 🎯 Model Selection Guide

| Task | Recommended Model | Why? |
|------|------------------|------|
| **Binary Classification** | Logistic Regression or Neural Network | Fast training, good accuracy |
| **Multiclass (small data)** | KNN | Simple, no training needed |
| **Interpretability needed** | Decision Tree | Easy to understand rules |
| **High accuracy needed** | Random Forest or Neural Network | Ensemble/Deep learning power |
| **Regression** | Linear Regression | Fast, simple, effective |
| **Clustering** | K-Means | Efficient, scalable |

## 🔧 Common Parameters

### All Classification Models
- `fit(X, y)` - Train the model
- `predict(X)` - Make predictions
- `score(X, y)` - Calculate accuracy

### Logistic Regression & Neural Network
- `predict_proba(X)` - Get probability estimates

### Linear Regression
- `score(X, y)` - Returns R² score (not accuracy)

## 📊 Feature Scaling

**Important:** Some models require feature scaling:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Models that need scaling:**
- ✅ Logistic Regression
- ✅ SVM
- ✅ Neural Network
- ✅ K-Means

**Models that don't need scaling:**
- ❌ Decision Tree
- ❌ Random Forest
- ❌ KNN (optional, but can help)

## 🎨 Visualization Examples

### Plot Training Loss
```python
import matplotlib.pyplot as plt

model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
model.fit(X_train, y_train)

plt.plot(model.losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
```

### Confusion Matrix
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.show()
```

## ⚙️ Hyperparameter Tuning

### Linear/Logistic Regression
```python
model = LogisticRegression(
    learning_rate=0.1,        # Try: 0.001, 0.01, 0.1, 1.0
    n_iterations=1000,        # Try: 500, 1000, 2000
    regularization='l2',      # Try: None, 'l1', 'l2'
    lambda_reg=0.01          # Try: 0.001, 0.01, 0.1
)
```

### KNN
```python
model = KNearestNeighbors(
    k=5,                     # Try: 3, 5, 7, 11
    metric='euclidean'       # Try: 'euclidean', 'manhattan'
)
```

### Decision Tree
```python
model = DecisionTree(
    max_depth=10,            # Try: 5, 10, 15, 20
    min_samples_split=2      # Try: 2, 5, 10
)
```

### Random Forest
```python
model = RandomForest(
    n_trees=10,              # Try: 5, 10, 50, 100
    max_depth=10,            # Try: 5, 10, 15
    min_samples_split=2,
    n_features=None          # Try: sqrt(n_features)
)
```

### Neural Network
```python
model = NeuralNetwork(
    hidden_layers=[64, 32],  # Try: [32], [64, 32], [128, 64, 32]
    learning_rate=0.1,       # Try: 0.01, 0.1, 1.0
    n_iterations=1000,       # Try: 500, 1000, 2000
    activation='relu'        # Try: 'relu', 'sigmoid', 'tanh'
)
```

### K-Means
```python
model = KMeans(
    n_clusters=3,            # Try: 2, 3, 5, 10
    max_iterations=100,
    random_state=42
)
```

## 📁 Files Overview

| File | Purpose |
|------|---------|
| `test_models.py` | Quick test of all models |
| `ML_Models_Complete_Example.ipynb` | Detailed examples with visualizations |
| `ML_Models_README.md` | Full documentation |
| `QUICK_START_GUIDE.md` | This file |
| `PROJECT_SUMMARY.md` | Complete project overview |

## 💡 Tips

1. **Start Simple**: Begin with Logistic Regression or Decision Tree
2. **Scale Features**: Use StandardScaler for gradient-based methods
3. **Monitor Training**: Check the loss curves to ensure convergence
4. **Compare Models**: Try multiple algorithms on your data
5. **Tune Hyperparameters**: Experiment with different parameters
6. **Validate Results**: Always use train/test split

## 🐛 Troubleshooting

### Model not converging?
- Increase `n_iterations`
- Adjust `learning_rate` (try smaller values)
- Scale your features

### Low accuracy?
- Try a different model
- Tune hyperparameters
- Check data quality
- Add more features

### SVM giving random results?
- Increase `n_iterations`
- Decrease `learning_rate`
- Make sure features are scaled

### Neural Network not learning?
- Check learning rate (try 0.01 to 0.1)
- Increase iterations
- Try different activation functions
- Verify features are scaled

## 🎯 Next Steps

1. **Run the test**: `python3 test_models.py`
2. **Open the notebook**: Jupyter with real examples
3. **Read the docs**: `ML_Models_README.md`
4. **Build something**: Apply to your own data!

---

**Happy Learning! 🚀**
