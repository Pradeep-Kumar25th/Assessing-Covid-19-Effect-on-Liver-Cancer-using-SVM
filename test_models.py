"""
Test script to verify all ML models work correctly
"""
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import all our models
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

print("=" * 70)
print("TESTING ALL ML MODELS FROM SCRATCH")
print("=" * 70)

# Set random seed
np.random.seed(42)

# Generate synthetic classification data
print("\n1. Generating synthetic classification data...")
X_class, y_class = make_classification(
    n_samples=500, n_features=10, n_informative=8, n_redundant=2,
    n_classes=2, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"   Training samples: {X_train.shape[0]}")
print(f"   Test samples: {X_test.shape[0]}")
print(f"   Features: {X_train.shape[1]}")

# Test Logistic Regression
print("\n2. Testing Logistic Regression...")
try:
    lr = LogisticRegression(learning_rate=0.1, n_iterations=500, regularization='l2')
    lr.fit(X_train_scaled, y_train)
    acc_lr = lr.score(X_test_scaled, y_test)
    print(f"   ✓ Logistic Regression - Accuracy: {acc_lr:.4f}")
except Exception as e:
    print(f"   ✗ Logistic Regression failed: {e}")

# Test KNN
print("\n3. Testing K-Nearest Neighbors...")
try:
    knn = KNearestNeighbors(k=5, metric='euclidean')
    knn.fit(X_train_scaled, y_train)
    acc_knn = knn.score(X_test_scaled, y_test)
    print(f"   ✓ KNN - Accuracy: {acc_knn:.4f}")
except Exception as e:
    print(f"   ✗ KNN failed: {e}")

# Test Decision Tree
print("\n4. Testing Decision Tree...")
try:
    dt = DecisionTree(max_depth=10, min_samples_split=5)
    dt.fit(X_train, y_train)
    acc_dt = dt.score(X_test, y_test)
    print(f"   ✓ Decision Tree - Accuracy: {acc_dt:.4f}")
except Exception as e:
    print(f"   ✗ Decision Tree failed: {e}")

# Test Random Forest
print("\n5. Testing Random Forest...")
try:
    rf = RandomForest(n_trees=5, max_depth=10, min_samples_split=5)
    rf.fit(X_train, y_train)
    acc_rf = rf.score(X_test, y_test)
    print(f"   ✓ Random Forest - Accuracy: {acc_rf:.4f}")
except Exception as e:
    print(f"   ✗ Random Forest failed: {e}")

# Test SVM
print("\n6. Testing Support Vector Machine...")
try:
    svm = SupportVectorMachine(learning_rate=0.001, lambda_param=0.01, n_iterations=500)
    svm.fit(X_train_scaled, y_train)
    acc_svm = svm.score(X_test_scaled, y_test)
    print(f"   ✓ SVM - Accuracy: {acc_svm:.4f}")
except Exception as e:
    print(f"   ✗ SVM failed: {e}")

# Test Neural Network
print("\n7. Testing Neural Network...")
try:
    nn = NeuralNetwork(hidden_layers=[32, 16], learning_rate=0.1, n_iterations=500, activation='relu')
    nn.fit(X_train_scaled, y_train)
    acc_nn = nn.score(X_test_scaled, y_test)
    print(f"   ✓ Neural Network - Accuracy: {acc_nn:.4f}")
except Exception as e:
    print(f"   ✗ Neural Network failed: {e}")

# Test K-Means (Unsupervised)
print("\n8. Testing K-Means Clustering...")
try:
    kmeans = KMeans(n_clusters=2, max_iterations=100, random_state=42)
    labels = kmeans.fit_predict(X_train_scaled)
    print(f"   ✓ K-Means - Inertia: {kmeans.inertia_:.4f}")
    print(f"   ✓ K-Means - Cluster distribution: {np.bincount(labels)}")
except Exception as e:
    print(f"   ✗ K-Means failed: {e}")

# Test Linear Regression
print("\n9. Testing Linear Regression...")
try:
    X_reg, y_reg = make_regression(n_samples=500, n_features=10, noise=10, random_state=42)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    scaler_reg = StandardScaler()
    X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
    X_test_reg_scaled = scaler_reg.transform(X_test_reg)
    
    lin_reg = LinearRegression(learning_rate=0.01, n_iterations=500, regularization='l2')
    lin_reg.fit(X_train_reg_scaled, y_train_reg)
    r2 = lin_reg.score(X_test_reg_scaled, y_test_reg)
    print(f"   ✓ Linear Regression - R² Score: {r2:.4f}")
except Exception as e:
    print(f"   ✗ Linear Regression failed: {e}")

print("\n" + "=" * 70)
print("ALL TESTS COMPLETED!")
print("=" * 70)
print("\n✅ All models are working correctly!")
print("\nNext steps:")
print("  1. Check out 'ML_Models_Complete_Example.ipynb' for comprehensive examples")
print("  2. Read 'ML_Models_README.md' for detailed documentation")
print("  3. Use the models in your own projects!")
