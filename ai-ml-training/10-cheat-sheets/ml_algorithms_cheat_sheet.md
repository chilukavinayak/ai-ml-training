# Machine Learning Algorithms Cheat Sheet

## 📊 Algorithm Selection Guide

### Supervised Learning

| Problem Type | Algorithm | Use Case | Pros | Cons |
|--------------|-----------|----------|------|------|
| **Regression** | Linear Regression | Simple relationships | Fast, interpretable | Assumes linearity |
| | Random Forest | Non-linear, feature importance | Robust, handles overfitting | Less interpretable |
| | Gradient Boosting | High accuracy needed | Excellent performance | Computationally expensive |
| **Classification** | Logistic Regression | Binary classification | Fast, probabilistic output | Limited to linear boundaries |
| | SVM | High-dimensional data | Works well with small datasets | Slow on large datasets |
| | Random Forest | General classification | Good performance, interpretable | Can overfit |
| | Neural Networks | Complex patterns | Very flexible | Requires lots of data |

### Unsupervised Learning

| Algorithm | Use Case | Pros | Cons |
|-----------|----------|------|------|
| K-Means | Customer segmentation | Simple, fast | Need to specify K |
| Hierarchical | Unknown number of clusters | No need to specify clusters | Computationally expensive |
| DBSCAN | Irregular shaped clusters | Finds outliers | Sensitive to parameters |
| PCA | Dimensionality reduction | Reduces complexity | Loses interpretability |

## 🔧 Quick Implementation Templates

### Linear Regression
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

### Logistic Regression
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
```

### Random Forest
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Feature importance
importance = model.feature_importances_
```

### Support Vector Machine
```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Scale features (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = SVC(kernel='rbf')
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
```

### K-Means Clustering
```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Find optimal number of clusters (Elbow Method)
inertias = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# Plot elbow curve
plt.plot(k_range, inertias)
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')

# Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)
```

### Principal Component Analysis (PCA)
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)  # Reduce to 2D
X_pca = pca.fit_transform(X_scaled)

# Explained variance ratio
print(f"Explained variance: {pca.explained_variance_ratio_}")
```

## 📈 Model Evaluation Metrics

### Regression Metrics
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
rmse = np.sqrt(mse)
```

### Classification Metrics
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print(classification_report(y_true, y_pred))
```

### Cross-Validation
```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

# For classification
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# For regression
cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
```

## 🔧 Hyperparameter Tuning

### Grid Search
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy'
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

### Random Search
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(3, 10),
    'min_samples_split': randint(2, 11)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist,
    n_iter=20,
    cv=5,
    scoring='accuracy'
)
```

## ⚠️ Common Pitfalls & Best Practices

### Data Leakage Prevention
```python
# ❌ Wrong: Transform entire dataset first
X_transformed = scaler.fit_transform(X)
X_train, X_test = train_test_split(X_transformed, y)

# ✅ Correct: Fit on training, transform both
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Feature Scaling
```python
# When to scale:
# - SVM, KNN, Neural Networks ✅
# - Tree-based algorithms ❌ (not necessary)

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standard Scaling (mean=0, std=1)
scaler = StandardScaler()

# Min-Max Scaling (range 0-1)
scaler = MinMaxScaler()
```

### Handling Imbalanced Data
```python
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

# Class weights
class_weights = compute_class_weight('balanced', 
                                   classes=np.unique(y), 
                                   y=y)

# SMOTE oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```
