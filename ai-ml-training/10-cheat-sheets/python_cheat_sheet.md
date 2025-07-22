# Python Cheat Sheet for AI/ML

## 📊 Data Types & Structures

```python
# Basic Data Types
integer = 42
float_num = 3.14
string = "Hello, World!"
boolean = True
none_type = None

# Collections
list_data = [1, 2, 3, 4]
tuple_data = (1, 2, 3, 4)
dict_data = {"key": "value", "age": 25}
set_data = {1, 2, 3, 4}

# List Comprehensions
squares = [x**2 for x in range(10)]
evens = [x for x in range(20) if x % 2 == 0]
```

## 🔧 Essential Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import tensorflow as tf
import torch
```

## 📈 NumPy Essentials

```python
# Array Creation
arr = np.array([1, 2, 3, 4, 5])
zeros = np.zeros((3, 4))
ones = np.ones((2, 3))
random_arr = np.random.rand(3, 3)

# Array Operations
arr.shape           # Shape of array
arr.dtype           # Data type
arr.reshape(5, 1)   # Reshape
arr.mean()          # Mean
arr.std()           # Standard deviation
arr.sum()           # Sum

# Indexing & Slicing
arr[0]              # First element
arr[-1]             # Last element
arr[1:4]            # Slice
arr[arr > 3]        # Boolean indexing
```

## 📊 Pandas Essentials

```python
# DataFrame Creation
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
df = pd.read_csv('file.csv')

# Data Exploration
df.head()           # First 5 rows
df.tail()           # Last 5 rows
df.info()           # Data types and info
df.describe()       # Statistical summary
df.shape            # Dimensions
df.columns          # Column names

# Data Selection
df['column']        # Select column
df[['col1', 'col2']] # Multiple columns
df.iloc[0]          # First row by position
df.loc[0]           # First row by label
df[df['col'] > 5]   # Boolean filtering

# Data Manipulation
df.dropna()         # Remove NaN values
df.fillna(0)        # Fill NaN with value
df.groupby('col').mean()  # Group by operation
df.sort_values('col')     # Sort by column
```

## 📈 Matplotlib/Seaborn Quick Plots

```python
# Basic Plots
plt.plot(x, y)              # Line plot
plt.scatter(x, y)           # Scatter plot
plt.hist(data)              # Histogram
plt.bar(categories, values) # Bar plot

# Seaborn Plots
sns.scatterplot(x='col1', y='col2', data=df)
sns.heatmap(df.corr())
sns.boxplot(x='category', y='value', data=df)
sns.pairplot(df)
```

## 🤖 Scikit-learn Workflow

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

## 🧠 Deep Learning (TensorFlow/Keras)

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Simple Neural Network
model = Sequential([
    Dense(64, activation='relu', input_shape=(input_dim,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train Model
model.fit(X_train, y_train, 
          epochs=100, 
          batch_size=32,
          validation_split=0.2)
```

## 🔧 Common Functions

```python
# File Operations
with open('file.txt', 'r') as f:
    content = f.read()

# Error Handling
try:
    risky_operation()
except Exception as e:
    print(f"Error: {e}")

# Lambda Functions
square = lambda x: x**2
filter_evens = lambda lst: [x for x in lst if x % 2 == 0]

# Map, Filter, Reduce
mapped = list(map(lambda x: x*2, [1, 2, 3]))
filtered = list(filter(lambda x: x > 0, [-1, 0, 1, 2]))
```

## 📊 Data Preprocessing

```python
# Feature Scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encoding Categorical Variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le = LabelEncoder()
encoded = le.fit_transform(categorical_data)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```
