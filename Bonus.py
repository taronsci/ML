import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Generate random data
x = np.linspace(10, 15, 100)
y_true = 3 * np.sin(x) + np.log(x-4) - np.cos(3 * x)

# Add noise
y = y_true + np.random.normal(0, 1, len(x))

# binary classification based on threshold function
threshold = 3 * np.cos(2 * x)
y_class = (y > threshold).astype(int)

# plt.scatter(x, y, c=y_class, cmap='viridis')

x = x.reshape(-1, 1)

# Create the logistic regression model
model = LogisticRegression()
model.fit(x, y_class)

# Make predictions
y_probs = model.predict_proba(x)[:, 1]

plt.scatter(x, y, c=y_class, cmap='viridis')
plt.plot(x, y_probs, color='red', label='Logistic Regression Line')
plt.legend()
plt.show()