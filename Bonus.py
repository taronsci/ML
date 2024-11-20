import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# Generate random data
x = np.linspace(10, 15, 200)
y_true = 3 * np.sin(x) + np.log(x - 4) - np.cos(3 * x)

# Add noise
y = y_true + np.random.normal(0, 1, len(x))

# binary classification based on threshold function
threshold = 3 * np.cos(2 * x)
y_class = (y > threshold).astype(int)

x = x.reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y_class, test_size=0.1, random_state=34, shuffle=True)

# 1a
# Create the logistic regression model
model = LogisticRegression()
model.fit(x_train, y_train)

# Make predictions
y_probs = model.predict_proba(x_test)[:, 1]
predictions = (y_probs >= 0.5).astype(int)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

# plot Logistic Regression Line
plt.scatter(x, y, c=y_class, cmap='viridis')
plt.plot(x_test, y_probs, color='red', label='Logistic Regression Line')
plt.legend()
plt.show()


# 1b
def find_parameters_for_SVM():
    """
    Function that finds parameters for SVM using GridSearchCV
    :return: best_model
    """
    # parameter grid
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100, 1000],
        'gamma': [10, 1, 0.1, 0.01, 0.001],
        'kernel': ['rbf']
    }

    # Train SVM
    svm_test_model = svm.SVC()
    grid_search = GridSearchCV(estimator=svm_test_model, param_grid=param_grid, cv=10, scoring='accuracy', verbose=0)
    grid_search.fit(x_train, y_train)

    # Print the best parameters and score
    print("Best Parameters:", grid_search.best_params_)
    print("Best Cross-Validation Score:", grid_search.best_score_)

    # Test the best model on the test data
    best_model = grid_search.best_estimator_

    print("Test Accuracy:", best_model.score(x_test, y_test), "\n")

    return best_model


# train SVM
params = find_parameters_for_SVM()
svm_model = svm.SVC(C=params.C, kernel='rbf', gamma=params.gamma, decision_function_shape='ovr')
svm_model.fit(x_train, y_train)

# Create a grid to plot the decision boundary
xx = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)

# Predict on the grid
Z = svm_model.predict(xx)

# Calculate accuracy
svm_prediction = svm_model.predict(x_test)
accuracy = accuracy_score(y_test, svm_prediction)
print(f"Accuracy: {accuracy}")

# Plot the results
plt.scatter(x, y, c=y_class, cmap='viridis', s=10)
plt.plot(xx, Z, color='red', lw=2)
plt.title('SVM Max Margin Classifier')
plt.show()
