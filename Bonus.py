import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
# for task 2
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
# for task 3
from sklearn.datasets import fetch_openml
from kmodes.kmodes import KModes

# To run code go to line 199

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


def run_Logistic_Regression():
    """
    Creates a logistic regression model on the data, calculates accuracy, plots results
    :return:
    """
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


def run_SVM():
    """
    Creates an SVM model on the data, calculates accuracy, plots results
    :return:
    """
    # train SVM
    params = find_parameters_for_SVM()
    svm_model = svm.SVC(C=params.C, kernel='rbf', gamma=params.gamma, decision_function_shape='ovr')
    svm_model.fit(x_train, y_train)

    # Create a grid to plot the decision boundary
    xx = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)

    # Predict on the grid
    z = svm_model.predict(xx)

    # Calculate accuracy
    svm_prediction = svm_model.predict(x_test)
    accuracy = accuracy_score(y_test, svm_prediction)
    print(f"Accuracy: {accuracy}")

    # Plot the results
    plt.scatter(x, y, c=y_class, cmap='viridis', s=10)
    plt.plot(xx, z, color='red', lw=2)
    plt.title('SVM Max Margin Classifier')
    plt.show()


def run_kmeans():
    """
    Creates k-means model on the data, elbow plot, clusters with first 2 features
    :return:
    """
    # Load the Wine dataset
    wine_data = load_wine()
    x = wine_data.data

    # elbow plot
    wcss = {}
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=2)
        kmeans.fit(x)
        wcss[i] = kmeans.inertia_

    plt.plot(wcss.keys(), wcss.values(), 'gs-')
    plt.xlabel("Values of 'k'")
    plt.ylabel('WCSS')
    plt.show()

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=2)
    kmeans.fit(x)

    # Get cluster labels and centroids
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Plot clusters (using first two features)
    plt.scatter(x[:, 0], x[:, 1], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.75, marker='X')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('K-means Clustering on Wine Dataset')
    plt.show()


# Load dataset
adult_data = fetch_openml(name='adult', version=2)
x1 = adult_data.data
x1 = x1.dropna()  # Drop rows with missing values


def determine_k(samples):
    """
    Creates elbow plot for k-modes using a sample(to make it faster) of the data
    :param samples: number of samples used from the dataset
    :return: the subset chosen from the dataset of length "samples"
    """
    subset = x1.sample(n=samples, random_state=2)

    costs = []
    for k in range(1, 15):  # Test k from 1 to 10 clusters
        kmode = KModes(n_clusters=k, init='Huang', n_init=10, verbose=1)
        kmode.fit(subset)
        costs.append(kmode.cost_)

    plt.plot(range(1, 15), costs, marker='o')
    plt.title('Elbow Plot for K-Modes Clustering')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Cost (Inertia)')
    plt.show()

    return subset


def run_kmodes(samples):
    """
    Creates k-modes model on a sample of the data
    :param samples: number of samples used from the dataset
    :return:
    """
    s = determine_k(samples)
    # Apply K-Modes clustering
    km = KModes(n_clusters=10, init='Huang', n_init=10, verbose=0)
    y_km = km.fit_predict(s)

    # Plot the results
    plt.scatter(s['age'], s['hours-per-week'], c=y_km, cmap='viridis', s=10)
    plt.title("K-Modes Clustering")
    plt.xlabel("Age")
    plt.ylabel("Hours per week")
    plt.show()


# Problem 1a
run_Logistic_Regression()
# Problem 1b
run_SVM()
# Problem 2
run_kmeans()
# Problem 3 - running determine_k(1000) gives the elbow plot, where 10 seems to be a good k
run_kmodes(1000)
