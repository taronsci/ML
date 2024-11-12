from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load MNIST data
mnist = datasets.fetch_openml('mnist_784')
x, y = mnist['data'], mnist['target']

# Preprocess data
x = x / 255.0  # Normalize pixel values
x_training, x_test, y_training, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train SVM
model = svm.SVC(C=10, kernel='rbf', gamma=0.01, decision_function_shape='ovr')  # one-versus-all
model.fit(x_training, y_training)

# Test the model
y_prediction = model.predict(x_test)
accuracy = accuracy_score(y_test, y_prediction)
print(f'Accuracy of SVM: {accuracy:.4f}')

# Accuracy: 0.9351 with linear kernel and default C = 1 and gamma = 1 / (n_features * X.var()))

# Used source below to get nice values for parameters
# https://dmkothari.github.io/Machine-Learning-Projects/SVM_with_MNIST.html
# Accuracy: 0.9810 with rbf kernel C = 10 and gamma = 0.01
