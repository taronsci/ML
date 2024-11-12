import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import numpy as np
from math import log2
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import wittgenstein as lw  # for RIPPER

#                          Results

# RIPPER and Naive Bayes both have 75% accuracy and misclassify 1 test sample,
# while training decision tree with gini-index achieves 100% accuracy on the test data

# Q7
# the data
num_samples = 14
data = {
    'ID': range(1, num_samples + 1),  # IDs
    'age': pd.Categorical(
        ['youth', 'youth', 'middle_aged', 'senior', 'senior', 'senior', 'middle_aged', 'youth', 'youth', 'senior',
         'youth', 'middle_aged', 'middle_aged', 'senior'],
        categories=['youth', 'middle_aged', 'senior'],
        ordered=True),
    'income': pd.Categorical(
        ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium',
         'medium', 'medium', 'high', 'medium'],
        categories=['low', 'medium', 'high'],
        ordered=True),
    'student': pd.Categorical(
        ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no'],
        categories=['yes', 'no'],
        ordered=False),
    'credit_rating': pd.Categorical(
        ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair',
         'fair', 'excellent', 'excellent', 'fair', 'excellent'],
        categories=['fair', 'excellent'],
        ordered=True),
    'Class: buys_computer': pd.Categorical(
        ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no'],
        categories=['yes', 'no'],
        ordered=False)
}

# Create DataFrame
df = pd.DataFrame(data)

#                                               preprocessing steps

# label encoding ordinal features
ordinal_columns = ['age', 'income', 'credit_rating']  # to use RIPPER add 'Class: buys_computer' to the list
for col in ordinal_columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# one hot encode non-ordinal features
df['student'] = df['student'].astype(str)
df = pd.get_dummies(df, columns=['student'])

# Split the data into training and test sets
x_training = df.iloc[:10].drop(columns=['Class: buys_computer', 'ID'])
y_training = df['Class: buys_computer'].iloc[:10]

x_test = df.iloc[10:14].drop(columns=['Class: buys_computer', 'ID'])
y_test = df['Class: buys_computer'].iloc[10:14]


# a)
# create decision tree
tree = DecisionTreeClassifier(criterion='gini')  # or criterion='entropy' for information gain
tree.fit(x_training, y_training)

# test accuracy on test data
decision_tree_predictions = tree.predict(x_test)
print(f"Accuracy for decision tree: {accuracy_score(y_test, decision_tree_predictions)}")

# # plot decision tree
# plt.figure(figsize=(12, 8))
# plot_tree(tree, feature_names=x_training.columns, class_names=tree.classes_, filled=True, fontsize=10)
# plt.show()


def entropy(data_column):
    """
    calculates entropy
    :param data_column:
    :return:
    """
    class_counts = data_column.value_counts()
    total = len(data_column)
    entropy_value = 0
    for count in class_counts:
        probability = count / total
        if probability > 0:  # to avoid log(0)
            entropy_value -= probability * log2(probability)
    return entropy_value


def information_gain(df, feature, class_label):
    """
    calculates information gain
    :param df: the dataframe
    :param feature:
    :param class_label:
    :return: information gain of the feature wrt the class_label
    """
    dataset_entropy = entropy(df[class_label])

    # Calculate the weighted sum
    feature_values = df[feature].unique()
    weighted_entropy = 0
    for value in feature_values:
        subset = df[df[feature] == value]
        weighted_entropy += (len(subset) / len(df)) * entropy(subset[class_label])

    return dataset_entropy - weighted_entropy


features = ['age', 'income', 'student_yes', 'credit_rating']
class_label = 'Class: buys_computer'

subset = df.iloc[:10]

# Calculates the information gain for the features using 10 samples
for feature in features:
    ig = information_gain(subset, feature, class_label)
    print(f"Information Gain for {feature}: {ig}")

# # calculates gini-index for single feature
# class_column = 'Class: buys_computer'
# gini_values = {}
# for age_group in df['age'].unique():
#     subset = df[df['age'] == age_group]
#     class_counts = subset[class_column].value_counts(normalize=True)
#     gini = 1 - (class_counts**2).sum()
#     gini_values[age_group] = gini
# print(gini_values)


#               b)

# Initialize and train the Naive Bayes classifier
nb_classifier = CategoricalNB()
nb_classifier.fit(x_training, y_training)

# test accuracy on test data
naive_bayes_predictions = nb_classifier.predict(x_test)
print(f'Accuracy for Naive Bayes: {accuracy_score(y_test, naive_bayes_predictions)}')

#               c)
# before fitting the data, update the preprocessing by encoding the class label

# model = lw.RIPPER()
# model.fit(x_training, y_training)
# predictions = model.predict(x_test)
#
# accuracy = accuracy_score(y_test, predictions)
# print(f"Accuracy of RIPPER: {accuracy}")

