# importing the libraries

import pandas as pd

# importing the dataset

dataset= pd.read_csv('../../../datasets/breast_cancer.csv')
x= dataset.iloc[:, 1:-1].values
y= dataset.iloc[:,-1].values

# splitting the dataset into the training and test set

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

# training the logistic regression model on the training set

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

# predicting the test results

y_predict = classifier.predict(x_test)

# making the confusion matrix

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_predict)
print(cm)

print(f"Accuracy: {accuracy_score(y_test, y_predict)*100:.2f}%")

# computing the accuracy with k-Fold Cross Validation

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)
print(f"Average of accuracy of 10 results: {accuracies.mean()*100:.2f}% ")
print(f"Standard Deviation: {accuracies.std()*100:.2f}% ")
