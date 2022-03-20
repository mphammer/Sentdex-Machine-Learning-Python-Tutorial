import numpy as np
from sklearn import preprocessing, model_selection, svm
import pandas as pd 

# Read in the data
df = pd.read_csv('data/breast-cancer-wisconsin.data')

# Look at the data
print(df.head())
print("")
print(df.info())
print("")

# Data Cleaning
df.replace('?', -99999, inplace=True) # most algorithms recognize that -99999 is an outlier and will treat it as such
df.drop(['id'], axis=1, inplace=True) # the 'id' (Sample code number) is not a good predictor so drop it

# Create Features DataSet and Labels Vector
X = np.array(df.drop(['class'], axis=1))
y = np.array(df['class'])

# Split the Data Sets into training and testing sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# Create the model and train it
classifier = svm.SVC()
classifier.fit(X_train, y_train)

# Evaluate the model
accuracy = classifier.score(X_test, y_test)
print("The accuracy of the SVM on classifying Cancer as Benign or Malignant is {:.2f}".format(accuracy))

