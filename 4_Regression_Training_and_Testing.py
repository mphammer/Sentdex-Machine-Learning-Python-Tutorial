
import pandas as pd 
import quandl
import math

import os
QAUNDL_API_KEY = os.getenv('NASDAQ_API_KEY')

# Previous Lesson
df = quandl.get("WIKI/GOOGL", api_key=QAUNDL_API_KEY)
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
df['High_Low_Percent'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0 # Volatility 
df['Percent_Change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0 
df = df[["Adj. Close", "High_Low_Percent", "Percent_Change", "Adj. Volume"]]
predict_column = 'Adj. Close'
df.fillna(-99999, inplace=True)
predict_ahead_amount = int(math.ceil(0.05 * len(df)))
df['label'] = df[predict_column].shift(-predict_ahead_amount) 
df.dropna(inplace=True)
print(df.head())

import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression 

# Create our table of features and an array for our the labels
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html
# The 1 means that we want to drop from the columns. 0 (default) would mean from the index (rows)
X = np.array(df.drop(['label'], axis=1))
y = np.array(df["label"])

# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.scale.html
# Standardize a dataset along any axis. By default it centers the mean around 0
X = preprocessing.scale(X)

# Create Training Sets and Testing Sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# Train the LinearRegression Model
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
classifier = LinearRegression()
classifier.fit(X_train, y_train) # train 

# See how well the model predicts new data
accuracy = classifier.score(X_test, y_test) # test the classifier 

print("The Linear Regression Model predicted stock Prices with an accuracy of {}".format(accuracy))

# Train and Test the SVM model
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
classifier = svm.SVR()
classifier.fit(X_train, y_train) # train 
accuracy = classifier.score(X_test, y_test) # test

print("The SVM with Support Vector Regression predicted stock Prices with an accuracy of {}".format(accuracy))