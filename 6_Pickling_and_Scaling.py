import numpy as np
import pandas as pd 
import quandl, math, datetime
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt 
from matplotlib import style 
style.use('ggplot')

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
predict_ahead_amount = int(math.ceil(0.01 * len(df)))
df['label'] = df[predict_column].shift(-predict_ahead_amount) 
X = np.array(df.drop(['label'], axis=1))
X = preprocessing.scale(X)
X = X[:-predict_ahead_amount]
X_unlabeled = X[-predict_ahead_amount:]
df.dropna(inplace=True)
y = np.array(df["label"])
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# https://docs.python.org/3/library/pickle.html
import pickle 

LOAD_CLASSIFIER = True # after running once, set this to True so you don't have to re-train everytime
if not LOAD_CLASSIFIER:
    print("Creating the Linear Regression classifier")
    # Create the classifier
    classifier = LinearRegression()

    # Train the classifier
    classifier.fit(X_train, y_train)

    # Save the classifier with pickle
    with open('models/linearregression.pickle', 'wb') as f:
        pickle.dump(classifier, f)
else:
    print("Loading the Linear Regression classifier")
    # Load the classifier with pickle
    pickle_in = open('models/linearregression.pickle', 'rb')
    classifier = pickle.load(pickle_in)

accuracy = classifier.score(X_test, y_test)
print("The classifier has an accuracy of {:.2f}%".format(accuracy*100.0))