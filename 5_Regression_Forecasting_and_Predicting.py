
import numpy as np
import pandas as pd 
import quandl, math
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression 

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

import datetime
import matplotlib.pyplot as plt 
from matplotlib import style 

style.use('ggplot')

# Create X feature dataset to have all rows that have labels (since we shifted up the last 'predict_ahead_amount' rows dont have labels )
X = np.array(df.drop(['label'], axis=1))
X = preprocessing.scale(X)
# This has the last rows that don't have labels
X_unlabeled = X[-predict_ahead_amount:] # get rows (end - predict_ahead_amount) through the end
X = X[:-predict_ahead_amount] # get rows 0 through (end - predict_ahead_amount)

# Create the labels set
df.dropna(inplace=True)
y = np.array(df["label"])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
classifier = LinearRegression()
classifier.fit(X_train, y_train) # train 
accuracy = classifier.score(X_test, y_test)

# Make a prediction on the rows that don't have lables yet
predictions_set = classifier.predict(X_unlabeled)

print("This model has an accuracy of {:.2f}%".format(accuracy*100.0))
print("Here are the predictions for {} day(s)".format(len(X_unlabeled)))
print(predictions_set)

# Create a Column at the end of the DataFrame filled with "NaN" values (Not a Number)
df['Predictions'] = np.nan 
print(df.tail())

# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html
# https://pandas.pydata.org/docs/reference/api/pandas.Series.name.html
last_date = df.iloc[-1].name # get the name of the last row (in this case the name of a row is the date)
last_unix_date = last_date.timestamp()
one_day = 86400 # seconds in a day
next_unix_date = last_unix_date + one_day

# add rows to the dateframe that have NaN for the features and the prediction value for the label (the label is the last column)
for prediction_value in predictions_set: 
    next_date = datetime.datetime.fromtimestamp(next_unix_date)
    next_unix_date += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [prediction_value]

print(df.head())
print(df.tail(len(predictions_set)+5))

# https://pandas.pydata.org/docs/reference/api/pandas.Series.plot.html
# pandas.Series.plot - the x value is the label for the row (the date) or the row's index. the y value is the actual value in the row
df['Adj. Close'].plot()
df['Predictions'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()