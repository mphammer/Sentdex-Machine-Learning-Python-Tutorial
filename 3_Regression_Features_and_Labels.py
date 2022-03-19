
import pandas as pd 
import quandl

import os
QAUNDL_API_KEY = os.getenv('NASDAQ_API_KEY')

# Previous Lesson
df = quandl.get("WIKI/GOOGL", api_key=QAUNDL_API_KEY)
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
df['High_Low_Percent'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0 # Volatility 
df['Percent_Change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0 
df = df[["Adj. Close", "High_Low_Percent", "Percent_Change", "Adj. Volume"]]
print(df.head())

import math

predict_column = 'Adj. Close'
df.fillna(-99999, inplace=True)

predict_ahead_amount = int(math.ceil(0.05 * len(df)))

# shift() moves the rows up or down in the table. 
# Since the rows are sorted by date that means the label will become the "Closing" stock value for X days n the future
# Where X == predict_ahead_amount
df['label'] = df[predict_column].shift(-predict_ahead_amount) 

# since we shifted up, now drop any rows that dont have labels in them
df.dropna(inplace=True)

print(df.head())

