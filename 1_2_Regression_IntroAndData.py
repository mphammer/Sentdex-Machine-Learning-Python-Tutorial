import pandas as pd 
import quandl 

import os
QAUNDL_API_KEY = os.getenv('NASDAQ_API_KEY')  # Create the key here: https://data.nasdaq.com/

# get google stock dataset
df = quandl.get("WIKI/GOOGL", api_key=QAUNDL_API_KEY)

# print the first 5 rows of the dataset
print(df.head())

# print the shape (rows, columns) of the dataset
print(df.shape)

# print the size of the dataset
print(df.size)

# print information about the dataset
print(df.info())

# Select only the meaningful features
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]

# Create new features that are more useful
df['High_Low_Percent'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0 # Volatility 
df['Percent_Change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0 

# Create a new DataFrame with only the columns that we care about
df = df[["Adj. Close", "High_Low_Percent", "Percent_Change", "Adj. Volume"]]
print(df.head())