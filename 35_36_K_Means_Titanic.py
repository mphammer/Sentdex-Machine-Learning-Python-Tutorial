import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd

'''
Information about the Data Set: #https://pythonprogramming.net/static/downloads/machine-learning-data/titanic.xls
-------------------------------------------------
Pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
survival Survival (0 = No; 1 = Yes)
name Name
sex Sex
age Age
sibsp Number of Siblings/Spouses Aboard
parch Number of Parents/Children Aboard
ticket Ticket Number
fare Passenger Fare (British pound)
cabin Cabin
embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
boat Lifeboat
body Body Identification Number
home.dest Home/Destination
'''

# Read in the DataSet
df = pd.read_excel('data/titanic.xls')

# Explore the DataSet
print(df.head())
print()
print(df.info())
print()

# Data Cleaning
df.drop(['body', 'name'], axis=1, inplace=True)
df.fillna(0, inplace=True)

def convert_non_numerical_data(df):
    for column_name in df.columns.values: 

        map_value_to_digit = {}
        def convert_value_to_digit(val):
            return map_value_to_digit[val]

        # Check if the column type is not an integer
        if df[column_name].dtype != np.int64 and df[column_name].dtype != np.float64:
            
            # Get the unique set of column values and assign each one a digit
            column_contents = df[column_name].values.tolist()
            unique_elements = set(column_contents)
            digit = 0
            for unique in unique_elements:
                if unique not in map_value_to_digit:
                    map_value_to_digit[unique] = digit
                    digit += 1 
            
            # Convert all values in the column to their digits
            df[column_name] = list(map(convert_value_to_digit, df[column_name]))
    return df 

df = convert_non_numerical_data(df)

# Create a feature set and prediction set
X = np.array(df.drop(['survived'], axis=1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

# Create the classifier and train it
classifier = KMeans(n_clusters=2)
classifier.fit(X)

# Analyze the performance 
correct = 0
for example_index in range(len(X)):
    example = X[example_index].astype(float)

    predict_feature_vector = np.array(example)
    predict_feature_vector = predict_feature_vector.reshape(-1, len(predict_feature_vector))
    
    prediction = classifier.predict(predict_feature_vector)

    # Note: KMeans doesn't know that survived==1. It could assign the survived the value 0
    # Therefore you may see the accuracy flip flop
    actual_value = y[example_index]
    if prediction[0] == actual_value:
        correct += 1

accuracy = correct/len(X)

print("The accuracy of survival prediction is {:.2f}".format(accuracy))
print(" - Note: KMeans doesn't know that survived==1. It could assign the survived the value 0. Therefore you may see the accuracy flip flop.")