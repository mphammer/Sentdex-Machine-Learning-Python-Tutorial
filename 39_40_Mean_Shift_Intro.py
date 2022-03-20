import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import MeanShift
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
original_df = pd.DataFrame.copy(df)

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
classifier = MeanShift()
classifier.fit(X)

# Get the labels (cluster names) and cluster centers that were created
labels = classifier.labels_
cluster_centers = classifier.cluster_centers_

# Populate a column with the cluster_group it was assigned to 
original_df['cluster_group'] = np.nan # Create empty column
for row_index in range(len(X)):
    original_df['cluster_group'].iloc[row_index] = labels[row_index]

n_clusters_ = len(np.unique(labels))
survival_rates = {}
for cluster_label in range(n_clusters_):
    # get dataframe with only the rows for this cluster label
    temp_df = original_df[ (original_df['cluster_group']==float(cluster_label)) ]
    
    # get a dataframe with only the rows that survived 
    survival_cluster = temp_df[  (temp_df['survived'] == 1) ]

    # Calculate the number that survived
    survival_rate = len(survival_cluster) / len(temp_df)
    
    survival_rates[cluster_label] = survival_rate
    
for cluster_label, survival_rate in survival_rates.items():
    print("Cluster {} has a survival rate of {}".format(cluster_label, survival_rate))
print()

for cluster_label, survival_rate in survival_rates.items():
    print("Describe Group {}".format(cluster_label))
    print(original_df[ (original_df['cluster_group']==cluster_label) ].describe())
    print()

