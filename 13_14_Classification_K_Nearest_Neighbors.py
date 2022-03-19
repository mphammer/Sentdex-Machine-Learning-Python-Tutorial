'''
Download Breast Cancer Data: 
- breast-cancer-wisconsin.data 
Edit the above file and add a header row on the first line: 
- id,clump_thickness,unif_call_size,unif_cell_shape,marg_adhesion,single_epith_cell_size,bare_nuclei,bland_chrom,norm_nuclei,mitoses,class

Read this to understand what is in the data: 
- breast-cancer-wisconsin.names

For the class feature, 2 mean bening and 4 means malignant.
Missing attributes have a "?"
'''

from matplotlib.pyplot import cla
import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd 

# Read in the data
df = pd.read_csv('data/breast-cancer-wisconsin.data')

# Look at the data
print(df.head())
print("")
print(df.info())
print("")

# Clean the data
df.replace('?', -99999, inplace=True) # most algorithms recognize that -99999 is an outlier and will treat it as such

# Create Feature Set and Labels Set
X = np.array(df.drop(['class'], axis=1))
y = np.array(df['class'])

# Create training and test data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# Create themodel and train it
classifier = neighbors.KNeighborsClassifier()
classifier.fit(X_train, y_train)

# Evaluate the model
accuracy = classifier.score(X_test, y_test)
print("The accuracy of the K Nearest Neighbors on classifying Cancer as Benign or Malignant is {:.2f}".format(accuracy))

######################################################################
# Now, notice how dropping a bad column improves the accuracy by a lot

# Data Cleaning
df.drop(['id'], axis=1, inplace=True) # the 'id' (Sample code number) is not a good predictor so drop it

# Create Feature Set and Labels Set
X = np.array(df.drop(['class'], axis=1))
y = np.array(df['class'])

# Create training and test data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# Create themodel and train it
classifier = neighbors.KNeighborsClassifier()
classifier.fit(X_train, y_train)

# Evaluate the model
accuracy = classifier.score(X_test, y_test)
print("The accuracy of the K Nearest Neighbors on classifying Cancer as Benign or Malignant is {:.2f}".format(accuracy))

########################################
# Lets make some predictions with the model

new_example = np.array([[4,2,1,1,1,2,3,2,1]]) # we just made this up 
prediction = classifier.predict(new_example)
print("The example with feature values {} is predicted to be {}".format(new_example, prediction))
print(" - 2 means benign and 4 means malignant")

new_examples = np.array([[4,2,1,1,1,2,3,2,1], [4,2,1,2,2,2,3,2,1]])
predictions = classifier.predict(new_examples)
print("The examples with feature values...\n {}\n... are predicted to be {}".format(new_examples, predictions))