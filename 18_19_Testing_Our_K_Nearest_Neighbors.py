import numpy as np 
import pandas as pd
from sklearn import preprocessing, model_selection, neighbors
from math import sqrt 
from collections import Counter 
import warnings 
import random

# From Previous Lesson
def euclidean_distance_linalg(y_actual, y_predicted):
    return np.linalg.norm(np.array(y_actual)-np.array(y_predicted))
def k_nearest_neighbors(dataset, predict_data_point, k=3):
    if len(dataset) >= k:
        warnings.warn('K is set to a value less than total voting groups')
    if k%2 == 0:
        warnings.warn('K should be set to an odd value to resolve "draw votes"')
    prediction_distance_and_group = []
    for group in dataset:
        for feature_data_point in dataset[group]:
            euclidean_distance = euclidean_distance_linalg(feature_data_point, predict_data_point)
            prediction_distance_and_group.append([euclidean_distance, group])
    top_k_votes_for_group = [i[1] for i in sorted(prediction_distance_and_group)[:k]]
    vote_result = Counter(top_k_votes_for_group).most_common(1)[0][0]

    # Calculate confidence 
    confidence = Counter(top_k_votes_for_group).most_common(1)[0][1] / k

    return vote_result, confidence

# Read in the Data
df = pd.read_csv('data/breast-cancer-wisconsin.data')

# Clean the data
df.replace('?', -99999, inplace=True) # most algorithms recognize that -99999 is an outlier and will treat it as such
df.drop(['id'], axis=1, inplace=True) # the 'id' (Sample code number) is not a good predictor so drop it
df = df.astype(float) # make sure everything is an int or float

# # Display information about the data
print(df.head())
print()
print(df.info())
print()

num_trials = 10

trial_results = []
for trial_num in range(num_trials):

    # Make the Data Usable by Our Function

    # convert dataset to list of lists and shuffle the rows
    df_list = df.values.tolist() 
    random.shuffle(df_list)

    # Split Training Data and Testing Data (where testing data is what we will be trying to classify)
    test_size = 0.2 
    train_data = df_list[:-int(test_size*len(df_list))] # store first 80% as train_data
    test_data = df_list[-int(test_size*len(df_list)):] # store last 20% as test_data

    # Put the Training Set into a format that our algorithm can handle
    train_set = {2:[], 4:[]}
    for sample in train_data:
        label = sample[-1] # get the value in the last column
        features = sample[:-1] # remove the last column since it has the label
        train_set[label].append(features) 

    # Put the Testing Data (data to classify) into a format that our algorithm can handle
    test_set = {2:[], 4:[]}
    for sample in test_data:
        label = sample[-1] # get the value in the last column
        features = sample[:-1] # remove the last column since it has the label
        test_set[label].append(features) 

    # Try to classify each value from the Test Set
    k = 5 # 5 is what the sklearn algorithm uses by default 
    num_correct_classifications = 0
    total_classification_attempts = 0
    for group in train_set:
        for data in train_set[group]:
            vote, confidence = k_nearest_neighbors(test_set, data, k=k) 
            if group == vote:
                num_correct_classifications +=1
            total_classification_attempts += 1

    accuracy = num_correct_classifications / total_classification_attempts
    # print("For k = {}, accuracy = {:.2f}".format(k, accuracy))

    trial_results.append(accuracy)

average_accuracy = sum(trial_results) / len(trial_results)
print("The average accuracy of our model is: {:.2f}".format(average_accuracy))

sklearn_trial_results = []
for trial_num in range(num_trials):
    X = np.array(df.drop(['class'], axis=1))
    y = np.array(df['class'])
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
    classifier = neighbors.KNeighborsClassifier()
    classifier.fit(X_train, y_train)
    accuracy = classifier.score(X_test, y_test)
    sklearn_trial_results.append(accuracy)

average_accuracy = sum(sklearn_trial_results) / len(sklearn_trial_results)
print("The average accuracy of the sklearn model is: {:.2f}".format(average_accuracy))