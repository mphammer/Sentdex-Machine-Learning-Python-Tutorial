'''
Euclidean Distance: 
Sqrt( SUM i=1..n of (q_i - p_i)^2 )

'''

import numpy as np 
from math import sqrt 
import matplotlib.pyplot as plt 
from matplotlib import style 
from collections import Counter 
import warnings 
style.use('fivethirtyeight')

# Write functions for Euclidean distance 

def euclidean_distance_python(y_actual, y_predicted):
    sum = 0
    for i in range(len(y_actual)):
        sum += (y_actual[i] - y_predicted[i])**2
    return sqrt(sum)

def euclidean_distance_numpy(y_actual, y_predicted):
    y_actual = np.array(y_actual)
    y_predicted = np.array(y_predicted)
    squared_errors = (y_actual - y_predicted)**2
    sum = np.sum(squared_errors)
    return np.sqrt(sum)

def euclidean_distance_linalg(y_actual, y_predicted):
    return np.linalg.norm(np.array(y_actual)-np.array(y_predicted))

e_dist_py = euclidean_distance_python([1,2,5], [1,3,9])
e_dist_np = euclidean_distance_numpy([1,2,5], [1,3,9])
e_dist_la = euclidean_distance_numpy([1,2,5], [1,3,9])
print(e_dist_py, e_dist_np, e_dist_la)

# Start K Nearest Neighbors

# Create a dataset 
dataset = {
    'blue': [[1,2],[2,3],[3,1]],
    'red': [[6,5],[7,7],[8,6]]
}

# Plot the dataset 
for cluster_color in dataset:
    cluster_points = dataset[cluster_color]
    for point in cluster_points:
        plt.scatter(point[0], point[1], s=100, color=cluster_color)

# Plot the data point we are trying to classify
new_feature = [5,7]
plt.scatter(new_feature[0], new_feature[1], s=50, color="green")

plt.show()

def k_nearest_neighbors(dataset, predict_data_point, k=3):
    if len(dataset) >= k:
        warnings.warn('K is set to a value less than total voting groups')
    if k%2 == 0:
        warnings.warn('K should be set to an odd value to resolve "draw votes"')
    prediction_distance_and_group = []
    for group in dataset:
        for feature_data_point in dataset[group]:
            euclidean_distance = euclidean_distance_linalg(feature_data_point, predict_data_point)
            # Store the distances to another point and the group
            prediction_distance_and_group.append([euclidean_distance, group])
    
    # sort the distances, pick the 'k' closest distances, and put the groups into an array
    top_k_votes_for_group = [i[1] for i in sorted(prediction_distance_and_group)[:k]]

    # select the most common group from the list of votes
    vote_result = Counter(top_k_votes_for_group).most_common(1)[0][0]

    return vote_result

result = k_nearest_neighbors(dataset, new_feature)
print("The data new datapoint {} is classified as '{}'".format(new_feature, result))