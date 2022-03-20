import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
style.use('ggplot')

# Create a DataSet
dataset = [
    [1, 2],
    [1.5, 1.8],
    [5, 8],
    [8, 8],
    [1, 0.6],
    [9, 11],
]

# Define the feature set
X = np.array(dataset)

# Create the classifier 
classifier = KMeans(n_clusters=2)

# Train the classifier
classifier.fit(X)

# Plot the algorithm 
centroids = classifier.cluster_centers_
labels = classifier.labels_
colors = ["g.","r.","c.","y."]
for i in range(len(X)):
    # plot the dataset
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 25)
# plot the centroids
plt.scatter(centroids[:, 0],centroids[:, 1], marker = "x", s=150, linewidths = 5, zorder = 10)
plt.show()