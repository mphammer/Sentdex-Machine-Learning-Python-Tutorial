import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
style.use('ggplot')

# Create K_Means classifier
class K_Means:
    def __init__(self, k=2, tolerance=0.001, max_iterations=300):
        self.k = k
        self.tolerance = tolerance # how much the centroid will move
        self.max_iterations = max_iterations # how many times we recalculate the centroid position 

    def fit(self, data):

        # Initialize the Centroids to random values (we use the first two data points)
        self.centroids = {}
        for i in range(self.k):
            self.centroids[i] = data[i]

        # Optimize the centroids 
        for i in range(self.max_iterations):

            # Initialize the Classifications Dictionary 
            self.centroid_classifications = {} # map of class to values in the set
            for i in range(self.k):
                self.centroid_classifications[i] = []

            for example in data:
                # Create a list of distances from the example/feature_vector/data_point to each centroid
                distance_to_centroids = [np.linalg.norm(example - self.centroids[centroid]) for centroid in self.centroids]
                
                # The classification should be the centroid that's the closest
                shorter_distance = min(distance_to_centroids)
                closer_centroid_id = distance_to_centroids.index(shorter_distance)
                self.centroid_classifications[closer_centroid_id].append(example) # classify the example to the closer centroid

            # Update the centroid positions
            prev_centroids = dict(self.centroids)
            for centroid_id in self.centroid_classifications:
                # Find the mean of all the examples that were assigned to the centroid
                mean_location_of_examples = np.average(self.centroid_classifications[centroid_id], axis=0)
                # Update the centroid to now be the mean location
                self.centroids[centroid_id] = mean_location_of_examples
            
            # If the update to any of the centroid positions is greater than the tolerance, then keep optimizing 
            optimized = True 
            for centroid in self.centroids:
                original_centroid = prev_centroids[centroid]
                current_centroid = self.centroids[centroid]
                if np.sum((current_centroid - original_centroid) / original_centroid*100.0) > self.tolerance:
                    optimized = False
            
            if optimized:
                break 


    def predict(self, data):
        distance_to_centroids = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        shorter_distance = min(distance_to_centroids)
        closer_centroid_id = distance_to_centroids.index(shorter_distance)
        return closer_centroid_id

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

# Create the algorithm and train it
classifier = K_Means()
classifier.fit(X)

colors = 10*["g","r","c","b","k"]

# Plot the centroids
for centroid in classifier.centroids:
    plt.scatter(classifier.centroids[centroid][0], classifier.centroids[centroid][1],
                marker="o", color="k", s=150, linewidths=5)

# Plot all of the datapoints (examples)
for classification in classifier.centroid_classifications:
    color = colors[classification]
    for example in classifier.centroid_classifications[classification]:
        plt.scatter(example[0], example[1], marker="x", color=color, s=150, linewidths=5)

# Plot unknown predictions  
unknowns = np.array([[1,3],
                    [8,9],
                    [0,3],
                    [5,4],
                    [6,4],])
for unknown in unknowns:
   classification = classifier.predict(unknown)
   plt.scatter(unknown[0], unknown[1], marker="*", color=colors[classification], s=150, linewidths=5)

plt.savefig('plots/k_means.png')
plt.show()
