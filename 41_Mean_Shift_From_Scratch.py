import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
style.use('ggplot')

# Create K_Means classifier
class Mean_Shift:
    def __init__(self, radius=4):
        self.radius = radius

    def fit(self, data):
        centroids = {}

        # Create a Centroid for each data point 
        for data_point in range(len(data)):
            centroids[data_point] = data[data_point]
        
        while True:
            new_centroids = []

            # For every centroid, calculate a new centroid position based on examples within its bandwidth 
            for centroid_key in centroids:
                examples_in_bandwidth = []
                centroid = centroids[centroid_key]
                # For every example (feature_set/row/data_point) see if it is wiithin the radius of the current centroid
                for example in data:
                    if np.linalg.norm(example - centroid) < self.radius:
                        examples_in_bandwidth.append(example)

                # Calculate a new centroid position as the average of the examples that are within the bandwidth
                new_centroid = np.average(examples_in_bandwidth, axis=0)
                new_centroids.append(tuple(new_centroid))

            # Get the unique centroid positions (convergence - some pervious centroids are updated to be the same position)
            unique_centroids = sorted(list(set(new_centroids)))

            prev_centroids = dict(centroids)

            # Add new centroids to the centroid dictionary 
            centroids = {} # clear out current centroids
            for centroid_id in range(len(unique_centroids)):
                centroids[centroid_id] = np.array(unique_centroids[centroid_id])

            # If no centroids positions are changed, then the algorithm is optimized
            optimized = True
            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False
                if not optimized:
                    break
            if optimized:
                break

        # Set the Class's centroids to the optimized centroids 
        self.centroids = centroids
    
    def predict(self, data):
        pass 

# Create DataSet 
X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8 ],
              [8, 8],
              [1, 0.6],
              [9,11],
              [8,2],
              [10,2],
              [9,3],])

# Create the classifier and train it 
classifier = Mean_Shift()
classifier.fit(X)

centroids = classifier.centroids 

colors = 10*["g","r","c","b","k"]

plt.scatter(X[:,0], X[:,1], s=150)

for centroid_id in centroids:
    plt.scatter(centroids[centroid_id][0], centroids[centroid_id][1], color='k', marker='*', s=150)

plt.show()
