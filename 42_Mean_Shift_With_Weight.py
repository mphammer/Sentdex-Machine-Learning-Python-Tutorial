import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets._samples_generator import make_blobs
import random 
style.use('ggplot')

# Create K_Means classifier
class Mean_Shift:
    def __init__(self, radius=None, radius_norm_step=100):
        self.radius = radius
        self.radius_norm_step = radius_norm_step

    def fit(self, data):
        # If the radius is None, set it to be a magnitude of all the data
        if self.radius == None:
            centroid_of_all_data = np.average(data, axis=0)
            norm_of_all_data = np.linalg.norm(centroid_of_all_data)
            self.radius = norm_of_all_data / self.radius_norm_step

        centroids = {}

        # Create a Centroid for each data point 
        for data_point in range(len(data)):
            centroids[data_point] = data[data_point]
        
        weights = [i for i in range(self.radius_norm_step)][::-1] # a descending list of numbers (so the starting weight is very high)
        
        while True:
            new_centroids = []

            # For every centroid, calculate a new centroid position based on examples within its bandwidth 
            for centroid_key in centroids:
                examples_in_bandwidth = []
                centroid = centroids[centroid_key]

                # For every example (feature_set/row/data_point) see if it is wiithin the radius of the current centroid
                for example in data:
                    # Get the distance from the example to the centroid 
                    distance = np.linalg.norm(example - centroid)
                    if distance == 0:
                        distance = 0.00000000001
                    
                    # Get a weight such that points that are farther away have a lower weight (points closer to the centroid have a higher weight)
                    weight_index = int(distance/self.radius)
                    if weight_index > self.radius_norm_step - 1: # check fi weight is too far away
                        weight_index = self.radius_norm_step - 1

                    # Add a number of examples depending on how much we should weight it - later when we take an average, these will be treated higher 
                    num_examples_to_add = weights[weight_index]**2
                    examples_to_add = num_examples_to_add * [example]
                    examples_in_bandwidth.extend(examples_to_add)

                # Calculate a new centroid position as the average of the examples that are within the bandwidth
                new_centroid = np.average(examples_in_bandwidth, axis=0)
                new_centroids.append(tuple(new_centroid))

            # Get the unique centroid positions (convergence - some pervious centroids are updated to be the same position)
            unique_centroids = sorted(list(set(new_centroids)))

            # Remove centroids that are very very close to each other
            centroids_to_remove = []
            for unique_centroid_ptr1 in unique_centroids:
                for unique_centroid_ptr2 in [i for i in unique_centroids]:
                    if unique_centroid_ptr1 == unique_centroid_ptr2:
                        # continue if they are the same centroid
                        continue
                    distance_between_centroids = np.linalg.norm(np.array(unique_centroid_ptr1)-np.array(unique_centroid_ptr2))
                    if distance_between_centroids <= self.radius:
                        centroids_to_remove.append(unique_centroid_ptr2)
                        break
            for redundant_centroid in centroids_to_remove:
                try:
                    unique_centroids.remove(redundant_centroid)
                except:
                    pass


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

        # Initialize dictionary for classifications
        self.classifications = {}
        for class_id in range(len(self.centroids)):
            self.classifications[class_id] = []
            
        for example in data:
            # get distance from the example to each centroid
            distances = [np.linalg.norm(example - self.centroids[centroid]) for centroid in self.centroids]
            class_id = (distances.index(min(distances)))

            # example that belongs to that cluster (group/class)
            self.classifications[class_id].append(example)
    
    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = (distances.index(min(distances)))
        return classification

# Create DataSet 
centers = random.randrange(2, 6)
X, y = make_blobs(n_samples=15, centers=centers, n_features=2)

# Create the classifier and train it 
classifier = Mean_Shift()
classifier.fit(X)

centroids = classifier.centroids 

colors = 10*["g","r","c","b","k"]

# plot the data points for each cluster (class)
for classification in classifier.classifications:
    color = colors[classification]
    for featureset in classifier.classifications[classification]:
        plt.scatter(featureset[0],featureset[1], marker = "x", color=color, s=150, linewidths = 5, zorder = 10)

# plot the centroids
for c in centroids:
    plt.scatter(centroids[c][0],centroids[c][1], color='k', marker = "*", s=150, linewidths = 5)


plt.savefig('plots/mean_shift.png')
plt.show()
