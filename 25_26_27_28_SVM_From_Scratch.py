import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

# Create the Data Set
data_dict = {
    -1: np.array([[1,7], [2,8], [3,8],]),
    1: np.array([[5,1], [6,-1], [7,3],])
}

class Support_Vector_Machine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1:'r', -1:'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.axis = self.fig.add_subplot(1,1,1)
    
    # train
    def fit(self, data):
        self.data = data
        # { ||w||: [w, b] }
        opt_dict = {}

        transforms = [[1,1], [-1,1], [-1,-1], [1,-1]]

        # Find the max and min feature values
        all_feature_values = []
        for yi in self.data:
            for featureset in self.data[yi]:
                all_feature_values.extend(featureset)
        self.max_feature_value = max(all_feature_values)
        self.min_feature_value = min(all_feature_values)
        all_feature_values = None # clear out this variable so you're not holding it in memory 

        # Initialize the step sizes
        # We first start by taking large steps sizes. By moving around a lot in the beginning we help
        # ensure that we wont get stuck in any local-optimums. 
        # We continually make the step size smaller so that we can take smaller steps which let us step
        # closer and closer to the optimum (hopefully the global optimum)
        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      self.max_feature_value * 0.001,
                    ]
        
        b_range_multiple = 5
        b_multiple = 5

        latest_optimum = self.max_feature_value * 10

        for step_size in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            optimized = False
            while not optimized:
                range_start = self.max_feature_value * b_range_multiple * -1
                range_end = self.max_feature_value * b_range_multiple

                # Step accross a large range of values by the step_size*b_multiple 
                for b in np.arange(range_start, range_end, step_size * b_multiple):
                    for transformation in transforms:
                        w_transform = w * transformation
                        found_option = True
                        for yi in self.data:
                            for xi in self.data[yi]:
                                # yi(xi.w+b) >= 1
                                if not yi * ( np.dot(w_transform, xi) + b ) >= 1:
                                    found_option = False
                                    
                        if found_option:
                            opt_dict[np.linalg.norm(w_transform)] = [w_transform, b]
                
                if w[0] < 0:
                    optimized = True
                    print('Optimized a step.')
                else:
                    w = w - step_size

            norms = sorted([n for n in opt_dict])
            # ||w|| : [w,b]
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] + step_size*2

    def predict(self, features):
        # sign( x.w+b )
        classification = np.sign( np.dot(np.array(features), self.w) + self.b )
        if classification != 0 and self.visualization:
            self.axis.scatter(features[0], features[1], s=200, marker="*", c=self.colors[classification])

        return classification

    def visualize(self):
        [[ self.axis.scatter(x[0], x[1], s=100, color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        # hyperplane = x.w+b
        # v = x.w+b
        # psv = 1
        # nsv = -1
        # dec = 0
        def get_hyperplane_y_coord(x, w, b, v):
            return (-w[0]*x-b+v) / w[1]

        datarange = (self.min_feature_value*0.9, self.max_feature_value*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # (w.x+b) = 1
        # Plot the Positive Support Vector Hyperplane
        psv1 = get_hyperplane_y_coord(hyp_x_min, self.w, self.b, 1)
        psv2 = get_hyperplane_y_coord(hyp_x_max, self.w, self.b, 1)
        self.axis.plot([hyp_x_min,hyp_x_max],[psv1,psv2], 'k')

        # (w.x+b) = -1
        # Plot the Negative Support Vector Hyperplane
        nsv1 = get_hyperplane_y_coord(hyp_x_min, self.w, self.b, -1)
        nsv2 = get_hyperplane_y_coord(hyp_x_max, self.w, self.b, -1)
        self.axis.plot([hyp_x_min,hyp_x_max],[nsv1,nsv2], 'k')

        # (w.x+b) = 0
        # Plot the Decision Boundary
        db1 = get_hyperplane_y_coord(hyp_x_min, self.w, self.b, 0)
        db2 = get_hyperplane_y_coord(hyp_x_max, self.w, self.b, 0)
        self.axis.plot([hyp_x_min,hyp_x_max],[db1,db2], 'y--')

        plt.savefig('plots/support_vector_machine.png')
        plt.show()

# Create our Support Vector Machine Class
svm = Support_Vector_Machine()

# Train the SVM on the data
svm.fit(data=data_dict)

predict_us = [[0,10],
              [1,3],
              [3,4],
              [3,5],
              [5,5],
              [5,6],
              [6,-5],
              [5,8]]

for p in predict_us:
    svm.predict(p)

svm.visualize()