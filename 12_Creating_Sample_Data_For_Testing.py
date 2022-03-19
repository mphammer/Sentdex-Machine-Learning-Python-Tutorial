from statistics import mean
from tkinter import X 
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

# Previous Lessons
def best_fit_slope_and_intercept(x_values, y_values):
    m = ((mean(x_values) * mean(y_values)) - (mean(x_values * y_values))) / (mean(x_values)**2 - mean(x_values**2))
    b = mean(y_values) - m * mean(x_values)
    return m, b
def squared_error(y_actual, y_predict):
    return sum( (y_predict - y_actual)**2 )
def coefficient_of_determination(y_actual, y_predict):
    y_mean_line = [mean(y_actual) for y in y_actual] # every point is the mean value of all the y_values
    squared_error_regression = squared_error(y_actual, y_predict) # squared error for your predictions
    squared_error_y_mean = squared_error(y_actual, y_mean_line) # squared error for the y_mean_line 
    return 1 - squared_error_regression / squared_error_y_mean


import random 

def create_trendline_dataset(amount_of_data, variance, slope=2, positive_correlation=False):
    '''
    This function creates points that vary around a line that starts at (0,0) and has a slope given by the input value
    '''
    point_along_line = 0
    x_values = []
    y_values = []
    for x in range(amount_of_data):
        y = point_along_line + random.randrange(-variance, variance)
        y_values.append(y)
        x_values.append(x)

        if positive_correlation:
            point_along_line += slope
        else:
            point_along_line -= slope

    return np.array(x_values, dtype=np.float64), np.array(y_values, dtype=np.float64)

# Create a dataset with a large variance and analyze a model
variance = 40
x_values, y_values = create_trendline_dataset(50, variance=variance, slope=2, positive_correlation=True)
m, b = best_fit_slope_and_intercept(x_values, y_values)
regression_line = [m*x+b for x in x_values]
r_squared = coefficient_of_determination(y_values, regression_line)

print("The a variance of {} the Coefficient of Determination is {:.2f}".format(variance, r_squared))
plt.scatter(x_values, y_values)
plt.plot(x_values, regression_line)
plt.show()

# Create a dataset with a small variance and analyze a model
variance = 10
x_values, y_values = create_trendline_dataset(50, variance=variance, slope=2, positive_correlation=True)
m, b = best_fit_slope_and_intercept(x_values, y_values)
regression_line = [m*x+b for x in x_values]
r_squared = coefficient_of_determination(y_values, regression_line)

print("The a variance of {} the Coefficient of Determination is {:.2f}".format(variance, r_squared))
plt.scatter(x_values, y_values)
plt.plot(x_values, regression_line)
plt.show()

# Create a dataset with a large variance and negative correlation, and analyze a model
variance = 80
x_values, y_values = create_trendline_dataset(50, variance=variance, slope=2, positive_correlation=False)
m, b = best_fit_slope_and_intercept(x_values, y_values)
regression_line = [m*x+b for x in x_values]
r_squared = coefficient_of_determination(y_values, regression_line)

print("The a variance of {} the Coefficient of Determination is {:.2f}".format(variance, r_squared))
plt.scatter(x_values, y_values)
plt.plot(x_values, regression_line)
plt.show()