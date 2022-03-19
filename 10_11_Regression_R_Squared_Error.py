from statistics import mean
from tkinter import X 
import numpy as np 

# Previous Lessons
x_values = np.array([1,2,3,4,5,6], dtype=np.float64)
y_values = np.array([5,4,6,5,6,7], dtype=np.float64)
def best_fit_slope_and_intercept(x_values, y_values):
    m = ((mean(x_values) * mean(y_values)) - (mean(x_values * y_values))) / (mean(x_values)**2 - mean(x_values**2))
    b = mean(y_values) - m * mean(x_values)
    return m, b
m, b = best_fit_slope_and_intercept(x_values, y_values)
regression_line = [m*x+b for x in x_values]

'''
Coefficient of Determination - Used to Evaluate How Good Your Model Is
r^2 == The Coefficient of Determination

r^2 = 1 - squared_error(y_hat) / squared_error(y_mean)

squared_error(y_hat) == The Error Between Your Predictions and the Actual Y Values
squared_error(y_mean) == The Error Between The Mean Value of All Y Values and the Actual Y Values (if you graph this it will be a straight line at the mean)

Theoretically your Model (predictions) should have much less error than just a straight line at the mean. Therefore (squared_error(y_hat) / squared_error(y_mean)) should
be very small... so then 1 - the_value should be close to 1.
'''

def squared_error(y_actual, y_predict):
    return sum( (y_predict - y_actual)**2 )

def coefficient_of_determination(y_actual, y_predict):
    y_mean_line = [mean(y_actual) for y in y_actual] # every point is the mean value of all the y_values
    squared_error_regression = squared_error(y_actual, y_predict) # squared error for your predictions
    squared_error_y_mean = squared_error(y_actual, y_mean_line) # squared error for the y_mean_line 
    return 1 - squared_error_regression / squared_error_y_mean

r_squared = coefficient_of_determination(y_values, regression_line)
print("The Coefficient of Determination is {:.2f}".format(r_squared))