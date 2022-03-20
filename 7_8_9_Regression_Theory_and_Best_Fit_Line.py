'''
Equation of a line: 
y = mx + b

Best Fit Slope Equation
m = mean(x) * mean(y) - mean(x*y)
    ----------------------------
    (mean(x))^2 - mean(x^2)

Y Intercept Equation
b = mean(y) - m * mean(x)
'''

from statistics import mean
from tkinter import X 
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

# Create the x and y values
x_values = np.array([1,2,3,4,5,6], dtype=np.float64)
y_values = np.array([5,4,6,5,6,7], dtype=np.float64)

def best_fit_slope(x_values, y_values):
    # mean(x) * mean(y)
    m1 = mean(x_values) * mean(y_values)
    # mean(x*y)
    m2 = mean(x_values * y_values) # np arrays will multiply each value together from each array
    # (mean(x))^2 
    m3 = mean(x_values)**2
    # mean(x^2)
    m4 = mean(x_values**2)
    return (m1 - m2) / (m3 - m4)

m = best_fit_slope(x_values, y_values)
print("The bets fit slope is {}".format(m))

def best_fit_intercept(x_values, y_values, m):
    return mean(y_values) - m * mean(x_values)

b = best_fit_intercept(x_values, y_values, m)
print("The bets fit y intercept is {}".format(b))

def best_fit_slope_and_intercept(x_values, y_values):
    m = best_fit_slope(x_values, y_values)
    b = best_fit_intercept(x_values, y_values, m)
    return m, b

regression_line = [m*x+b for x in x_values]

def predict(x_value, m, b):
    return m * x_value + b

predict_x_values = [7, 8, 9, 10]
predict_y_values = []
for x in predict_x_values:
    predict_y_values.append(predict(x, m, b))

# Visualize the data
plt.scatter(x_values, y_values) # plot the data
plt.scatter(predict_x_values, predict_y_values, color='g') # plot our predictions
plt.plot(x_values, regression_line) # plot the regression line
plt.savefig('plots/linear_regression.png')
plt.show()

