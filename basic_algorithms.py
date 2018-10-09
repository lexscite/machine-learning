# Basic algorithms

# Set of basic mathematical algorithms for machine learning calculations

from statistics import mean
# Sci-computing lib (N-dimensional array).
import numpy as np
# Graphing lib.
import matplotlib.pyplot as plt
from matplotlib import style

# Some random data set
xs = np.array([1, 2, 3, 4, 5], dtype=np.float64)
ys = np.array([5, 4, 6, 5, 6], dtype=np.float64)

# Algorithm to find the best fit line among the set of coord system points
def best_fit_slope_and_intercept(xs, ys):
  # See this wikipedia article for detailed info on presented formula.
  # <https://en.wikipedia.org/wiki/Linear_equation#Slope%E2%80%93intercept_form>
  # Basically it says that every line can be represented as its slope and
  # interception. So we take average values and calculate average parameters.
  m = (((mean(xs) * mean(ys)) - mean(xs * ys)) /
       ((mean(xs) * mean(xs)) - mean(xs * xs)))
  b = mean(ys) - m * mean(xs)
  
  return m, b

def squared_error(ys_origin, ys_line):
  return sum((ys_line - ys_origin) * (ys_line - ys_origin))

# Coefficent of determination is a number from 0 to 1 which basically
# represents how close the regression line to the "truth". It's as
# "best" as it's close to 1.
def coefficent_of_determination(ys_origin, ys_line):
  # Calculate mean of original ys
  y_mean_line = [mean(ys_origin) for y in ys_origin]
  # Calculate squared error of regression line ys
  squared_error_regr = squared_error(ys_origin, ys_line)
  # Calculate squared error of mean line
  squared_error_y_mean = squared_error(ys_origin, y_mean_line)
  # Calculate coefficent of determination
  return 1 - (squared_error_regr / squared_error_y_mean)

# Represent best fit line from data set
m, b = best_fit_slope_and_intercept(xs, ys)
regression_line = [(m * x) + b for x in xs]

# Get coefficent of determination of regression line
regression_squared_error = coefficent_of_determination(ys, regression_line)
print(regression_squared_error)

# Build graph
#style.use('ggplot')
#plt.scatter(xs, ys, color='#003f72')
#plt.plot(xs, regression_line, label='Regression line')
#plt.legend(loc=4)
#plt.show()
