import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


# Load the dataset
ds = pd.read_csv("C:\\Users\\vijender\\Downloads\\Life Expectancy Data.csv")

# Print the column names to check for exact names
print(ds.columns)

# Ensure the column names are correct
gdp_column = "GDP"
life_expectancy_column = "Life expectancy "

# Normalize the input data
ds[gdp_column] = ds[gdp_column] / ds[gdp_column].max()
ds[life_expectancy_column] = ds[life_expectancy_column] / ds[life_expectancy_column].max()

# Define the mean squared error function
def mean_squared_function(w, b, points):
    x = points[gdp_column]
    y = points[life_expectancy_column]
    total_error = np.sum((y - (w * x + b)) ** 2)
    return total_error / float(len(points))

# Define the gradient descent function
def gradient_descent(w_now, b_now, points, L):
    x = points[gdp_column]
    y = points[life_expectancy_column]
    n = len(points)
    w_gradient = -(2 / n) * np.sum(x * (y - (w_now * x + b_now)))
    b_gradient = -(2 / n) * np.sum(y - (w_now * x + b_now))
    w = w_now - w_gradient * L
    b = b_now - b_gradient * L
    return w, b


w = 0
b = 0
L = 0.03
epochs = 1000

# Perform gradient descent
for i in range(epochs):
    w, b = gradient_descent(w, b, ds, L)

print(w, b)

# Plot the results
plt.scatter(ds[gdp_column], ds[life_expectancy_column], color="black")
plt.plot(np.linspace(ds[gdp_column].min(), ds[gdp_column].max(), 100), w * np.linspace(ds[gdp_column].min(), ds[gdp_column].max(), 100) + b, color="red")
plt.xlabel('GDP (normalized)')
plt.ylabel('Life expectancy (normalized)')

plt.show()
