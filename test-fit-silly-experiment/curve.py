import numpy as np
import math
from scipy.optimize import curve_fit

def model(z, c1, c2, c3, c4):
    return c1 + c2 * np.log(z[:,0]) ** c4 + c3 * z[:,1]

# convert to log in order to minimize relative error rather than absolute error
def preprocess(x):
    return np.log(x * 10000)
def postprocess(x):
    return np.exp(x) / 10000

def abs_error_function(params, *args):
    z, x = args
    c1, c2, c3, c4 = params
    return postprocess(x) - postprocess(model(z, c1, c2, c3, c4))

def relative_error_function(params, *args):
    z, x = args
    c1, c2, c3, c4 = params
    predicted_values = model(z, c1, c2, c3, c4)
    return (x - predicted_values) / x

# Given data points
data = np.array([
    (preprocess(0.000015),      500,  1),
    (preprocess(0.000029),      500,  6),
    (preprocess(0.000032),     2331, 10),
    (preprocess(0.000038),     9687, 12),
    (preprocess(0.000086),    99922, 17),
    (preprocess(0.000110),   290937, 19),
    (preprocess(0.000161),   499413, 25),
    (preprocess(0.000298),  9215993, 29),
])

#data[:, 1] = np.log(data[:, 1])

z = data[:, 1:]
x = data[:, 0]

# Initial guess for parameters
initial_guess = (0, 0.001, 1, 1)
initial_guess = (0, 1, 1, 1)
#initial_guess = ( 4.56585372e+04,  1.93306232e-03, -1.75282767e+03,  9.54442260e-01)
#initial_guess = (-6.22526680e-01,  2.49845140e-06,  3.83100945e-01,  5.71790333e+00)

# Define bounds for parameters
bounds = ([-10000, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])

# Fit the model to the data
params, _ = curve_fit(model, z, x, p0=initial_guess, bounds=bounds, maxfev=100000)

# Calculate errors
errors = abs_error_function(params, z, x)
absolute_errors = np.abs(errors)

x = postprocess(x)
percentage_errors = (absolute_errors / x) * 100

# Print results
print("Parameters:", params)
print("Desired values:       ", ", ".join([f"{val:.8f}" for val in (x)]))
print("Predicted values:     ", ", ".join([f"{val:.8f}" for val in (x + errors)]))
print("Absolute Errors:      ", ", ".join([f"{val:.8f}" for val in absolute_errors]))
print("Percentage Errors (%):", ", ".join([f"{val: 10.2f}" for val in percentage_errors]))
