import numpy as np
from matplotlib import pyplot as plt
import scaling
import os
import numpy as np



# Read data matrix X and labels y from text file.
def read_data(file_name):
    # Check if file exists
    if not os.path.exists(file_name):
        print(f"File not found: {file_name}")
        return None, None

    try:
        # Load the data assuming whitespace-delimited columns
        data = np.loadtxt(file_name, delimiter=None, encoding='utf-8')
        X = data[:, 0].reshape(-1, 1)  # Reshape to make it a column vector
        y = data[:, 1]
        return X, y
    except Exception as e:
        print(f"Error reading the file '{file_name}': {e}")
        return None, None

# Implement gradient descent algorithm to compute w = [w0, w1].
def train(X, y, alpha, epochs):
    # Add bias term (column of ones) to X
    X = np.c_[np.ones(X.shape[0]), X]
    w = np.zeros(X.shape[1])  # Initialize weights

    for epoch in range(epochs):
        gradient = compute_gradient(X, y, w)
        w -= alpha * gradient  # Update the weights

    return w

# Compute Root Mean Squared Error (RMSE).
def compute_rmse(X, y, w):
    X = np.c_[np.ones(X.shape[0]), X]
    predictions = X.dot(w)
    return np.sqrt(np.mean((predictions - y) ** 2))

# Compute objective (cost) function.
def compute_cost(X, y, w):
    X = np.c_[np.ones(X.shape[0]), X]
    predictions = X.dot(w)
    return np.mean((predictions - y) ** 2) / 2

# Compute the gradient for gradient descent.
def compute_gradient(X, y, w):
    m = X.shape[0]
    predictions = X.dot(w)
    error = predictions - y
    grad = (X.T.dot(error)) / m
    return grad

##======================= Main program =======================##

# Read the training and test data.
Xtrain, ttrain = read_data("C:\\Users\\Kübra\\Desktop\\train.txt")
Xtest, ttest = read_data("C:\\Users\\Kübra\\Desktop\\test.txt")

# Standardize the features
mean, std = scaling.mean_std(Xtrain)
Xtrain_std = scaling.standardize(Xtrain, mean, std)
Xtest_std = scaling.standardize(Xtest, mean, std)

# Train the model
alpha = 0.1
epochs = 500
w = train(Xtrain_std, ttrain, alpha, epochs)

# Print model parameters
print("Trained parameters:", w)

# Compare with normal equation solution
X_b = np.c_[np.ones(Xtrain_std.shape[0]), Xtrain_std]
w_normal_eq = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(ttrain)
print("Parameters from normal equation:", w_normal_eq)

# Plot cost function over epochs
costs = []
for epoch in range(epochs):
    costs.append(compute_cost(Xtrain_std, ttrain, w))
plt.plot(range(epochs), costs)
plt.xlabel('Epoch')
plt.ylabel('Cost J(w)')
plt.title('Cost function over epochs')
plt.show()

# Visualize training data, test data, and linear regression line
plt.plot(Xtrain, ttrain, 'bo', label="Train data")
plt.plot(Xtest, ttest, 'gx', label="Test data")

# Plot linear regression line
X_line = np.linspace(Xtrain.min(), Xtrain.max(), 100).reshape(-1, 1)
X_line_std = scaling.standardize(X_line, mean, std)
X_line_std_b = np.c_[np.ones(X_line_std.shape[0]), X_line_std]
y_line = X_line_std_b.dot(w)
plt.plot(X_line, y_line, 'r-', label="Linear regression")

plt.xlabel("Floor size")
plt.ylabel("House Price")
plt.legend()
plt.show()
