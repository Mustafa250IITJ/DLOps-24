import numpy as np

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ReLU function
def relu(x):
    return np.maximum(0, x)

# Leaky ReLU function
# def leaky_relu(x):
#     return np.where(x > 0, x, x * 0.01)
def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha*x, x)

# Tanh function
def tanh(x):
    return np.tanh(x)


random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]

for rand in random_values:
    print("Sigmoid value of", rand, "=", sigmoid(rand))

# random_values = np.array([-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6])
# # apply the activation functions to the values
# y_sigmoid = sigmoid(random_values)
# print("Sigmoid of random_values", random_values, "is\n", y_sigmoid)

for rand in random_values:
    print("ReLU value of", rand, "=", relu(rand))

for rand in random_values:
    print("Leaky-ReLU value of", rand, "=", leaky_relu(rand))

for rand in random_values:
    print("Tanh value of", rand, "=", tanh(rand))
