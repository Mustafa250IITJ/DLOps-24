import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]

for rand in random_values:
    print("Sigmoid value of", rand, "=", sigmoid(rand))

# random_values = np.array([-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6])
# # apply the activation functions to the values
# y_sigmoid = sigmoid(random_values)
# print("Sigmoid of random_values", random_values, "is\n", y_sigmoid)