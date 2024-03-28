import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ReLU function
def relu(x):
    return np.maximum(0, x)

# Leaky ReLU function
def leaky_relu(x):
    return np.where(x > 0, x, x * 0.01)

# Tanh function
def tanh(x):
    return np.tanh(x)

# generate an array of range -10 to 10
x = np.linspace(-10, 10, 100)

# apply the activation functions to the values
y_sigmoid = sigmoid(x)
y_relu = relu(x)
y_leaky_relu = leaky_relu(x)
y_tanh = tanh(x)

# create the plots
plt.figure(figsize=(8, 6))
plt.plot(x, y_sigmoid)
plt.title('Sigmoid')
plt.xlabel('Input')
plt.ylabel('Output')

plt.figure(figsize=(8, 6))
plt.plot(x, y_relu)
plt.title('ReLU')
plt.xlabel('Input')
plt.ylabel('Output')

plt.figure(figsize=(8, 6))
plt.plot(x, y_leaky_relu)
plt.title('Leaky ReLU')
plt.xlabel('Input')
plt.ylabel('Output')

plt.figure(figsize=(8, 6))
plt.plot(x, y_tanh)
plt.title('Tanh')
plt.xlabel('Input')
plt.ylabel('Output')

# display the plots
plt.show()
