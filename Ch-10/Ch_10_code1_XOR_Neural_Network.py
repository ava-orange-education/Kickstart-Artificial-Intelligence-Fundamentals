
from random import choice
from numpy import array, dot, random
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

# Sigmoid activation function
def sig_act(x):
    return 1 / (1 + np.exp(-x))

# XOR (linearly non-separable problem)
train_data = [(array([0, 0, 1]), 0), (array([0, 1, 1]), 1), (array([1, 0, 1]), 1), (array([1, 1, 1]), 0)]
v = random.rand(3, 3)  # Weights connecting input to hidden layer
w = random.rand(4, 1)  # Weights connecting hidden to output layer
eta = 0.45
n = 15000

# Training
for i in range(n):
    x, expected = choice(train_data)
    net1 = dot(x, v)
    y = sig_act(net1)  # Hidden layer output
    bias = np.array([1])  # Input to bias at output layer
    y_in = np.concatenate((y, bias), axis=0)
    net2 = dot(y_in, w)
    y_pred = sig_act(net2)  # Output layer response

    # Output layer error gradient
    delta_out = (expected - y_pred) * y_pred * (1 - y_pred)
    # Updating the Hidden-Output weight
    w = w + (eta * delta_out * y_in.reshape(-1, 1))

    # Output error propagates to hidden layer
    bp = delta_out * w[:3]
    y_dash = y * (1 - y)
    delta_hid = np.multiply(y_dash.reshape(-1, 1), bp)

    # Updating weights connecting input to hidden layer
    for j in range(3):
        v[:, j] = v[:, j] + (eta * delta_hid[j] * x)

# Testing (feedforward flow only)
print("Testing Results:")
for x, _ in train_data:
    net1 = dot(x, v)
    y = sig_act(net1)
    bias = np.array([1])  # Input to bias at output layer
    y_in = np.concatenate((y, bias), axis=0)
    net2 = dot(y_in, w)
    y_pred = sig_act(net2)
    print("{} :-> {:.4f}".format(x[:3], y_pred[0]))

# Visualizing hidden layer outputs
hidden_neurons = np.array([[0.5, 0.5, 0.5], [0.39, 0.12, 0.97], [0.45, 0.11, 0.97], [0.67, 0.85, 0.99]])
HN = pd.DataFrame(hidden_neurons, columns=['H1', 'H2', 'H3'])
HN['label'] = ['0', '1', '1', '0']

# 3D scatter plot
fig = px.scatter_3d(HN, x='H1', y='H2', z='H3', color=HN.label, title="Hidden Layer Outputs")
fig.show()
