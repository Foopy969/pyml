from model import Model
import layer
import optimizer
import numpy as np

global x_train, y_train, x_test, y_test

with np.load('mnist.npz') as data:
    x_train = data['x_train'] / 255.0
    y_train = data['y_train']
    x_test = data['x_test'] / 255.0
    y_test = data['y_test']

x_train = x_train.reshape(x_train.shape[0], -1)
y_train = np.identity(10)[y_train]
x_test = x_test.reshape(x_test.shape[0], -1)
y_test = np.identity(10)[y_test]

a = Model([
    layer.Dense(784, 64, 'RELU'),
    layer.Dense(64, 16, 'RELU'),
    layer.Dense(16, 10, 'RELU'),
], optimizer.Adam(0.0005))

print(a)

a.fit(x_train, x_test, y_train, y_test, 20)