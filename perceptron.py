import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def derivated_sigmoid(x):
    return x*(1-x)

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0, 1, 1, 0]]).T

assert np.shape(x)[0] == np.shape(y)[0]

w = 2*np.random.random((np.shape(x)[1], 1))-1
b =1
lr = 0.0001
for i in range(100000):
    layer_1 = x
    y_hat = sigmoid((np.dot(layer_1, w) + b))

    error = y - y_hat
    weight_updates = lr*error*derivated_sigmoid(y_hat)
    w += np.dot(x.T, weight_updates)
    b += np.dot(lr,error)

print("--Output--")
print(y_hat)