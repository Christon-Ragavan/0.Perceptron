## 1. Motivation

The main motivation is to undetstand the nuts and bolts of a simple working neural network. 
Hence we will use a simple neural network implemented in numpy to undercover the details. We will model a simple EX-OR gate in a neural network fashion.

## 2. Topics Covered

1. Forward Propogation (or) training (or) weight calculation
2. Backward Propogation (or) weight updates
3. Loss
4. Weights
5. Bias
6. Activation Function

## 3. Basics
### 3.1 Perceptron
The simplest form of perceptron contains one node. The sum of the products of the weights and the inputs is calculated at that node. It is a linear classifier.
One such example is shown as below in the fig

![Perceptron](/images/perceptron_01.png)



### 3.2 EXOR gate 

### Bais
#### 1. Forward Propogation 


##### Bais
Due to absence of bias, model will train over point passing through origin only, which is not in accordance with real-world scenario. Also with the introduction of bias, the model will become more flexible.

```python
import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def derivated_sigmoid(x):
    return x*(1-x)

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0, 1, 1, 0]]).T

assert np.shape(x)[0] == np.shape(y)[0]

w = 2 * np.random.random((np.shape(x)[1], 1))-1
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

```
