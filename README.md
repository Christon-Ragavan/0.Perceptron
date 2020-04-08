## 1. Motivation

The main motivation is to undetstand the nuts and bolts of a simple working neural network. 
Hence we will use a simple neural network implemented in numpy to undercover the details. We will model a simple EX-OR gate in a neural network fashion.
 https://christon-ragavan.github.io/0.Perceptron/

## 2. Topics Covered

1. Forward Propogation (or) weight calculation
2. Backward Propogation (or) weight updates
3. Loss
4. Weights
5. Bias
6. Activation Function

## 3. Basics
### 3.1 Perceptron
The simplest form of perceptron contains one node. The sum of the products of the weights and the inputs is calculated at that node. It is a linear classifier.
One such example is shown as below in the fig

![Perceptron](/images/perceptron_1.png)



### 3.2 XOR gate 
The truth table below shows that the output of an Exclusive-OR gate ONLY goes “HIGH” when both of its two input terminals are at “DIFFERENT” logic levels with respect to each other. If these two inputs, A and B are both at logic level “1” or both at logic level “0” the output is a “0” making the gate an “odd but not the even gate”. In other words, the output is “1” when there are an odd number of 1’s in the inputs.
![XOR Gate](/images/xor_gate.jpg)


## 4. Perceptron

We need a model which output y_hat from the truth table above given the input x and its traning labels y. 
Our perceptron model f(x) can be defined as
![f(x) =x_i\cdot  w_i + b ](https://render.githubusercontent.com/render/math?math=f(x)%20%3Dx_i%5Ccdot%20%20w_i%20%2B%20b%20)

As we see f(x) is a simple dot product of input x_i with its wight w_i with added bias. This modeling of perceptron is well explain with forward and backward propogation which we will discuss below. 

### 4.1 Data Preparation
Data Preparation is the first step for any data driven method. We have x as a input and y as its labels. We henerate our x and y based ont he truth table given above. 

```python
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0, 1, 1, 0]]).T
assert np.shape(x)[0] == np.shape(y)[0]
```


### 4.1 Forward Propogation
In each epoc, the sum of the products of the weights and the inputs is calculated at that node. Simply put.


### 4.1 Activation

![f(x) =  \frac{\mathrm{1} }{\mathrm{1} + e^{-x} } ](https://render.githubusercontent.com/render/math?math=f(x)%20%3D%20%20%5Cfrac%7B%5Cmathrm%7B1%7D%20%7D%7B%5Cmathrm%7B1%7D%20%2B%20e%5E%7B-x%7D%20%7D%20)

![\sigma(x) =  \frac{\mathrm{1} }{\mathrm{1} + e^{-x} } ](https://render.githubusercontent.com/render/math?math=%5Csigma(x)%20%3D%20%20%5Cfrac%7B%5Cmathrm%7B1%7D%20%7D%7B%5Cmathrm%7B1%7D%20%2B%20e%5E%7B-x%7D%20%7D%20)

```math
a + b = c
```


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
