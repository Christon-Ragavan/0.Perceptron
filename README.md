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

![Perceptron](/images/perceptron_2.png)



### 3.2 XOR gate 
The truth table below shows that the output of an Exclusive-OR gate ONLY goes “HIGH” when both of its two input terminals are at “DIFFERENT” logic levels with respect to each other. If these two inputs, A and B are both at logic level “1” or both at logic level “0” the output is a “0” making the gate an “odd but not the even gate”. In other words, the output is “1” when there are an odd number of 1’s in the inputs.
![XOR Gate](/images/xor_gate.jpg)


## 4. Perceptron

We need a model which output y_hat from the truth table above given the input x and its traning labels y. 
Our perceptron model f(x) can be defined as


![basic perceptron](/images/e1.png)


As we see f(x) is a simple dot product of input x_i with its wight w_i with added bias. This modeling of perceptron is well explain with forward and backward propogation which we will discuss below. 

### 4.1 Data Preparation
Data Preparation is the first step for any data driven method. We have x as a input and y as its labels. We henerate our x and y based ont he truth table given above. 

```python
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0, 1, 1, 0]]).T
assert np.shape(x)[0] == np.shape(y)[0]
```

### 4.2 Basic Terms
#### 4.2.1 Weights
 A weight w represent the strength of the connection between units. If the weight from node 1 to node 2 has greater magnitude, it means that neuron 1 has greater influence over neuron 2. A weight brings down the importance of the input value
for our example we can set initialize our weights as

```python
w = 2 * np.random.random((np.shape(x)[1], 1))-1
```

#### 4.2.1 Bias
Bias b is like the intercept added in a linear equation. it is a vector. It is an additional parameter in the Neural Network which is used to adjust the output along with the weighted sum of the inputs to the neuron. Thus, Bias is a constant which helps the model in a way that it can fit best for the given data.


### 4.3 Forward Propogation
Forward Propogation is a two step process.
#### 4.3.1 Activation
First, in each epoc, the sum of the products of the weights and the inputs is calculated at that node. Simply put.


![Activation](/images/e2.png)


#### 4.3.1 Activation Transfer

Followed by this we need an activation function ![\sigma](https://render.githubusercontent.com/render/math?math=%5Csigma). It’s just a thing function that you use to get the output of node. It is also known as Transfer Function. It is used to determine the output of neural network like yes or no. It maps the resulting values in between 0 to 1 or -1 to 1 etc. (depending upon the function).
in many literation  ![\sigma](https://render.githubusercontent.com/render/math?math=%5Csigma) is often refereed as activation function which performs  ![\sigma(f_p)](https://render.githubusercontent.com/render/math?math=%5Csigma(f_p)). 
We use sigmoid activation funtion which can be defines as:


![Sigmoid Activation](/images/e3.png)


In python we can discribe these as below, where y_hat denotes the ouptut:

```python

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

y_hat = sigmoid((np.dot(layer_1, w) + b))
```

In Backpropagation a very import step is computing weight. 

#### 4.3.1 Compute Weights
In order to compute weight you need to have a loss term which says how good your model is perfoeming. This loss function can be as simple as a distance vector between them or a arthematic difference. 

![Loss](/images/e4.png)


### 4.3 Backpropagation
Backpropagation in other words weight update.
Backpropagation  is a widely used algorithm in training feedforward neural networks for supervised learning. In fitting a neural network, backpropagation computes the gradient of the loss function with respect to the weights of the network for a single input–output example, and does so efficiently, unlike a naive direct computation of the gradient with respect to each weight individually. 

![Derivated Sigmoid](/images/e5.png)

Hence we can compute and updates weights by dot product of learning rate, error and 
![\frac{\partial \sigma }{\partial x} ](https://render.githubusercontent.com/render/math?math=%5Cfrac%7B%5Cpartial%20%5Csigma%20%7D%7B%5Cpartial%20x%7D%20)


```python
def derivated_sigmoid(x):
    return x*(1-x)
    weight_updates = lr*error*derivated_sigmoid(y_hat)

```


## 5. Code


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
