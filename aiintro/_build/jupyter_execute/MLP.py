<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Multi-Layer-Perceptron" data-toc-modified-id="Multi-Layer-Perceptron-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Multi Layer Perceptron</a></span><ul class="toc-item"><li><span><a href="#Notations-and-Basic-Characteristics" data-toc-modified-id="Notations-and-Basic-Characteristics-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Notations and Basic Characteristics</a></span></li><li><span><a href="#Early-MLP-Example:-Autonomos-Driving" data-toc-modified-id="Early-MLP-Example:-Autonomos-Driving-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Early MLP Example: Autonomos Driving</a></span></li><li><span><a href="#Architecture" data-toc-modified-id="Architecture-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Architecture</a></span><ul class="toc-item"><li><span><a href="#Number-of-hidden-layers" data-toc-modified-id="Number-of-hidden-layers-1.3.1"><span class="toc-item-num">1.3.1&nbsp;&nbsp;</span>Number of hidden layers</a></span></li><li><span><a href="#Activation--and-Loss-functions" data-toc-modified-id="Activation--and-Loss-functions-1.3.2"><span class="toc-item-num">1.3.2&nbsp;&nbsp;</span>Activation- and Loss-functions</a></span><ul class="toc-item"><li><span><a href="#Activation-functions-in-hidden-layers" data-toc-modified-id="Activation-functions-in-hidden-layers-1.3.2.1"><span class="toc-item-num">1.3.2.1&nbsp;&nbsp;</span>Activation functions in hidden layers</a></span></li><li><span><a href="#Activation-functions-in-the-output-layer-and-loss-functions" data-toc-modified-id="Activation-functions-in-the-output-layer-and-loss-functions-1.3.2.2"><span class="toc-item-num">1.3.2.2&nbsp;&nbsp;</span>Activation functions in the output layer and loss functions</a></span></li></ul></li></ul></li><li><span><a href="#Gradient-Descent-Learning" data-toc-modified-id="Gradient-Descent-Learning-1.4"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>Gradient Descent Learning</a></span></li></ul></li><li><span><a href="#MLP-Implementation-and-Demonstration" data-toc-modified-id="MLP-Implementation-and-Demonstration-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>MLP Implementation and Demonstration</a></span><ul class="toc-item"><li><span><a href="#Definition-of-helper-functions" data-toc-modified-id="Definition-of-helper-functions-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Definition of helper functions</a></span></li><li><span><a href="#Implementation-of-Single-Forward--and-Backward-Pass-in-MLP" data-toc-modified-id="Implementation-of-Single-Forward--and-Backward-Pass-in-MLP-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Implementation of Single Forward- and Backward Pass in MLP</a></span><ul class="toc-item"><li><span><a href="#Classification-into-$K=3$-classes" data-toc-modified-id="Classification-into-$K=3$-classes-2.2.1"><span class="toc-item-num">2.2.1&nbsp;&nbsp;</span>Classification into $K=3$ classes</a></span><ul class="toc-item"><li><span><a href="#Forward-Pass:" data-toc-modified-id="Forward-Pass:-2.2.1.1"><span class="toc-item-num">2.2.1.1&nbsp;&nbsp;</span>Forward-Pass:</a></span></li><li><span><a href="#Contribution-to-Loss-Function" data-toc-modified-id="Contribution-to-Loss-Function-2.2.1.2"><span class="toc-item-num">2.2.1.2&nbsp;&nbsp;</span>Contribution to Loss Function</a></span></li><li><span><a href="#Backward-Pass" data-toc-modified-id="Backward-Pass-2.2.1.3"><span class="toc-item-num">2.2.1.3&nbsp;&nbsp;</span>Backward Pass</a></span></li><li><span><a href="#Forward-Pass-with-adapted-weights" data-toc-modified-id="Forward-Pass-with-adapted-weights-2.2.1.4"><span class="toc-item-num">2.2.1.4&nbsp;&nbsp;</span>Forward-Pass with adapted weights</a></span></li><li><span><a href="#New-Contribution-to-Loss-Function" data-toc-modified-id="New-Contribution-to-Loss-Function-2.2.1.5"><span class="toc-item-num">2.2.1.5&nbsp;&nbsp;</span>New Contribution to Loss Function</a></span></li></ul></li></ul></li></ul></li><li><span><a href="#Multilayer-Perceptron-(MLP)-for-handwritten-digit-recognition" data-toc-modified-id="Multilayer-Perceptron-(MLP)-for-handwritten-digit-recognition-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Multilayer Perceptron (MLP) for handwritten digit recognition</a></span><ul class="toc-item"><li><span><a href="#Class-MLP" data-toc-modified-id="Class-MLP-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Class MLP</a></span></li><li><span><a href="#Load-labeled-data" data-toc-modified-id="Load-labeled-data-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Load labeled data</a></span></li><li><span><a href="#Split-labeled-dataset-into-training--and-test-partition" data-toc-modified-id="Split-labeled-dataset-into-training--and-test-partition-3.3"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>Split labeled dataset into training- and test-partition</a></span></li><li><span><a href="#Generate,-configure,-train-and-test-MLP" data-toc-modified-id="Generate,-configure,-train-and-test-MLP-3.4"><span class="toc-item-num">3.4&nbsp;&nbsp;</span>Generate, configure, train and test MLP</a></span></li></ul></li></ul></div>

# Multi Layer Perceptron
This notebook is based on the theory on Neural Networks as described in [notebook SLP](SLP.ipynb). Since Multi Layer Perceptrons (MLPs) are an extension of SLPs, it is strongly recommended to first read [notebook SLP](SLP.ipynb).

## Notations and Basic Characteristics
A Multi Layer Perceptron (MLP) with $L\geq 2$ layers is a Feedforward Neural Network (FNN), which consists of 
* an input-layer (which is actually not counted as *layer*)
* an output layer 
* a sequence of $L-1$ hidden layers inbetween the input- and output-layer

Usually the number of hidden layers is 1,2 or 3. All neurons of a layer are connected to all neurons of the successive layer. A layer with this property is also called a **fully-connected layer** or a **dense layer**. 

An example of a $L=3$ layer MLP is shown in the following picture. 

<img src="https://maucher.home.hdm-stuttgart.de/Pics/mlpL3.png" alt="Drawing" style="width: 600px;"/>



As in the case of SLPs, the biases in MLP can be modelled implicitily by including to all non-output-layers a constant neuron $x_0=1$, or by the explicit bias-vector $\mathbf{b^l}$ in layer $l$. In the picture above, the latter option is applied.

In order to provide a unified description the following notation is used:
* the number of neurons in layer $l$ is denoted by $z_l$.
* the output of the layer in depth $l$ is denoted by the vector $\mathbf{h^l}=(h_1^l,h_2^l,\ldots,h_{z_l}^l)$, 
* $\mathbf{x}=\mathbf{h^0}$ is the input to the network,
* $\mathbf{y}=\mathbf{h^L}$ is the network's output,
* $\mathbf{b^l}$ is the bias-vector of layer $l$,
* $W^l$ is the weight-matrix of layer $l$. It's entry $W_{ij}^l$ is the weight from the j.th neuron in layer $l-1$ to the i.th neuron in layer $l$. Hence, the weight-matrix $W^l$ has $z_l$ rows and $z_{l-1}$ columns.

With this notation the **Forward-Pass** of the MLP in the picture above can be calculated as follows:

**Output of first hidden-layer:**
$$\left( \begin{array}{c} h_1^1 \\ h_2^1 \\ h_3^1 \\ h_4^1 \end{array} \right) = g\left( \left( \begin{array}{ccc} W_{11}^1 & W_{12}^1 & W_{13}^1 \\ W_{21}^1 & W_{22}^1 & W_{23}^1 \\ W_{31}^1 & W_{32}^1 & W_{33}^1 \\ W_{41}^1 & W_{42}^1 & W_{43}^1 \end{array} \right) \left( \begin{array}{c} x_1 \\ x_2 \\ x_3 \end{array} \right) + \left( \begin{array}{c} b_1^1 \\ b_2^1 \\ b_3^1 \\ b_4^1 \end{array} \right) \right)$$



**Output of second hidden-layer:**

$$\left( \begin{array}{c} h_1^2 \\ h_2^2 \\ h_3^2 \end{array} \right) = g\left( \left( \begin{array}{cccc} W_{11}^2 & W_{12}^2 & W_{13}^2 & W_{14}^2\\ W_{21}^2 & W_{22}^2 & W_{23}^2 & W_{24}^2\\ W_{31}^2 & W_{32}^2 & W_{33}^2 & W_{34}^2 \end{array} \right) \left( \begin{array}{c} h^1_1 \\ h^1_2 \\ h^1_3 \\ h^1_4 \end{array} \right) + \left( \begin{array}{c} b_1^2 \\ b_2^2 \\ b_3^2 \end{array} \right) \right)$$

**Output of the network:**

$$y = \left( \begin{array}{c} h_1^3 \\ \end{array} \right) = g\left( \left( \begin{array}{ccc} W_{11}^3 & W_{12}^3 & W_{13}^3 \end{array} \right) \left( \begin{array}{c} h^2_1 \\ h^2_2 \\ h^2_3 \end{array} \right) + \left( \begin{array}{c} b_1^3 \end{array} \right) \right)$$

As in the case of [Single Layer Perceptrons](SLP.ipynb) the three categories regression, binary- and $K$-ary classification are distinguished. In contrast to SLPs, MLPs are able to **learn non-linear** models. This difference is depicted below: The left hand side shows the linear classification-boundary, as learned by a SLP, whereas on the right-hand side the non-linear boundary, as learned by a MLP from the same training data, is plotted.  

<img src="https://maucher.home.hdm-stuttgart.de/Pics/nonlinearClassification.png" alt="Drawing" style="width: 800px;"/>

## Early MLP Example: Autonomos Driving
The ALVINN net is a MLP with one hidden layer. It has been designed and trained for *road following* in autonomous driving. The input has been provided by a simple $30 \times 32$ greyscale camera. As shown in the picture below, the hidden layer contains only 4 neurons. In the output-layer each of the 30 neurons belongs to one "steering-wheel-direction". The training data has been collected by recording videos while an expert driver steers the car. For each frame (input) the steering-wheel-direction (label) has been tracked. 

<img src="https://maucher.home.hdm-stuttgart.de/Pics/alvinnNN.jpg" style="height:400px"/>

After training the vehicle cruised autonomously for 90 miles on a highway at a speed of up to 70mph. The test-highway has not been included in the training cruises. 

## Architecture
### Number of hidden layers
In the design of MLPs, the number of required hidden layers and the number of neurons per hidden layer are crucial hyperparameters, which strongly influence the network's performance. Appropriate values for these parameters strongly depend on the application and data at hand. They can not be calculated analytically, but have to be determined in corresponding evaluation- and optimization experiments.

In order to roughly determine ranges for a suitable number of hidden neurons, one should consider, that an increasing number of hidden neurons 

* requires more training data to learn a robust model, since more parameters must be learned,
* allows to learn more complex models,
* increases the risk of overfitting.

### Activation- and Loss-functions
#### Activation functions in hidden layers
The type of activation function to be used in the hidden-layers of a MLP is an hyperparameter, which must be configured by the user, i.e. it is not determined by e.g. the application category. Typical activations for the hidden-layers are:

* sigmoid
* tanh
* relu
* leaky relu

Finding the best, or at least an appropriate, activation function for the application and data at hand requires empirical analysis.
#### Activation functions in the output layer and loss functions
The configuration of the activation function in the output-layer and the loss function, which is minimized in the training-stage, depend on the application-category in the same way as in the [SLP](SLP.ipynb):

**Regression:**
* Number of neurons in the output-layer: 1
* Activation function in the output-layer: identity 
* Loss Function: Sum of Squared Errors (SSE)

**Binary Classification:**
* Number of neurons in the output-layer: 1
* Activation function in the output-layer: sigmoid 
* Loss Function: binary Cross-Entropy

**$K$-ary Classification:**
* Number of neurons in the output-layer: K
* Activation function in the output-layer: softmax 
* Loss Function: Cross-Entropy

## Gradient Descent Learning
For training, MLPs apply the same approach as SLPs: Gradient Descent. The general consept of Gradient Descent learning is:
1. Define a **Loss Function** $E(T,\Theta)$, which somehow measures the deviation between the current network $\mathbf{y}$ output and the target output $\mathbf{r}$.
2. Calculate the gradient of the Loss Function: $$\nabla E(T,\Theta) = \left( \begin{array}{c}  \frac{\partial E}{\partial W^l_{1,0}} \\ \frac{\partial E}{\partial W_{1,1}^l} \\ \vdots \\  \frac{\partial E}{\partial W^l_{K,d+1}} \end{array} \right). $$
3. Adapt all parameters into the direction of the negative gradient. This weight adaptation guarantees that the Loss Function is iteratively minimized.: $$W^l_{i,j}=W^l_{i,j}+\Delta W^l_{i,j} = W^l_{i,j}+\eta \cdot -\frac{\partial E}{\partial W^l_{i,j}},$$ where $\eta$ is the important hyperparameter **learning rate**. 

This approach is described in detail in [notebook SLP](SLP.ipynb). For the MLP, here we just present the *Backward Pass* weight adaptation-rule, resulting from the aforementioned Gradient Descent approach. The algorithm is denoted **Backpropagation Algorithm**.

The weight-matrix $W^l$ in layer $l$, with $l \in \lbrace 1,\ldots,L\rbrace$, is adapted in each iteration by 
$$W^l=W^l+ \Delta W^l,$$
where
$$
\Delta W^l = \eta \sum\limits_{t=1}^N \boldsymbol{D}_t^l * (\mathbf{h}_t^{l-1})^T
$$
for Gradient Descent Batch-Learning, and
$$
\Delta W^l = \eta \boldsymbol{D}_t^l * (\mathbf{h}_t^{l-1})^T
$$
for Stochastic Gradient Descent (SGD) Online-Learning. In this adaptation formulas

* $\mathbf{h}_t^{l-1}$ is the output-vector at layer $l-1$, if the t.th training-element $\mathbf{x}_t$ is at the input of the MLP,
* the matrix $\boldsymbol{D}_t^l$ is calculated recursively as follows:

$$
\begin{array}{|c|c|}
		\hline
		layer \; l & \boldsymbol{D}_t^l \\
		\hline
		L & \boldsymbol{\Delta}_t \\
		L-1 & \left( (W^{L})^T * \boldsymbol{\Delta}_t \right) \cdot g'(W^{L-1}\mathbf{h}_t^{L-2}) \\
		L-2 & \left((W^{L-1})^T * \boldsymbol{D}_t^{L-1} \right) \cdot g'(W^{L-2}\mathbf{h}_t^{L-3}) \\
		\vdots & \vdots \\
		l & \left((W^{l+1})^T * \boldsymbol{D}_t^{l+1} \right) \cdot g'(W^{l}\mathbf{h}_t^{l-1}) \\
        \hline
\end{array}
$$
where:
* $*$ denotes matrix-multiplication
* $\cdot$ denotes elementwise-multiplication
* $g'()$ is the first derivation of the activation function applied in layer $l$.
* the error-vector $\Delta_t$ is as defined in [notebook SLP](SLP.ipynb):

$$\boldsymbol{\Delta}_t=\left( \begin{array}{c} \Delta_{t,1} \\ \Delta_{t,2} \\ \vdots \\ \Delta_{t,z_L} \end{array} \right) = \left( \begin{array}{c} r_{t,1} - h^L_{t,1} \\ r_{t,2} - h^L_{t,2} \\ \vdots \\ r_{t,z_L} - h^L_{t,z_L} \end{array} \right)$$

Note that in the calculation of $\boldsymbol{D}_t^l$ depends on the weight-matrices $W^{l+1},\ldots W^{L}$. It is important that for this the old weight-matrices (before the update in the current iteration) are used.


# MLP Implementation and Demonstration

%matplotlib inline
import numpy as np
np.set_printoptions(precision=4,suppress=True)
import random
from matplotlib import pyplot as plt

## Definition of helper functions

#Activation functions
def sigmoid(z):
    """Sigmoid activation function."""
    return 1.0/(1.0+np.exp(-z))

def softmax(z):
    """Softmax activation function."""
    return np.exp(z)/np.sum(np.exp(z),axis=0)

def identity(z):
    """Identity activation function."""
    return z

#Derivative of sigmoid function
def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def oneToMany(j,d=10):
    """Return a d-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((d, 1))
    e[j] = 1.0
    return e

#Loss functions and performance metrics
def mse(r,y):
    return 1.0/len(r)*np.sqrt((r-y)**2)

def accuracy(r,y):
    return np.count_nonzero(r==y)/float(len(r))

def crossEntropy(r,y):
    return np.sum(np.sum(r*np.log2(y)))

## Implementation of Single Forward- and Backward Pass in MLP
As in the corresponding section of the [SLP notebook](SLP.ipynb), this subsection demonstrates
* the forward-pass, i.e. the calculation of the MLP-output from a given input and given weight-matrices
* the backward-pass, i.e the adaptation of the weight-matrices in dependence of the current error-vector at the output of the MLP.

In order to keep this demo as simple as possible only one training element is applied.

### Classification into $K=3$ classes

For this demonstration the MLP architecture of the following picture is applied. The task of this network is to classify input observations with $d=3$ features into one of $K=3$ classes.

<img src="https://maucher.home.hdm-stuttgart.de/Pics/mlpL2K3.png" alt="Drawing" style="width: 600px;"/>

The single hidden layer applies a sigmoid- and the output-layer a softmax-activation. In the picture above explicit biases are drawn. In the following demonstrations, the biases are implemented by the constant $x_0=1$ and the first column of the weight-matrices.

Define an input-vector with 4 elements (the first element is the constant bias $x_0=1$). The target of this input shall be class 3, i.e. $r=(0,0,1)$. Moreover, an arbitrary 
* weight-matrix W1 of size $(4\times4)$ is generated. These are the weight of the first hidden layer, including the biases (first column)
* weight-matrix W2 of size $(3\times5)$ is generated. These are the weight of the output-layer, including the biases (first column)

np.random.seed(1234)
x=np.array([1,1,2,3]) # Training-Element with d=3 features plus the bias x_0=1 (first element in this vector)
print("Current input x=")
print(x)
r=[0,0,1] #Current class is 3. 
print("Current target r=")
print(r)
W1=0.1*(np.random.randint(0,10,(4,4))-5) #Assumed current weight-matrix W1. First column refers to biases in layer 1
print("Current matrix W1=")
print(W1)
W2=0.1*(np.random.randint(0,10,(3,5))-5) #Assumed current weight-matrix W2. first column referst to biases in output layer
print("Current matrix W2=")
print(W2)

#### Forward-Pass:
Calculate for the given input the current output of the MLP:

in1=np.dot(W1,x)
h1=sigmoid(in1)
print("Current output at layer 1 h1=")
print(h1)

h1ext=np.append([1],h1) # input to layer 2 is output of layer 1 + the constant 1 for the bias
print(h1ext)
in2=np.dot(W2,h1ext)
y=h2=softmax(in2)
print("Current output at otuput layer:\ny = h2=")
print(y)

#### Contribution to Loss Function
The current contribution of this training-element to the cross-entropy error-function is:

print(crossEntropy(r,y))

#### Backward Pass
Next, the weight-matrices are adapted according to Stochastic Gradient Descent and the Backpropagation algorithm.

learnrate=0.1
Delta=np.transpose(np.atleast_2d(r-y))
print("Current error vector at output:\n Delta=")
print(Delta)
dW2=learnrate*np.dot(Delta,np.atleast_2d(h1ext))
print("Weight adaptation for otuput layer:\n dW2=")
print(dW2)

print(np.transpose(np.atleast_2d(sigmoid_prime(in1))))
D1=np.dot(np.transpose(W2[:,1:]),Delta)*np.transpose(np.atleast_2d(sigmoid_prime(in1)))#Note that the first column from W2 (biases) have to be excluded!
print("Current matrix D1:\n D1=")
print(D1)
dW1=learnrate*np.dot(D1,np.atleast_2d(x))
print("Weight adaptation for hidden layer 1:\n dW1=")
print(dW1)

newW1=W1+dW1
print("New weight matrix in hidden layer:\n W1=")
print(newW1)
newW2=W2+dW2
print("New weight matrix in output layer:\n W2=")
print(newW2)

#### Forward-Pass with adapted weights
Now, the adapted MLP's output for the same input vector is calculated:

in1=np.dot(newW1,x)
h1=sigmoid(in1)
print("Current output at layer 1 after weight adaptation h1=")
print(h1)

h1ext=np.append([1],h1)
in2=np.dot(newW2,h1ext)
y=h2=softmax(in2)
print("Current output of output layer after weight adaptation:\ny = h2=")
print(y)

#### New Contribution to Loss Function
As can be seen, the new output has a lower contribution to the cross-entropy error-function, than before the weight-adaptation:

print(crossEntropy(r,y))

# Multilayer Perceptron (MLP) for handwritten digit recognition

This example demonstrates the application of a MLP classifier to recognize handwritten digits between 0 and 9. Each handwritten digit is represented as a $8 \times 8$-greyscale image of 4 Bit depth. An image displaying digit $i$ is labeled by the class index $i$ with $i \in \lbrace 0,1,2,\ldots,9\rbrace$. The entire dataset contains 1797 labeled images. This dataset is often applied as a benchmark for evaluating and comparing machine learning algorithms. The dataset is available from different sources. E.g. it is contained in the *scikits-learn* datasets directory. 

## Class MLP
This class implements a Multilayer Perceptron (MLP) with Stochastic Gradient Descent (SGD) learning.
Gradients are calculated using backpropagation.  


class MLP(object):

    def __init__(self, layerlist):
        """The list ``layerlist`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1."""
        
        self.num_layers = len(layerlist)
        self.layerlist = layerlist
        self.biases = [np.random.randn(y, 1) for y in layerlist[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(layerlist[:-1], layerlist[1:])]
        self.testCorrect=[]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            a = sigmoid(np.dot(w, a)+b)
        #softmax activation in output-layer    
        b=self.biases[-1]
        w=self.weights[-1]    
        a = softmax(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                numCorrect=self.evaluate(test_data)
                self.testCorrect.append(numCorrect)
                print("Epoch {0}: {1} / {2}".format(j, numCorrect, n_test))     
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        b=self.biases[-1]
        w=self.weights[-1]    
        z = np.dot(w, activation)+b
        zs.append(z)
        activation = softmax(z)
        activations.append(activation)
        # Weight adaptations in output layer
        delta = self.cost_derivative(activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        # Weight adaptations in all other layers
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

## Load labeled data

The image _handwritten digits_ dataset is loaded from the scikits-learn *datasets* directory. The *load_digits()* function returns a so called *Bunch*, which contains 2 numpy arrays - the images and the corresponding labeles:

from sklearn import datasets
digits = datasets.load_digits()
n_samples = len(digits.images)
print(type(digits))
print(type(digits.images))
print(type(digits.target))
print("Number of labeled images: ",n_samples)
training_data=digits
print(digits.images[0])
print(digits.target[0])

In order to understand the representation of the digits the first 4 images are dispayed in a *matplotlib*-figure. Moreover, the contents of the first image are printed. Each image is a $8 \times 8$-numpy array with integer entries between $0$ (white) and $15$ (black). 

plt.figure(figsize=(12, 10))
NIMAGES=4
for index in range(NIMAGES):
    plt.subplot(1,NIMAGES, index+1)
    plt.imshow(digits.images[index,:], cmap=plt.cm.gray_r)
    plt.title('Training sample of class: %i' % digits.target[index])

## Split labeled dataset into training- and test-partition

NUMTRAIN=1000
training_inputs = [np.reshape(x, (64, 1)) for x in digits.images[:NUMTRAIN]]
training_results = [oneToMany(y) for y in digits.target[:NUMTRAIN]]
training_data = list(zip(training_inputs, training_results))
#test_data = zip(test_inputs, test_results)
test_inputs = [np.reshape(x, (64, 1)) for x in digits.images[NUMTRAIN:]]
test_data = list(zip(test_inputs, digits.target[NUMTRAIN:]))
print(type(training_data[0][0]))
print(len(training_data))
#print training_data[0]

## Generate, configure, train and test MLP

net = MLP([64,30, 10])
net.SGD(training_data, 400, 10, .03,test_data)

if len(net.testCorrect)>0:
    plt.figure(figsize=(12,10))
    plt.plot(range(len(net.testCorrect)),net.testCorrect)
    plt.title("Number of correct classifications")
    plt.xlabel("epoch")
    #plt.hold(True)
    plt.plot([0,len(net.testCorrect)],[n_samples-NUMTRAIN,n_samples-NUMTRAIN])
    plt.show()

print(("Accuracy: ",net.testCorrect[-1]/float((n_samples-NUMTRAIN))))

