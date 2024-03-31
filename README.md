# Neural Networks Cheat Sheet

## Definitions:

*s = sample*\
*S = number of samples in training batch*\
*l = layer*\
*L = number of layers*\
$n_l$ *= neuron at layer l*\
$N_l$ *= number of neurons in layer l*\
$w_{n_{l-1}n_l}$ *= weight between neurons* $n_{l-1}$ *and* $n_l$\
$b$ *= bias of neuron* $n_l$\
$z$ *= intermediate quantity of neuron* $n_l$\
$y$ *= output of neuron* $n_l$\
$\hat y$ *= target output of neuron* $n_l$\
$A$ *= activation function at neuron* $n_l$ *{Step, Linear, ReLU, Sigmoid, Tanh...}*\
$C$ *= cost function {MSE, SSE, WSE, NSE...}*\
$O$ *= optimization function {Gradient Descend, ADAM, Quasi Newton Method...}*\
$α$ *= learning rate*

<p align="center">
  <img src="https://github.com/alejandrofsevilla/neural_network_notes/assets/110661590/2522d49c-d13d-4544-b7bb-59072d4dabf4" />
</p>

## Neuron Equations:
### Neuron Intermediate Quantity:
$$ \begin{flalign} & z = \sum_{n_{l-1}}^{N_{l-1}}(w_{n_{l-1}n_l} \cdot y_{n_{l-1}} + b) & \end{flalign}$$
### Neuron Output:
$$ \begin{flalign} & y = A\big(z\big) & \end{flalign}$$

## Optimization Algorithm:
In order to reduce the errors of the network, weights and biases are updated after a certain period through an optimization equation $O$, which is a function of the derivatives of the cost function $C$ with respect to the network parameters:

$$ \begin{flalign} &
\Delta w_{n_{l-1}n_l} = - α \cdot O\big(\frac {\partial C}{\partial {w_{n_{l-1}n_l}}}\big)
& \end{flalign} $$

$$ \begin{flalign} &
\Delta b_{n_l} = - α \cdot O\big(\frac {\partial C}{\partial {b_{n_l}}}\big)
& \end{flalign} $$

### Gradient Descend Optimization Algorithm:
Network parameters are updated after every training batch $S$, averaging across all training samples.

$$ \begin{flalign} &
O \big( \frac{\partial C}{\partial {w_{n_{l-1}n_l}}} \big) = \frac{1}{S} \cdot \sum_{s}^S{\frac{\partial C}{\partial {w_{n_{l-1}n_l}}}}
& \end{flalign}$$

## Stochastic Gradient Descend Optimization Algorithm:
It is a gradient descend performed after every training sample $s$.

$$ \begin{flalign} &
O \big( \frac{\partial C}{\partial {w_{n_{l-1}n_l}}} \big) = \frac{\partial C}{\partial {w_{n_{l-1}n_l}}}
& \end{flalign}$$

## ADAM (Adaptive Moment Estimation):
Network parameters are updated after every training samples batch $S$, with an adapted value of the cost function derivatives.

$$ \begin{flalign} &
O \big( \frac{\partial C}{\partial {w_{n_{l-1}n_l}}} \big) = \frac{m_t}{\sqrt{v_t}+\epsilon}
& \end{flalign}$$

$$ \begin{flalign} &
m_t = \beta_1 \cdot m_{t-1} + (1+\beta_1) \cdot \big( \frac{1}{S} \cdot \sum_{s}^S{\frac{\partial C}{\partial {w_{n_{l-1}n_l}}}} \big)
& \end{flalign}$$

$$ \begin{flalign} &
v_t = \beta_2 \cdot v_{t-1} + (1+\beta_2) \cdot \big( \frac{1}{S} \cdot \sum_{s}^S{\frac{\partial C}{\partial {w_{n_{l-1}n_l}}}} \big)
& \end{flalign}$$

Where typically:

$m_0 = 0$ \
$v_0 = 0$ \
$\epsilon = 10^{-8}$ \
$\beta_1 = 0.9$ \
$\beta_2 = 0.999$

## Chain Rule:
The chain rule allows to separate the derivatives described above into components.

$$ \begin{flalign} &
\frac {\partial C}{\partial {w_{n_{l-1}n_l}}} 
= \frac{\partial C}{\partial z_{n_l}} \cdot \frac{\partial z_{n_l}}{\partial {w_{n_{l-1}n_l}}}
= \frac{\partial C}{\partial z_{n_l}} \cdot y_{n_{l-1}}
= \frac{\partial C}{\partial y_{n_l}} \cdot \frac{\partial y_{n_l}}{\partial z_{n_l}} \cdot y_{n_{l-1}}
= \dot C\big(y_{n_l}, \hat y_{n_l}\big) \cdot \dot A\big(z_{n_l}\big) \cdot y_{n_{l-1}}
& \end{flalign}$$

$$ \begin{flalign} &
\frac {\partial C}{\partial {b}} 
= \frac{\partial C}{\partial z_{n_l}}
= \frac{\partial C}{\partial z_{n_l}}
= \frac{\partial C}{\partial y_{n_l}} \cdot \frac{\partial y_{n_l}}{\partial z_{n_l}}
= \dot C\big(y_{n_l}, \hat y_{n_l}\big) \cdot \dot A\big(z_{n_l}\big)
& \end{flalign}$$

## Backpropagation
In order to compute the terms $\dot C \big(y_{n_l}, \hat y_{n_l}\big)$, it would be required to have the output target value for each neuron, $\hat y_{n_l}$. However, a training data set only counts on the value of $\hat y_{n_l}$ for the last layer, where $l = L$. Instead, for all previous layers  $l < L$, components $\dot C \big( y_{n_l}, \hat y_{n_l} \big)$ are computed as a weighted sum of the components obtained for the following layer $\dot C \big(y_{n_{l+1}}, \hat y_{n_{l+1}}\big)$ :

$$ \begin{flalign} &
\dot C \big( y, \hat y \big) = \sum_{n_{l+1}}^{N_{l+1}} w_{n_{l}n_{l+1}} \cdot \dot C \big( y_{n_{l+1}}, \hat y_{n_{l+1}} \big) 
& \end{flalign}$$

## Regularization:

## Cost Functions:
### Quadratic Cost:

$$ \begin{flalign} &
C\big(y, \hat y\big) = \dfrac{1}{2} \big(y - \hat y\big)^2
& \end{flalign} $$

$$ \begin{flalign} &
\dot C\big(y, \hat y\big) = \big(y - \hat y\big)
& \end{flalign} $$

### Cross Entropy Cost:
$$ \begin{flalign} &
C\big(y, \hat y\big) = -\big({\hat y} \text{ ln } y + (1 - {\hat y}) \cdot \text{ ln }(1-y)\big)
& \end{flalign} $$

$$ \begin{flalign} &
\dot C\big(y, \hat y\big) = \frac{y - \hat y}{(1-y) \cdot y}
& \end{flalign} $$

### Exponential Cost:
$$ \begin{flalign} &
C \big( y, \hat y, \tau \big) = \tau \cdot \exp(\frac{1}{\tau} (y - \hat y)^2)
& \end{flalign} $$

$$ \begin{flalign} &
\dot C \big( y, \hat y, \tau \big) = \frac{2}{\tau} \big( y - \hat y \big) \cdot C\big(y, \hat y, \tau \big)
& \end{flalign} $$

### Hellinger Distance:

$$ \begin{flalign} &
C\big(y, \hat y\big) = \dfrac{1}{\sqrt{2}} \big(\sqrt{y} - \sqrt{\hat{y}} \big)^2
& \end{flalign} $$

$$ \begin{flalign} &
\dot C\big(y, \hat y\big) = \dfrac{\sqrt{y} - \sqrt{\hat y}}{\sqrt{2} \cdot \sqrt{y} }
& \end{flalign} $$

## Activation Functions:
### Binary Step:
$$ \begin{flalign} &
\begin{split}A \big(z\big) = \begin{Bmatrix} 1 & z ≥ 0 \\
 0 & z < 0 \end{Bmatrix}\end{split}
& \end{flalign} $$

$\dot A \big(z\big) = 0$

### Linear:
$A \big(z\big) = z$\
$\dot A \big(z\big) = 1$

### ReLU (Rectified Linear Unit):
$$ \begin{flalign} &
\begin{split}A \big(z\big) = \begin{Bmatrix} z & z > 0 \\
 0 & z ≤ 0 \end{Bmatrix}\end{split}
& \end{flalign} $$

$$ \begin{flalign} &
\begin{split}\dot A \big(z\big) = \begin{Bmatrix} 1 & z > 0 \\
 0 & z ≤ 0 \end{Bmatrix}\end{split}
& \end{flalign} $$

### Leaky ReLU:
$$ \begin{flalign} &
\begin{split}A \big(z\big) = \begin{Bmatrix} z & z > 0 \\
 0.01 \cdot z & z ≤ 0 \end{Bmatrix}\end{split}
& \end{flalign} $$

$$ \begin{flalign} &
\begin{split}\dot A \big(z\big) = \begin{Bmatrix} 1 & z > 0 \\
 0.01 & z ≤ 0 \end{Bmatrix}\end{split}
& \end{flalign} $$

### Sigmoid:
$A \big(z\big) = \frac{1} {1 + e^{-z}}$\
$\dot A \big(z\big) = A(z) \cdot (1-A(z))$

### Tanh (Hyperbolic Tangent):
$A \big(z\big) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}$\
$\dot A \big(z\big) = 1 - {A(z)}^2$

## References:
https://en.wikipedia.org/wiki/Activation_function#Table_of_activation_functions \
http://neuralnetworksanddeeplearning.com/ \
https://stats.stackexchange.com/questions/154879/a-list-of-cost-functions-used-in-neural-networks-alongside-applications \
https://towardsdatascience.com/optimizers-for-training-neural-network-59450d71caf6



