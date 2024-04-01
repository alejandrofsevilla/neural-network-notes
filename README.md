# Neural Networks Cheat Sheet

## List of Symbols

$\large s$ *= sample*\
$\large S$ *= number of samples in training batch*\
$\large l$ *= layer*\
$\large L$ *= number of layers*\
$\large n_l$ *= neuron at layer l*\
$\large N_l$ *= number of neurons in layer l*\
$\large w_{n_{l-1}n_l}$ *= weight between neurons* $n_{l-1}$ *and* $n_l$\
$\large b_{n_l}$ *= bias of neuron* $n_l$\
$\large z_{n_l}$ *= intermediate quantity of neuron* $n_l$\
$\large y_{n_l}$ *= output of neuron* $n_l$\
$\large \hat y_{n_l}$ = *target output of neuron* $n_l$\
$\large A$ *= activation function at neuron* $n_l$ *{Step, Linear, ReLU, Sigmoid, Tanh...}*\
$\large C$ *= cost function {MSE, SSE, WSE, NSE...}*\
$\large O$ *= optimization function {Gradient Descend, ADAM, Quasi Newton Method...}*\
$\large α$ *= learning rate*

## Neuron Equations
<p align="center">
  <img src="https://github.com/alejandrofsevilla/neural_network_notes/assets/110661590/2522d49c-d13d-4544-b7bb-59072d4dabf4" />
</p>

### Neuron Intermediate Quantity:
$$ \large 
z_{n_l} = \sum_{n_{l-1}}^{N_{l-1}}(w_{n_{l-1}n_l} \cdot y_{n_{l-1}} + b_{n_l}) 
$$
### Neuron Output:
$$ \large
y_{n_l} = A_{n_l}\big(z_{n_l}\big)
$$


## Optimization Algorithm:
In order to reduce the errors of the network, weights and biases are updated after a certain period, through an optimization function $O$, which is a function of the derivatives of the cost function $C$ with respect to the network parameters:

$$ \large
\Delta w_{n_{l-1}n_l} = - α \cdot O\big(\frac {\partial C}{\partial {w_{n_{l-1}n_l}}}\big)
$$


$$ \large
\Delta b_{n_l} = - α \cdot O\big(\frac {\partial C}{\partial {b_{n_l}}}\big)
$$

### Chain Rule:
The chain rule allows to separate the derivatives of the cost function into components:

$$ \large
\frac {\partial C}{\partial {w_{n_{l-1}n_l}}} 
= \frac{\partial C}{\partial z_{n_l}} \cdot \frac{\partial z_{n_l}}{\partial {w_{n_{l-1}n_l}}}
= \frac{\partial C}{\partial z_{n_l}} \cdot y_{n_{l-1}}
= \frac{\partial C}{\partial y_{n_l}} \cdot \frac{\partial y_{n_l}}{\partial z_{n_l}} \cdot y_{n_{l-1}}
= \dot C\big(y_{n_l}, \hat y_{n_l}\big) \cdot \dot A_{n_l}\big(z_{n_l}\big) \cdot y_{n_{l-1}}
$$

$$ \large
\frac {\partial C}{\partial {b_{n_l}}} 
= \frac{\partial C}{\partial z_{n_l}}
= \frac{\partial C}{\partial y_{n_l}} \cdot \frac{\partial y_{n_l}}{\partial z_{n_l}}
= \dot C\big(y_{n_l}, \hat y_{n_l}\big) \cdot \dot A_{n_l}\big(z_{n_l}\big)
$$

### Backpropagation
The terms $\large \dot C \big(y_{n_l} \hat y_{n_l}\big)$ depend on the output target value for each neuron, $\large\hat y_{n_l}$. However, a training data set only counts on the value of $\large \hat y_{n_l}$ for the last layer, where $\large l = L$. Instead, for all previous layers  $\large l < L$, components $\large \dot C \big( y_{n_l}, \hat y_{n_l} \big)$ are computed as a weighted sum of the components obtained for the following layer $\large \dot C \big(y_{n_{l+1}}, \hat y_{n_{l+1}}\big)$ :

$$ \large
\dot C \big( y, \hat y \big) = \sum_{n_{l+1}}^{N_{l+1}} w_{n_{l}n_{l+1}} \cdot \dot C \big( y_{n_{l+1}}, \hat y_{n_{l+1}} \big) 
$$

## Activation Functions:
### Binary Step:
<p align="center">
  <img src="https://github.com/alejandrofsevilla/neural_networks_cheat_sheet/assets/110661590/298db2aa-8a86-46c4-b59b-9312685d7ebd" alt="drawing" width="500"/>
</p>

$$ \large
\begin{split}A \big(z\big) = \begin{Bmatrix} 1 & z ≥ 0 \\
 0 & z < 0 \end{Bmatrix}\end{split}
$$

$$ \large 
\dot A \big(z\big) = 0
$$

### Linear:
<p align="center">
  <img src="https://github.com/alejandrofsevilla/neural_networks_cheat_sheet/assets/110661590/63038915-6d89-47e4-ae3c-fee5258c1a5b" alt="drawing" width="500"/>
</p>

$$ \large
A \big(z\big) = z
$$

$$ \large
\dot A \big(z\big) = 1
$$

### ReLU (Rectified Linear Unit):
<p align="center">
  <img src="https://github.com/alejandrofsevilla/neural_networks_cheat_sheet/assets/110661590/49f87385-4b0f-4945-a64f-6833469d0381" alt="drawing" width="500"/>
</p>

$$ \large
\begin{split}A \big(z\big) = \begin{Bmatrix} z & z > 0 \\
 0 & z ≤ 0 \end{Bmatrix}\end{split}
$$

$$ \large
\begin{split}\dot A \big(z\big) = \begin{Bmatrix} 1 & z > 0 \\
 0 & z ≤ 0 \end{Bmatrix}\end{split}
$$

### Leaky ReLU:
<p align="center">
  <img src="https://github.com/alejandrofsevilla/neural_networks_cheat_sheet/assets/110661590/3d41c6af-b516-4614-af46-1a1344738fab" alt="drawing" width="500"/>
</p>

$$ \large
\begin{split}A \big(z, \tau \big) = \begin{Bmatrix} z & z > 0 \\
\tau \cdot z & z ≤ 0 \end{Bmatrix}\end{split}
$$

$$ \large
\begin{split}\dot A \big(z, \tau \big) = \begin{Bmatrix} 1 & z > 0 \\
\tau & z ≤ 0 \end{Bmatrix}\end{split}
$$

Where typically:

$$ \large \tau=0.01 $$

### Sigmoid:
<p align="center">
  <img src="https://github.com/alejandrofsevilla/neural_networks_cheat_sheet/assets/110661590/644dee7c-100e-4b24-b8ba-6e2e659135a9" alt="drawing" width="500"/>
</p>

$$ \large
A \big(z\big) = \frac{1} {1 + e^{-z}}
$$

$$ \large
\dot A \big(z\big) = A(z) \cdot (1-A(z))
$$

### Tanh (Hyperbolic Tangent):
<p align="center">
  <img src="https://github.com/alejandrofsevilla/neural_networks_cheat_sheet/assets/110661590/eefc2afd-2c7c-4a2d-8ffd-c4a9efdc2c8e" alt="drawing" width="500"/>
</p>

$$ \large
A \big(z\big) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}
$$

$$ \large
\dot A \big(z\big) = 1 - {A(z)}^2 
$$

### Other Activation Functions:
see: 
* https://en.wikipedia.org/wiki/Activation_function#Table_of_activation_functions 
* https://keras.io/api/layers/activations/

## Cost Functions:
### Quadratic Cost:

$$ \large
C\big(y, \hat y\big) = \dfrac{1}{2} \big(y - \hat y\big)^2
$$

$$ \large
\dot C\big(y, \hat y\big) = \big(y - \hat y\big)
$$

### Cross Entropy Cost:
$$ \large
C\big(y, \hat y\big) = -\big({\hat y} \text{ ln } y + (1 - {\hat y}) \cdot \text{ ln }(1-y)\big)
$$

$$ \large
\dot C\big(y, \hat y\big) = \frac{y - \hat y}{(1-y) \cdot y}
$$

### Exponential Cost:
$$ \large
C \big( y, \hat y, \tau \big) = \tau \cdot \exp(\frac{1}{\tau} (y - \hat y)^2)
$$

$$ \large
\dot C \big( y, \hat y, \tau \big) = \frac{2}{\tau} \big( y - \hat y \big) \cdot C\big(y, \hat y, \tau \big)
$$

### Hellinger Distance:

$$ \large
C\big(y, \hat y\big) = \dfrac{1}{\sqrt{2}} \big(\sqrt{y} - \sqrt{\hat{y}} \big)^2
$$

$$ \large
\dot C\big(y, \hat y\big) = \dfrac{\sqrt{y} - \sqrt{\hat y}}{\sqrt{2} \cdot \sqrt{y} }
$$

## Optimization Functions:
### Gradient Descend:
Network parameters are updated after every training batch $S$, averaging across all training samples.

$$ \large
O \big( \frac{\partial C}{\partial {w_{n_{l-1}n_l}}} \big) = \frac{1}{S} \cdot \sum_{s}^S{\frac{\partial C}{\partial {w_{n_{l-1}n_l}}}}
$$

### Stochastic Gradient Descend:
It is a gradient descend performed after every training sample $s$.

$$ \large
O \big( \frac{\partial C}{\partial {w_{n_{l-1}n_l}}} \big) = \frac{\partial C}{\partial {w_{n_{l-1}n_l}}}
$$

### ADAM (Adaptive Moment Estimation):
Network parameters are updated after every training batch $S$, with an adapted value of the cost function derivatives.

$$ \large
O \big( \frac{\partial C}{\partial {w_{n_{l-1}n_l}}} \big) = \frac{m_t}{\sqrt{v_t}+\epsilon}
$$

$$ \large
m_t = \beta_1 \cdot m_{t-1} + (1+\beta_1) \cdot \big( \frac{1}{S} \cdot \sum_{s}^S{\frac{\partial C}{\partial {w_{n_{l-1}n_l}}}} \big)
$$

$$ \large
v_t = \beta_2 \cdot v_{t-1} + (1+\beta_2) \cdot \big( \frac{1}{S} \cdot \sum_{s}^S{\frac{\partial C}{\partial {w_{n_{l-1}n_l}}}} \big)
$$

Where typically:

$$ \large m_0 = 0 $$

$$ \large v_0 = 0 $$

$$ \large \epsilon = 10^{-8} $$

$$ \large \beta_1 = 0.9 $$

$$ \large \beta_2 = 0.999 $$

## Regularization:
Extra terms are added to the cost function in order to prevent over-fitting.

### L2:

$$ \large
\Delta C = \frac{\lambda}{2n} \sum^W w^2
$$

$\large \lambda$ *= regularization rate* \
$\large W=$  *total number of weights in the network* \

## References:
https://en.wikipedia.org/wiki/Activation_function#Table_of_activation_functions \
http://neuralnetworksanddeeplearning.com/ \
https://stats.stackexchange.com/questions/154879/a-list-of-cost-functions-used-in-neural-networks-alongside-applications \
https://towardsdatascience.com/optimizers-for-training-neural-network-59450d71caf6



