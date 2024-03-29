# Neural Network Notes
## Naming convention:
*i = iteration*\
*I = number of iterations*\
*s = sample*\
*S = number of samples*\
*l = layer*\
*L = number of layers*\
$n_l$ *= neuron at layer l*\
$N_l$ *= number of neurons in layer l*\
$w_{n_{l-1}n_l}$ *= weight between neurons* $n_{l-1}$ *and* $n_l$\
$b_{n_l}$ *= bias at neuron* $n_l$\
$z_{n_l}$ *= intermediate quantity of neuron* $n_l$\
$y_{n_l}$ *= output of neuron* $n_l$\
$\hat y_{n_L}$ *= training sample output for a neuron in the last layer* $n_L$\
$σ_l$ *= activation function at layer l {Step, Linear, ReLU, Sigmoid, Tanh...}*\
$C$ *= cost function {MSE, SSE, WSE, NSE...}*\
$O$ *= optimization function {Gradient Descend, ADAM, Quasi Newton Method...}*\
$δ_{n_l}$ *= error at neuron n_l*\
$α$ *= learning rate*

## Problem:
We need to optimize all values $w_{n_{l-1}n_l}$ that minimizes cost function $C$.

## Neuron Equations:
### Neuron Intermediate Quantity:
$$ \begin{flalign} & z_{n_l} = \sum_{n_{l-1}}^{N_{l-1}}(w_{n_{l-1}n_l} \cdot y_{n_{l-1}}) & \end{flalign}$$
### Neuron Output:
$$ \begin{flalign} & y_{n_l} = σ_l\big(z_{n_l}\big) & \end{flalign}$$
### Neuron Error:
$$\begin{flalign} & \delta_{n_l} = \frac{\partial C}{\partial z_{n_l}} & \end{flalign} $$
### Neuron Bias:
We add an extra constant parameter $b_{n_l}=1$ to the neuron input vector and a corresponding weight $w_{{n_l}b}$ that will be adjusted together with the others.

## Optimization Equations:
$$ \begin{flalign} &
w_{{n_{l-1}n_l}{(i+1)}} = w_{{n_{l-1}n_l}{(i)}} - α \cdot O_{(i)}
& \end{flalign} $$

where $O$ is a function of the derivative of the cost function with respect to the corresponding weight:

$$ \begin{flalign} &
O = f\big(\frac {\partial C_{n_l}}{\partial {w_{n_{l-1}n_l}}}\big)
& \end{flalign} $$

## Chain Rule:

$$ \begin{flalign} &
\frac {\partial C_{n_l}}{\partial {w_{n_{l-1}n_l}}} 
= \frac{\partial C_{n_l}}{\partial z_{n_l}} \cdot \frac{\partial z_{n_l}}{\partial {w_{n_{l-1}n_l}}}
= \frac{\partial C_{n_l}}{\partial z_{n_l}} \cdot y_{n_{l-1}}
= \frac{\partial C_{n_l}}{\partial y_{n_l}} \cdot \frac{\partial y_{n_l}}{\partial z_{n_l}} \cdot y_{n_{l-1}}
= \dot C_{n_l} \cdot \dot σ_{n_l} \cdot y_{n_{l-1}}
& \end{flalign}$$

Therefore, we need the derivatives of the cost and activation functions, $\dot C_{n_l} = f\big(y_{n_l}, \hat y_{n_L}\big)$ and $\dot σ_{n_l} = f\big(z_{n_l}\big)$.

## Backpropagation
We only have a value of the target at the last layer, $\hat y_{n_L}$. In order to compute the derivatives of the cost function in previous layers, $\dot{C_{n_l}}$, we write such derivatives terms of the derivatives of next layer:


## Cost Function Examples:
### Mean Squared Error:

$$ \begin{flalign} &
C = \dfrac{1}{2S}\sum_{s = 1}^S \big(y_{n_L} - \hat y_{n_L}\big)^2
& \end{flalign} $$

$$ \begin{flalign} &
\dot{C} = \dfrac{1}{S}\sum_{s = 1}^S \big(y_{n_L} - \hat y_{n_L}\big)
& \end{flalign} $$

### Mean Binary Cross Cost
$$ \begin{flalign} &
C = -\sum_{s = 1}^S \big({\hat y_{n_L}} \text{ ln } y_{n_L} + (1 - {\hat y_{n_L}}) \cdot \text{ ln }(1-y_{n_L})\big)
& \end{flalign} $$

$$ \begin{flalign} &
\dot{C} = \dfrac{1}{S}\sum_{s = 1}^S \frac{y_{n_L} - \hat y_{n_L}}{(1-y_{n_L}) \cdot y_{n_L}}
& \end{flalign} $$

...


## Activation Function Examples:
### Linear:
$y_{n_l} = z_{n_l}$\
$\dot y_{n_l} = 1$

### ReLU (Rectified Linear Unit):
$$ \begin{flalign} &
\begin{split}y_{n_l} = \begin{Bmatrix} z_{n_l} & z_{n_l} > 0 \\
 0 & z_{n_l} <= 0 \end{Bmatrix}\end{split}
& \end{flalign} $$

$$ \begin{flalign} &
\begin{split}\dot y_{n_l} = \begin{Bmatrix} 1 & z_{n_l} > 0 \\
 0 & z_{n_l} <= 0 \end{Bmatrix}\end{split}
& \end{flalign} $$

### Sigmoid:
$y_{n_l} = \frac{1} {1 + e^{-z_{n_l}}}$\
$\dot y_{n_l} = y_{n_l} \cdot (1-y_{n_l})$

### Tanh (Hyperbolic Tangent):
$y_{n_l} = \frac{e^{z_{n_l}} - e^{-z_{n_l}}}{e^{z_{n_l}} + e^{-z_{n_l}}}$\
$\dot y_{n_l} = 1 - y_{n_l}^{2}$

...
