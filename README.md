# Neural Network Notes

## Definitions:
*i = training iteration*\
*I = number of training iterations*\
*s = sample*\
*S = number of samples*\
*l = layer*\
*L = number of layers*\
$n_l$ *= neuron at layer l*\
$N_l$ *= number of neurons in layer l*\
$w_{n_{l-1}n_l}$ *= weight between neurons* $n_{l-1}$ *and* $n_l$\
$b_{n_l}$ *= bias of neuron* $n_l$\
$z_{n_l}$ *= intermediate quantity of neuron* $n_l$\
$y_{n_l}$ *= output of neuron* $n_l$\
$\hat y_{n_l}$ *= target output of neuron* $n_l$\
$A_{n_l}$ *= activation function at neuron* $n_l$ *{Step, Linear, ReLU, Sigmoid, Tanh...}*\
$C$ *= cost function {MSE, SSE, WSE, NSE...}*\
$O$ *= optimization function {Gradient Descend, ADAM, Quasi Newton Method...}*\
$δ_{n_l}$ *= error at neuron n_l*\
$α$ *= learning rate*

<p align="center">
  <img src="https://github.com/alejandrofsevilla/neural_network_notes/assets/110661590/2522d49c-d13d-4544-b7bb-59072d4dabf4" />
</p>

## Neuron Equations:
### Neuron Intermediate Quantity:
$$ \begin{flalign} & z_{n_l} = \sum_{n_{l-1}}^{N_{l-1}}(w_{n_{l-1}n_l} \cdot y_{n_{l-1}}) & \end{flalign}$$
### Neuron Output:
$$ \begin{flalign} & y_{n_l} = A_{n_l}\big(z_{n_l}\big) & \end{flalign}$$
### Neuron Error:
$$\begin{flalign} & \delta_{n_l} = \frac{\partial C}{\partial z_{n_l}} & \end{flalign} $$
### Neuron Bias:
A constant variable $x_b=1$ is added to the neuron input vector, together with a corresponding weight $w_b$ so that $b_{n_l} = x_b \cdot w_{{n_l}b} = w_{{n_l}b}$.

## Optimization Equations:
$$ \begin{flalign} &
w_{{n_{l-1}n_l}} = w_{{n_{l-1}n_l}} - α \cdot O
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
= \dot C_{n_l} \cdot \dot A_{n_l} \cdot y_{n_{l-1}}
& \end{flalign}$$

where:

$\dot C_{n_l} = f\big(y_{n_l}, \hat y_{n_l}\big)$ and $\dot A_{n_l} = f\big(z_{n_l}\big)$.

## Backpropagation
In order to compute the derivatives of the cost function $\dot C_{n_l} \big(y_{n_l}, \hat y_{n_l}\big) $, we would need the value of the target output for each neuron $\hat y_{n_l}$ , but in our training data, we only have a value of $\hat y_{n_l}$ at the last layer $l = L$. Instead, for all layers where $l < L$ , we compute the derivatives of the cost function as a weighted sum of the derivatives of the cost function in the next layer:

$$ \begin{flalign} &
\dot C_{n_l} = \sum_{n_{l+1} = 1}^{N_{l+1}} w_{n_{l}n_{l+1}} \cdot \dot C_{n_{l+1}} 
& \end{flalign}$$

## Optimization Function Examples:
// TODO: add examples

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

// TODO: more examples

## Activation Function Examples:
### Linear:
$A_{n_l} = z_{n_l}$\
$\dot A_{n_l} = 1$

### ReLU (Rectified Linear Unit):
$$ \begin{flalign} &
\begin{split}A_{n_l} = \begin{Bmatrix} z_{n_l} & z_{n_l} > 0 \\
 0 & z_{n_l} <= 0 \end{Bmatrix}\end{split}
& \end{flalign} $$

$$ \begin{flalign} &
\begin{split}\dot A_{n_l} = \begin{Bmatrix} 1 & z_{n_l} > 0 \\
 0 & z_{n_l} <= 0 \end{Bmatrix}\end{split}
& \end{flalign} $$

### Sigmoid:
$A_{n_l} = \frac{1} {1 + e^{-z_{n_l}}}$\
$\dot A_{n_l} = A_{n_l}(z_{n_l}) \cdot (1-A_{n_l}(z_{n_l}))$

### Tanh (Hyperbolic Tangent):
$A_{n_l} = \frac{e^{z_{n_l}} - e^{-z_{n_l}}}{e^{z_{n_l}} + e^{-z_{n_l}}}$\
$\dot A_{n_l} = 1 - \big({A_{n_l}(z_{n_l})}\big)^2$

// TODO: more examples

## References:
// TODO: add links
