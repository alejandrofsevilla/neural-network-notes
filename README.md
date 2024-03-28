# Deep Learning Notes.
## Naming convention:
*i = iteration*\
*s = sample*\
*l = layer*\
*L = number of layers*\
$n_l$ *= neuron at layer l*\
$N_l$ *= number of neurons in layer l*\
$w_{n_{l-1}n_l}$ *= weight between neurons* $n_{l-1}$ *and* $n_l$\
$z_{n_l}$ *= intermediate quantity of neuron* $n_l$\
$y_{n_l}$ *= output of neuron* $n_l$\
$σ_l$ *= activation function at layer l {Step, Linear, ReLU, Sigmoid, Tanh...}*\
$C$ *= cost function {MSE, SSE, WSE, NSE...}*\
$O$ *= optimization function {Gradient Descend, ADAM, Quasi Newton Method...}*\
$δ_{is{n_l}}$ *= error of neuron* $n_l$ *during iteration i and sample s*\
$α$ *= learning rate*

## Optimization Equations:
$$ \begin{flalign} &
w_{{n_{l-1}n_l}{(i+1)}} = w_{{n_{l-1}n_l}{(i)}} - α * O_{(i)}
& \end{flalign} $$

where $O$ is a function of the derivatives of the Cost Function with respect to the corresponding weight:

$$ \begin{flalign} &
O = f\big(\frac {\partial C}{\partial {w_{n_{l-1}n_l}}}\big)
& \end{flalign} $$

## Optimization Methods:
### MSE (Mean Squared Error):

$$ \begin{flalign} &
O_{(i)} = \big(\frac{\partial C}{\partial {w_{n_{l-1}n_l}}}\big)_{(i)}
& \end{flalign} $$

### ADAM (Adaptive Moment Estimation):
...

## Chain Rule:

$$ \begin{flalign} &
\frac {\partial C}{\partial {w_{n_{l-1}n_l}}} 
= \frac{\partial C}{\partial z_{n_l}} \frac{\partial z_{n_l}}{\partial {w_{n_{l-1}n_l}}}
= \frac{\partial C}{\partial z_{n_l}} y_{n_{l-1}}
= \frac{\partial C}{\partial y_{n_l}} \frac{\partial y_{n_l}}{\partial z_{n_l}} y_{n_{l-1}}
& \end{flalign}$$

## Neuron Equations:
### Neuron Intermediate Quantity:
$$ \begin{flalign} & z_{n_l} = \sum_{n_{l-1}}^{N_{l-1}}(w_{n_{l-1}n_l} * y_{n_{l-1}}) & \end{flalign}$$
### Neuron Output:
$$ \begin{flalign} & y_{n_l} = σ_l\big(z_{n_l}\big) & \end{flalign}$$
### Neuron Error:
$$\begin{flalign} & \delta_{n_l} = \frac{\partial C}{\partial z_{n_l}} & \end{flalign} $$

```math
SSE = \sum_{i = 1}^N\big(y_i - \hat y_i\big)^2
```
```math
Mean Squared Error:
MSE = \dfrac{1}{N}\sum_{i = 1}^N\big(y_i - \hat y_i\big)^2
```  
