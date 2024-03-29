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
$δ_{is{n_l}}$ *= error of neuron* $n_l$ *during iteration i and sample s*\
$α$ *= learning rate*

## Optimization Equations:
$$ \begin{flalign} &
w_{{n_{l-1}n_l}{(i+1)}} = w_{{n_{l-1}n_l}{(i)}} - α \cdot O_{(i)}
& \end{flalign} $$

where $O$ is a function of the derivative of the cost function with respect to the corresponding weight:

$$ \begin{flalign} &
O = f\big(\frac {\partial C}{\partial {w_{n_{l-1}n_l}}}\big)
& \end{flalign} $$

## Chain Rule:

$$ \begin{flalign} &
\frac {\partial C}{\partial {w_{n_{l-1}n_l}}} 
= \frac{\partial C}{\partial z_{n_l}} \cdot \frac{\partial z_{n_l}}{\partial {w_{n_{l-1}n_l}}}
= \frac{\partial C}{\partial z_{n_l}} \cdot y_{n_{l-1}}
= \frac{\partial C}{\partial y_{n_l}} \cdot \frac{\partial y_{n_l}}{\partial z_{n_l}} \cdot y_{n_{l-1}}
= \dot{C}\big(y_{n_l}\big) \cdot \dotσ\big(z_{n_l}\big) \cdot y_{n_{l-1}}
& \end{flalign}$$

Therefore, we need the derivatives of the cost and activation functions, $\dot{C}$ and $\dotσ$.

## Cost Functions:
### MSE (Mean Squared Error):

$$ \begin{flalign} &
MSE = \dfrac{1}{2S}\sum_{s = 1}^S\big(y_{n_L} - \hat y_{n_L}\big)^2
& \end{flalign} $$

$$ \begin{flalign} &
\dot{MSE} = \dfrac{1}{S}\sum_{s = 1}^Sy_{n_L} - \hat y_{n_L}
& \end{flalign} $$

## Activation Functions:
### Linear:
$y_{n_l} = z_{n_l}$\
$\hat y_{n_l} = 1$

### ReLU:
$$ \begin{flalign} &
\begin{split}y_{n_l} = \begin{Bmatrix} z_{n_l} & z_{n_l} > 0 \\
 0 & z_{n_l} <= 0 \end{Bmatrix}\end{split}
& \end{flalign} $$

$$ \begin{flalign} &
\begin{split}\hat y_{n_l} = \begin{Bmatrix} 1 & z_{n_l} > 0 \\
 0 & z_{n_l} <= 0 \end{Bmatrix}\end{split}
& \end{flalign} $$

### Sigmoid:
$y_{n_l} = \frac{1} {1 + e^{-z_{n_l}}}$\
$\hat y_{n_l} = y_{n_l} \cdot (1-y_{n_l})$

## Neuron Equations:
### Neuron Intermediate Quantity:
$$ \begin{flalign} & z_{n_l} = \sum_{n_{l-1}}^{N_{l-1}}(w_{n_{l-1}n_l} \cdot y_{n_{l-1}}) & \end{flalign}$$
### Neuron Output:
$$ \begin{flalign} & y_{n_l} = σ_l\big(z_{n_l}\big) & \end{flalign}$$
### Neuron Error:
$$\begin{flalign} & \delta_{n_l} = \frac{\partial C}{\partial z_{n_l}} & \end{flalign} $$
### Neuron Bias:
We add an extra constant parameter $b_{n_l}=1$ to the neuron input vector and a corresponding weight $w_{{n_l}b}$ that will be adjusted with the others.

