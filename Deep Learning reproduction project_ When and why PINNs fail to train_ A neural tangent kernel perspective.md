---
title: 'Deep Learning reproduction project: When and why PINNs fail to train: A neural tangent kernel perspective'
disqus: hackmd
---

## Students

Soufyan Roubiou, 4954866, s.roubiou@student.tudelft.nl
Yang Wang, 926941, y.wang-40@tudelft.nl

## Table of Contents

- Introduction
- Theory
    - Physics-Informed Neural Networks (PINNs)
    - Training dynamics of PINNs through Neural Tangent Kernel (NTK)
- Implementation
- Results
- Conclusion

## Introduction


Physics-informed neural network (PINNs) [2] have shown remarkable empirical results when it comes to solving problems involving partial differential equations. There is, however, not a good understanding how such a constrained neural network behaves during training via gradient descent. Sometimes even is less known about why PINNs sometimes fail to train at all.

To research this one could make use of a Neural Tangent Kernel (NTK): a kernel that describes the evolution of fully-connected neural networks in the infinite width limit during training via gradient descent. Under appropriate conditions, the NTK of PINNs converge to a deterministic kernel that stays constant during training in the infinite width limit. This allows the user to analyze the training dynamics of PINNs using their limiting NTK. From this follows a discrepancy in the convergence rate of the different loss components contributing to the total training error. 

## Problem

In this blog, we aim to reproduce part of the results of the paper 'When and why PINNs fail to train: A neural tangent kernel perspective'. The paper introduces a novel gradient descent algorithm that utilizes the eigenvalues of the NTK to adaptively calibrate the total convergence rate of the total training error to tackle the issue of discrepancy in convergence rates. 

In this blog, we specifically aim to reproduce Figures 1 and 2 of the original paper. To this end we aim to derive and implement the NTKs of a PINN and retrieve its eigenvalues at initialization and at the last step of training. We also find the relative change of the model parameters θ and the NTK of PINNs ***K*** for a fully-connected neural network with one hidden layer and different widths via *n* iterations of full-batch gradient descent.


## Theory

### Physics-informed Neural Networks (PINNs):

Traditionally, a neural network (NN) may schematically be defined as follows[1]:

![](https://i.imgur.com/6tMJXGk.png)


We have some input data $x$ which is trained by our NN and then produces an output $u$. Our NN contains a number of free parameters $\theta$ that can be trained using a loss function that in some way calculates the error between $u$ and the true output labels $u_{true}$. A downside of using a NN like this is that it may have trouble generalising far away from the training data. This is where a PINN can be useful. Schematically, the PINN looks as follows[1]:

![](https://i.imgur.com/9gur07z.png)

Notice that now we also calculate the derivatives of our output $u$, and try to minimize the underlying residual, resulting in an extra term for the loss function, which we will show shortly.


An important part of implementing the PINN, is the partial differential equation (PDE) that will act as the true solution, and with which we will calculate the error of the PINN. To this end, we introduce our PDE as

$$
\mathcal{L}[u](\textbf{x}) = f(\textbf{x}), x \in \Omega
$$

$$
u(\textbf{x}) = g(\textbf{x}), x \in \partial\Omega
$$

where $\mathcal{L}$ is a differential operator and $u(\textbf{x}) : \overline{\Omega} \rightarrow R$ is the unknown solution with $\textbf{x} = (x_1, x_2, ..., x_d)$ where $R$ contains only real numbers. In this blog post we do not consider any time-dependent differential equations. We will instead consider a simple 1D-Poisson equation to train our model on, which is defined as 

$$
u_{xx}(x) = f(x), \forall x \in \Omega
$$

$$
u(x) = g(x), \forall x \in \partial\Omega
$$

where $u_{xx}(x)$ signifies the second-order derivative with respect to $x$ of $u(x)$. For our specific problem, we set $\Omega$ to be the unit interval [0,1] and fabricate the exact solution to this problem taking the form $u(x) = sin(\pi x)$. The corresponding $f$ and $g$ are then given as

$$
f(x) = -\pi^2sin(\pi x), x \in [0,1]
$$

$$
g(x) = 0, x = 0, 1.
$$

We introduce a loss function following the paper, assuming that we can approximate our true solution $u(\textbf{x})$ on our domain with some neural network $u(\textbf{x}, \boldsymbol{\theta})$ where $\boldsymbol{\theta}$ is a collection of all parameters in the network. We can then define a residual of the PDE $r(\textbf{x}, \boldsymbol{\theta})$ as 

$$
r(\textbf{x}, \boldsymbol{\theta}) := \mathcal{L}u(\textbf{x}, \boldsymbol{\theta}) - f(x).
$$

We can learn the parameters of $u(\textbf{x}, \boldsymbol{\theta})$ by minimizing the loss function

$$
\mathcal{L}(\boldsymbol{\theta}) = \mathcal{L}_b(\boldsymbol{\theta}) + \mathcal{L}_r(\boldsymbol{\theta})
$$

where

$$
\mathcal{L}_b(\boldsymbol{\theta}) = \dfrac{1}{2}\sum_{i=1}^{N_b} |u(\textbf{x}_b^i, \boldsymbol{\theta}) - g(\textbf{x}_b^i)|^2
$$

$$
\mathcal{L}_r(\boldsymbol{\theta}) = \dfrac{1}{2}\sum_{i=1}^{N_r} |r(\textbf{x}_b^i, \boldsymbol{\theta})|^2.
$$

Here the $b$ and $r$ denote the boundary and residual losses respectively, while $N_b$ and $N_r$ denote the batch sizes for our training data.

### Training dynamics of PINNs through Neural Tangent Kernel (NTK)

In this section we shortly introduce the NTK of a PINN. We will not go into the derivation deeply, as that is quite mathematically heavy. Instead, we will immediately give the definitions of the necessary NTKs by minimizing our previously defined composite loss function by gradient descent with an infinitesimally small learning rate, yielding the continuous-time gradient flow system

$$
\dfrac{d\boldsymbol{\theta}}{dt} = - \nabla \mathcal{L}(\boldsymbol{\theta}).
$$

Following the derivations of the paper, we introduce the following three NTK matrices:

$$
\textbf{K}_{uu}(t) = \textbf{J}_u(t)\textbf{J}_u^T(t)
$$

$$
\textbf{K}_{rr}(t) = \textbf{J}_r(t)\textbf{J}_r^T(t)
$$

$$
\textbf{K}(t) = \begin{bmatrix}
\textbf{J}_u(t)\\
\textbf{J}_r(t)
\end{bmatrix} \begin{bmatrix}
\textbf{J}_u^T(t),&
\textbf{J}_r^T(t),
\end{bmatrix} 
$$

where $J_u(t)$ and $J_r(t)$ are the Jacobian matrices of $u(t)$ and $\mathcal{L}u(t)$ with respect to $\boldsymbol{\theta}$ respectively. Here we have introduced $\textbf{K}(t)$, which is the neural tangent kernel of the physics-informed neural networks (NTK of PINNs). Using this formulation for our NTK of PINNs and having defined the loss for our PINN as well as the PDE we want to model, we can now implement everything.

## Experimental setup

In order to reproduce the results, we follow the paper in their methods. For Figure 1, we do not train our model yet. Instead, we plot the eigenvalues of $K$, $K_{uu}$ and $K_{rr}$ at initialization in descending order for different fabricated solutions of our 1D-Poisson equations $u(x) = \sin(a \pi x)$, where $a = 1, 2, 4$.


For Figure 2a and 2b, we train a fully-connected neural network with one hidden layer and different widths (10, 100, 500) via $n = 10000$ iterations of full-batch gradient descent with a learning rate of $10^{-5}$. For Figure 2c, we compare the eigenvalues of the NTK of PINNs $\textbf{K}$ at initialization (n = 0) and at the last step of training (n = 10000). The paper does not specify for what width this is done, but we assume the width to be 500.



## Implementation

To reproduce Figure 1 and Figure 2 of the original paper, We reimplement the necessary code for the PINNs and NTK of PINNs using Pytorch. We first create our PINN, which consists of a simple sequential NN which has an input layer, one hidden layer with activation function $Tanh(x)$ as well as an output layer. Important to note is that the weights and biases of both layers are randomly initialised. Hence, it is almost impossible to exactly recreate Figure 1 of the paper, as that shows a plot of the eigenvalues of $K$, $K_{uu}$ and $K_{rr}$ at initialization. Instead, we apply scaling to the weights and biases in our network to try and approximate the figure from the paper. Note that the slopes of both figures are exactly the same, it is just the scaling that is not equal.

The same issue persists for the creation of Figure 2. As we again define and initialize our model with randomly sampled weights and biases, as the paper has done, we get results that do not match up exactly to the figure in the paper. The general structure is contained, and we see that the eigenvalues of $\textbf{K}$ do not change over the course of the iterations, but there is great variation in the relative change of parameters $\boldsymbol{\theta}$ and $\textbf{K}$ for different runs.

Next, in order to determine the NTK of PINNs we calculate the Jacobian matrix of $u(t)$ and $\mathcal{L}u(t)$, both with respect to $\boldsymbol{\theta}$. We then calculate the NTK of PINNs at initialization, which we can then plot.

The next step is to set up the training and test functions. We do this using PyTorch functionality. We define the error to be the mean-squared error (MSE) as the paper does the same. Finally, we also define a function that defines our PDE and defines the unit interval for which our PDE runs over. We can then create our model and train it. 


Next we will present the structure of the implementation followed by a detailed code. The implementation consists of the following components:
1. Define sample numbers and widths of model layers
2. Define the PINNs model
3. Define computation of NTK
4. Define training and testing functions
5. Define dataset generation
6. Reproduce figure 1
    6.1. Generate datasets
    6.2. Create model
    6.3. Initialize model for different fabricate functions
    6.4. Plot eigenvalues of NTK at different initilization (Figure 1)
7. Reproduce figure 2
    7.1. Generate datasets
    7.2. Create model with different hidden width
    7.3. Training different models
    7.4. Plot relative changes of parameters and NTKs (Figure 2)

Detailed code is as follows:

1. Define sample numbers and widths of model layers
```python=
# number of samples
n_x_boundary = 100
n_x_interior = 100
# widths of model layers
in_dim = 1
out_dim = 1
hidden_width = 100
```

2. Define the PINNs model


```python=
class PINN_1D(nn.Module):
  def __init__(self,in_dim, out_dim, hidden_width, weight_std, bias_std):
    super(PINN_1D,self).__init__()
    self.in_dim = in_dim
    self.out_dim = out_dim
    self.hidden_width = hidden_width
    self.weight_std = weight_std
    self.bias_std = bias_std

    self.PINN=nn.Sequential(
        nn.Linear(self.in_dim, self.hidden_width),
        nn.Tanh(),
        nn.Linear(self.hidden_width, self.out_dim)
    )

    self.init_PINN()
  
  # Initialize model parameters
  def init_PINN(self):

    layer1 = self.PINN[0]
    layer2 = self.PINN[2]
  

    std_l1 = 1./np.sqrt(layer1.weight.shape[0])
    weight_random_l1 = torch.normal(0,1,size=layer1.weight.shape,dtype = torch.float32) * std_l1 * self.weight_std
    bias_random_l1 = torch.normal(0,1,size=layer1.bias.shape,dtype = torch.float32) * self.bias_std
    layer1.weight = nn.Parameter(weight_random_l1)
    layer1.bias = nn.Parameter(bias_random_l1)


    std_l2 = 1./np.sqrt(layer2.weight.shape[0])
    weight_random_l2 = torch.normal(0,1,size=layer2.weight.shape,dtype = torch.float32)  * std_l2 * self.weight_std 
    bias_random_l2 = torch.normal(0,1,size=layer2.bias.shape,dtype = torch.float32)  * self.bias_std
    layer2.weight = nn.Parameter(weight_random_l2)
    layer2.bias = nn.Parameter(bias_random_l2)

    

  def forward(self, x):
    return self.PINN(x)
```
3. Define computation of NTK
```python=
# Jacobian function 
def calculate_jacobian(x_boundary, x_interior, model):

  model.train()

  mu_x, sigma_x = x_interior.mean(0), x_interior.std(0)
  J_u = []
  for x in x_boundary:
    x.requires_grad_()
    model.zero_grad()
    model(x).backward()
    grads = []
    for param in model.parameters():
      grads.append(param.grad.view(-1))
    grads = torch.cat(grads)
    J_u.append(grads)
  J_u = torch.stack(J_u,0)


  J_f = []
  for x in x_interior:
    x.requires_grad_()
    model.zero_grad()
    u_pred = model(x)
    
    u_x = torch.autograd.grad(u_pred, x, grad_outputs=torch.ones_like(u_pred),create_graph=True)[0] / sigma_x
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),create_graph=True)[0] / sigma_x

    u_xx.backward(retain_graph=True)
    
    grads = []
    for param in model.parameters():
      if param.grad==None:
        grads.append(torch.zeros_like(torch.Tensor(param.data.shape).view(-1)).to(x.device))
      else:
        grads.append(param.grad.view(-1))
    grads = torch.cat(grads)
    J_f.append(grads)
  J_f = torch.stack(J_f,0)

  J_uf = torch.cat((J_u,J_f),0)

  return J_u, J_f, J_uf

# NTK function
def compute_ntks(J_u, J_f, J_uf):
  K_uu = torch.matmul(J_u,J_u.T)
  K_ff = torch.matmul(J_f,J_f.T)
  K_uf = torch.matmul(J_uf,J_uf.T)

  L_uu = torch.linalg.eigvals(K_uu)
  L_ff = torch.linalg.eigvals(K_ff)
  L_uf = torch.linalg.eigvals(K_uf)

  L_uu_real = torch.real(L_uu)
  L_uu_real, indeces = torch.sort(L_uu_real,descending=True)
  L_uu_real = L_uu_real.cpu()

  L_ff_real = torch.real(L_ff)
  L_ff_real, indeces = torch.sort(L_ff_real,descending=True)
  L_ff_real = L_ff_real.cpu()

  L_uf_real = torch.real(L_uf)
  L_uf_real, indeces = torch.sort(L_uf_real,descending=True)
  L_uf_real = L_uf_real.cpu()
  return L_uf_real, L_uu_real, L_ff_real, K_uf
```

4. Define training and testing functions
```python=
# Train function
def train(x_boundary, x_interior, g_boundary, f_interior, model, loss_fn, optimizer):

  model.train()

  # Normalize
  mu_x, sigma_x = x_interior.mean(0), x_interior.std(0)

  x_interior = (x_interior-mu_x)/sigma_x 
  x_boundary = (x_boundary-mu_x)/sigma_x 
  x_interior.requires_grad_()
  x_boundary.requires_grad_()
  

  # boundary prediction
  u_boundary_pred = model(x_boundary)
  g_pred = u_boundary_pred
  g_true = g_boundary
  
  # interior prediction
  u_interior_pred = model(x_interior)
  u_x = grad(u_interior_pred, x_interior, grad_outputs=torch.ones_like(u_interior_pred),create_graph=True)[0]
  u_xx = grad(u_x, x_interior, grad_outputs=torch.ones_like(u_x),create_graph=True)[0]
  f_pred = u_xx
  f_true = f_interior
  

  # loss
  L_b = loss_fn(g_pred, g_true)
  L_r = loss_fn(f_pred, f_true)
  L = L_b + L_r
  

  # backpropagation
  optimizer.zero_grad()
  L.backward()
  optimizer.step()

  # record parameters
  params = []
  for param in model.parameters():
    params.append(param.data.view(-1))
  params = torch.cat(params)

  # record ntk and eigenvalues
  J_u, J_f, J_uf = calculate_jacobian(x_boundary, x_interior, model)
  L_uf_real, L_uu_real, L_ff_real, K = compute_ntks(J_u, J_f, J_uf)
  
  eign = L_uf_real

  loss = L.item()
  return loss, params, K, eign



# Test function
def test(a, x_boundary, x_interior, g_boundary, f_interior, model, loss_fn):
  # Normalize
  mu_x, sigma_x = x_interior.mean(0), x_interior.std(0)

  x_interior = (x_interior-mu_x)/sigma_x 
  x_boundary = (x_boundary-mu_x)/sigma_x 
  
  model.eval()
  with torch.no_grad():
    u_boundary_pred = model(x_boundary)
    u_interior_pred = model(x_interior)
  
  u_boundary_true = u_fn(a, x_boundary)
  u_interior_true = u_fn(a, x_interior)
  # g_true = g_boundary

  loss_boudary = loss_fn(u_boundary_pred, u_boundary_true)
  loss_interior = loss_fn(u_interior_pred, u_interior_true)
  print(f"loss_boundary: {loss_boudary} loss_interior: {loss_interior}")
```
5. Define dataset generation function
```python=
# Generate dataset
def create_pde(n_x_boundary, n_x_interior, u_fn, f_fn, g_fn, a, device):
  x_boundary_left = torch.zeros(n_x_boundary // 2,1)
  x_boundary_right = torch.ones(n_x_boundary // 2,1)
  x_boundary = torch.vstack([x_boundary_left,x_boundary_right]).to(device)

  x_interior = torch.linspace(0,1, n_x_interior).to(device)
  x_interior = torch.unsqueeze(x_interior, dim=1)

  u_boundary = u_fn(x_boundary, a).to(device)
  u_interior = u_fn(x_interior, a).to(device)
  
  g_boundary = g_fn(x_boundary, a).to(device)
  f_interior = f_fn(x_interior, a).to(device)
  return x_boundary, x_interior, g_boundary, f_interior, 
```

6. Reproduce figure 1
Compute eigenvalues of NTKs at initialization
```python=
# compute eigenvalues of ntk at initialization
L_init = []
weight_std = 1.7*1e-1
bias_std = 3*1e-3 
for a in [1,2,4]:
  model = PINN_1D(in_dim, out_dim, hidden_width, weight_std, bias_std).to(device)

  print(a)
  x_boundary, x_interior, g_boundary, f_interior = create_pde(n_x_boundary, n_x_interior, u_fn, f_fn, g_fn, a, device)
  
  # Normalize
  mu_x, sigma_x = x_interior.mean(0), x_interior.std(0)
  x_interior = (x_interior-mu_x)/sigma_x 
  x_boundary = (x_boundary-mu_x)/sigma_x 

  
  J_u, J_f, J_uf = calculate_jacobian(x_boundary, x_interior, model)
  L_uf_init, L_uu_init, L_ff_init, K = compute_ntks(J_u, J_f, J_uf)
  L_init.append([L_uf_init, L_uu_init, L_ff_init])
```
Plot figure 1
```python=
# plot figure 1
fig, ax = plt.subplots(1, 3,figsize=(20, 4))
ax[0].set_xscale('log')
ax[1].set_xscale('log')
ax[2].set_xscale('log')


# fig.suptitle('Horizontally stacked subplots')
ax[0].plot(L_init[0][0], label = 'a = 1')
ax[1].plot(L_init[0][1], label = 'a = 1')
ax[2].plot(L_init[0][2], label = 'a = 1')

ax[0].plot(L_init[1][0], label = 'a = 2')
ax[1].plot(L_init[1][1], label = 'a = 2')
ax[2].plot(L_init[1][2], label = 'a = 2')

ax[0].plot(L_init[2][0], label = 'a = 4')
ax[1].plot(L_init[2][1], label = 'a = 4')
ax[2].plot(L_init[2][2], label = 'a = 4')

ax[0].legend()
ax[1].legend()
ax[2].legend()

ax[0].set_title('K_uf')
ax[1].set_title('K_uu')
ax[2].set_title('K_ff')

ax[0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax[1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax[2].ticklabel_format(axis='y', style='sci', scilimits=(0,0))

ax[0].set_ylim(0, 3.5e5)
ax[1].set_ylim(0, 5e3)
ax[2].set_ylim(0, 3.5e5)

ax[0].set_yticks([0, 1e5, 2e5, 3e5])
ax[2].set_yticks([0, 1e5, 2e5, 3e5])

plt.show()
```

7. Reproduce figure 2
Train the PINNs network and record NTKs and network arameters
```python=
# Record relative changes of NTKs and parameters of networks with 
# different hidden widths
param_diff_all = []
ntk_diff_all = []
eign_all = []

weight_std = 1*1e1 
bias_std = 0.3*1e1 


# Train models with different hidden width
for hidden_width in [10, 100, 500]:

  # Create model
  model = PINN_1D(in_dim, out_dim, hidden_width, weight_std, bias_std).to(device)

  # Define loss and optimizer
  loss_fn = nn.MSELoss(reduction='mean')
  optimizer = torch.optim.SGD(model.parameters(),lr=1e-5)
  scheduler = ExponentialLR(optimizer, gamma = 0.9)

  a = 1
  x_boundary, x_interior, g_boundary, f_interior = create_pde(n_x_boundary, n_x_interior, u_fn, f_fn, g_fn, a, device)
  
  # Record data at every iteration
  params = []
  ntks = []
  eigns = []

  epoch = 1000
  for iter in range(epoch):
    loss, param, ntk, eign = train(x_boundary, x_interior, g_boundary, f_interior, model, loss_fn, optimizer)
    param = param.cpu()
    ntk = ntk.cpu()

    params.append(param)
    ntks.append(ntk)
    eigns.append(eign)

    if iter % 200 == 0:
      print(f"loss: {loss:>3f}      || iter: {iter:>5d}")

    if iter % 1000 == 0:
      scheduler.step()
  print("Done!")

  test(a, x_boundary, x_interior, g_boundary, f_interior, model, loss_fn)
  # data at initialization
  param_0 = params[0]
  ntk_0 = ntks[0]
  eign_0 = eigns[0]

  # data differences
  param_diffs = []
  ntk_diffs = []
  for i in range(len(params)):
    param_diff = np.linalg.norm(params[i] - param_0, ord=2)/np.linalg.norm(param_0, ord=2)
    ntk_diff = np.linalg.norm(ntks[i] - ntk_0, ord=2)/np.linalg.norm(ntk_0, ord=2)

    param_diffs.append(param_diff)
    ntk_diffs.append(ntk_diff)

  param_diff_all.append(param_diffs)
  ntk_diff_all.append(ntk_diffs)
  eign_all.append(eigns)
```
Plot figure 2
```python=
# plot figure 2
fig, ax = plt.subplots(1, 3,figsize=(20, 4))
ax[2].set_xscale('log')

ax[0].plot(param_diff_all[0], label = 'Width = 10')
ax[0].plot(param_diff_all[1], label = 'Width = 100')
# ax[0].plot(param_diff_all[2], label = 'Width = 500')
ax[1].plot(ntk_diff_all[0], label = 'Width = 10')
ax[1].plot(ntk_diff_all[1], label = 'Width = 100')
# ax[1].plot(ntk_diff_all[2], label = 'Width = 500')
ax[2].plot(eign_all[-1][0], label = 'n = 0')
ax[2].plot(eign_all[-1][-1], label = 'n = 10,000', linestyle = '--')

ax[0].legend()
ax[1].legend()
ax[2].legend()

ax[0].set_title('Change of parameters theta')
ax[1].set_title('Change of kernels $K$')
ax[2].set_title('trained')

ax[0].ticklabel_format(axis='y')
ax[1].ticklabel_format(axis='y')
ax[2].ticklabel_format(axis='y', style='sci', scilimits=(0,0))

ax[2].set_yscale('log')
ax[2].set_yticks([1e-21, 1e-16, 1e-11, 1e-6, 1e-1, 1e4])
plt.show()
```



## Results

In this section we display the original figures taken from the paper, as well as the reproduced figures created by us.

### Reproduction of Figure 1
Original figure
![](https://i.imgur.com/Fic22Ky.png)
Reproduced figure
![](https://i.imgur.com/NJZ4OwG.png)

### Reproduction of Figure 2
Original figure
![](https://i.imgur.com/XclqCjc.png)
Reproduced figure
![](https://i.imgur.com/Ws0T5So.png)




## Conclusion
In this project, we reimplemented a Physics-Informed Neural Network (PINN) and its Neural Tangent Kernel (NTK) in the framework of Pytorch. The eigenvalues of NTK of PINN for different fabricate functions were plotted in descent order in Figure 1. Our reimplementation shows consistence in patterns of eigenvalues with that of the original paper. In Figure 2, the relative change of parameters of the PINN and that of the NTKs at each iteration were plotted. The eigenvalues of the NTKs at the initilization and after training were presented as well. The reproduced figure presents similar structure as that of the original paper, demonstrating a favorable reproduction of the paper.


There are numerous benefits to performing a reproduction. First and foremost, we can determine whether the results portrayed by the paper are correct and easily verifiable. As shown by our figures, the results indeed appear to be valid, and we have been able to verify them. Important side-note here is that the authors make use of randomly sampled initialization, this makes it hard to create identical reproductions of the figures, but we can get a good idea of whether the methods employed are valid or not. One of the main conclusions of the paper is that the NTK of PINNs converges to a deterministic kernel. We indeed confirm that by our reproduction. Lastly, this reproduction gave us the ability to convert the tensorflow-based source code of the authors into PyTorch. This code can be used for future work to build upon. 

Soufyan worked on the implementation of NTKs, while Yang worked on the implementation of the PINN. Together we wrote the report and produced the results, as well as discussed any problems we had.


Bibliography
---
[1] Ben Moseley, AUGUST 28, 2021, https://benmoseley.blog/my-research/so-what-is-a-physics-informed-neural-network/.
[2] Wang, Sifan, Xinling Yu, and Paris Perdikaris. "When and why PINNs fail to train: A neural tangent kernel perspective." Journal of Computational Physics 449 (2022): 110768.