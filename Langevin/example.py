import torch
import torch.distributions as D

eps = 0.01
steps = 500
n_samples = 10000

# d/dx -log p(x) for N(0, 1)
force = lambda x: -x

# Prior distribution from which the initial positions are sampled
prior = D.Uniform(torch.Tensor([-10, -10]), torch.Tensor([10, 10]))

# Run Langevin Dynamics
x = prior.sample((n_samples, ))
for i in range(steps):
    x = x + eps * force(x) + torch.sqrt(2 * eps) * torch.randn(size=x.shape)

x.mean().item(), x.std().item()
# >>> something close to (0.0, 1.0)