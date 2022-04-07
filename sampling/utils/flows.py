import jax.numpy as np
import jax
from jax import random
key = random.PRNGKey(1)  # For seeding random numbers in JAX
# from jax.config import config
# config.update("jax_enable_x64", True)  # needed to switch from float32 to float64

import math
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import os
import numpy as onp  # "Old" numpy
import matplotlib
import matplotlib.image
import matplotlib.pyplot
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import collections as mc
from matplotlib import rcParams, ticker, cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython.display import HTML

import NPEET.npeet.entropy_estimators

rcParams['animation.html'] = 'jshtml'  # Makes the default animation an interactive video
rcParams['animation.embed_limit'] = 2**128  # Allow bigger animations
plt.style.use('seaborn-talk')  # also try 'seaborn-paper', 'fivethirtyeight'


class PlanarFlow(nn.Module):

    def __init__(self, data_dim):
        super().__init__()

        self.u = nn.Parameter(torch.rand(data_dim))
        self.w = nn.Parameter(torch.rand(data_dim))
        self.b = nn.Parameter(torch.rand(1))
        self.h = nn.Tanh()
        self.h_prime = lambda z: (1 - self.h(z) ** 2)

    def constrained_u(self):
        """
        Constrain the parameters u to ensure invertibility
        """
        wu = torch.matmul(self.w.T, self.u)
        m = lambda x: -1 + torch.log(1 + torch.exp(x))
        return self.u + (m(wu) - wu) * (self.w / (torch.norm(self.w) ** 2 + 1e-15))

    def forward(self, z):
        u = self.constrained_u()
        hidden_units = torch.matmul(self.w.T, z.T) + self.b
        x = z + u.unsqueeze(0) * self.h(hidden_units).unsqueeze(-1)
        psi = self.h_prime(hidden_units).unsqueeze(0) * self.w.unsqueeze(-1)
        log_det = torch.log((1 + torch.matmul(u.T, psi)).abs() + 1e-15)
        return x, log_det

class NormalizingFlow(nn.Module):

    def __init__(self, flow_length, data_dim):
        super().__init__()

        self.layers = nn.Sequential(
            *(PlanarFlow(data_dim) for _ in range(flow_length)))

    def forward(self, z):
        log_jacobians = 0
        for layer in self.layers:
            z, log_jacobian = layer(z)
            log_jacobians += log_jacobian
        return z, log_jacobians
    
    
def train(flow, optimizer, nb_epochs, log_density, batch_size, data_dim, device):
    training_loss = []
    for epoch in tqdm(range(nb_epochs)):
        # Generate new samples from the flow
        z0 = torch.randn(batch_size, data_dim).to(device)
        zk, log_jacobian = flow(z0)

        # Evaluate the exact and approximated densities
        flow_log_density = gaussian_log_pdf(z0) - log_jacobian
        exact_log_density = log_density(zk).to(device)

        # Compute the loss
        reverse_kl_divergence = (flow_log_density - exact_log_density).mean()
        optimizer.zero_grad()
        loss = reverse_kl_divergence
        loss.backward()
        optimizer.step()

        training_loss.append(loss.item())
    return training_loss


def plot_flow_density(flow, ax, device, lims=onp.array([[-4, 4], [-4, 4]]), cmap="coolwarm", title=None,
                      nb_point_per_dimension=1000):
    # Sample broadly from the latent space
    latent_space_boundaries = onp.array([[-15, 15], [-15, 15]]);
    xx, yy = onp.meshgrid(
        onp.linspace(latent_space_boundaries[0][0], latent_space_boundaries[0][1], nb_point_per_dimension),
        onp.linspace(latent_space_boundaries[1][0], latent_space_boundaries[1][1], nb_point_per_dimension))
    z = torch.tensor(onp.concatenate((xx.reshape(-1, 1), yy.reshape(-1, 1)), axis=1), dtype=torch.float)
    # Generate data points and evaluate their densities
    zk, log_jacobian = flow(z.to(device))
    final_log_prob = gaussian_log_pdf(z) - log_jacobian.cpu()
    qk = torch.exp(final_log_prob)

    ax.set_xlim(lims[0][0], lims[0][1]); ax.set_ylim(lims[1][0], lims[1][1])
    plot = ax.pcolormesh(
        zk[:, 0].detach().data.cpu().reshape(nb_point_per_dimension, nb_point_per_dimension),
        zk[:, 1].detach().data.cpu().reshape(nb_point_per_dimension, nb_point_per_dimension) * -1,
        qk.detach().data.reshape(nb_point_per_dimension, nb_point_per_dimension),
        cmap=cmap,
        rasterized=True,
    )
    if title is not None:
        plt.title(title, fontsize=22)
    return plot

def plot_exact_density(ax, exact_log_density, lims=onp.array([[-4, 4], [-4, 4]]), nb_point_per_dimension=100,
                       cmap="coolwarm", title=None):
    xx, yy = onp.meshgrid(onp.linspace(lims[0][0], lims[0][1], nb_point_per_dimension),
                         onp.linspace(lims[1][0], lims[1][1], nb_point_per_dimension))
    z = torch.tensor(onp.concatenate((xx.reshape(-1, 1), yy.reshape(-1, 1)), axis=1))
    eld = exact_log_density(z)
    density = torch.exp(eld).reshape(nb_point_per_dimension, nb_point_per_dimension).cpu()
    plot = ax.imshow(density, extent=([lims[0][0], lims[0][1], lims[1][0], lims[1][1]]), cmap=cmap)

    if title is not None:
        plt.title(title, fontsize=22)
    
    return plot
        
def gaussian_log_pdf(z):
    """
    Arguments:
    ----------
        - z: a batch of m data points (size: m x data_dim)
    """
    return -.5 * (torch.log(torch.tensor([math.pi * 2], device=z.device)) + z ** 2).sum(1)

def density_from_grid(xs, grid, c):
    """
    Args:
        xs (Tensor): data of shape (batch_size, 2)
        grid (Tensor): density value at each pixel
        c (float): indicates samples come from range (-c, +c)
    """
    
    # Convert from gaussian range to pixel range
    size = min(grid.shape)
    xs = (xs + c) * size / (2 * c) 
    
    total_area = (2*c)**2
    Z = torch.sum(grid) * total_area / (size * size)
    
    
    # For convenience 
    zeros = torch.zeros_like(xs, device=xs.device)
    zeros += 0.00000000000000000000
    ones = torch.ones_like(xs, device=xs.device)
    zero = torch.tensor(0.0, device=xs.device).type(torch.float64)
    zero += 0.00000000000000000000
    one = torch.tensor(1.0, device=xs.device).type(torch.float64)
    
    # Make sure we won't try to evaluate outside of grid
    nxs = torch.where(xs >= 0, xs, zeros)
    nxs = torch.where(xs < min(grid.shape), nxs, zeros)
    indicies = nxs
    
    # Get density at values
    densities = torch.Tensor([ grid[int(i[1])][int(i[0]) ] for i in indicies ]).type(torch.float64).to(xs.device)
    
    # Fill in bad indicies with zero density
    bds1 = torch.max(torch.where(xs >= 0, zeros, ones), dim=1).values
    bds2 = torch.max(torch.where(xs < min(grid.shape), zeros, ones), dim=1).values
    densities = torch.where(bds1 == 1, zero, densities)
    densities = torch.where(bds2 == 1, zero, densities)

    densities /= Z
    densities = torch.log(densities)
    
    return densities