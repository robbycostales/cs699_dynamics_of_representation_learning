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


def contour(potential, ax, center=(0,0), x_max=3, n_x=100):
    """Take a 2-d function, potential, and plot contour plot on the supplied matplotlib axis, ax. 
    center gives the center and x_max the range to plot around the center.
    n_x is resolution of grid for evaluation. """
    grid = np.linspace(-x_max, x_max, n_x)
    xv, yv = np.meshgrid(grid + center[0], grid + center[1])
    x_grid = np.array([xv, yv]).reshape((2, n_x * n_x)).T
    energy = potential(x_grid)
    cs = ax.contourf(xv, yv, energy.reshape((n_x, n_x)), locator=ticker.LogLocator(), cmap=cm.PuBu_r)  # LogLocator gives log color scale
    ax.set_title("Potential energy function")
    ax.set_ylabel('$x_2$')
    ax.set_xlabel('$x_1$')

    # Add a colorbar - wow, kind of complicated
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cs, cax=cax)

def vector_field(grad, ax, center=(0,0), x_max=3, n_x=20):
    """Plot vector field for some function, grad, on Axis, ax."""
    ax.set_title("Negative gradient field / Force")
    n_x = 20  # Fewer points for better visibility of arrows
    grid = np.linspace(-x_max, x_max, n_x)
    xv, yv = np.meshgrid(grid + center[0], grid + center[1])
    x_grid = np.array([xv, yv]).reshape((2, n_x * n_x)).T
    grads = grad(x_grid)
    ax.quiver(xv, yv, -grads[:,0], -grads[:,1])
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

def viz_trajectories(t, f=None, image=None, n_x=100, decay=0.95):
    """t is the time series of 2-d trajectories. indices are loop, time, (x,y) location. 
    n_x controls resolution, decay control trailing line decay. 
    f is an optional function to plot the potential."""
    assert t.shape[2] == 2, "Dimensions should be loop, time, (x,y) location"
    ns = t.shape[1] - 1
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
    fig.set_size_inches(8, 8, forward=True)
    
    x_lims = (np.min(t[:,:,0]), np.max(t[:,:,0]))
    y_lims = (np.min(t[:,:,1]), np.max(t[:,:,1]))
    
    if f is not None:  # Contour plot, if f is available
        xv, yv = np.meshgrid(np.linspace(np.min(t[:,:,0]), np.max(t[:,:,0]), n_x), np.linspace(np.min(t[:,:,1]), np.max(t[:,:,1]), n_x))
        x_grid = np.array([xv, yv]).reshape((2, n_x * n_x)).T
        energy = f(x_grid)

        e_grid = energy.reshape((n_x, n_x))
        xs_grid = x_grid[:, 0].reshape((n_x, n_x))
        ys_grid = x_grid[:, 1].reshape((n_x, n_x))
        ax.contourf(xs_grid, ys_grid, e_grid, 
                    locator=ticker.LogLocator(subs='auto'), cmap=cm.PuBu_r,  # Log color scale, 5 contours
                    zorder=0, alpha=0.7)  # zorder puts the contour plot behind other plots
    
    if image is not None:  # Plot image if available
        ax.imshow(image, alpha=0.2)
        x_lims = (0, image.shape[0])
        y_lims = (0, image.shape[1])
        
    cols = []
    print("Formatting data...")
    for ti in t:
        ls = np.array([ti[:-1], ti[1:]]).transpose((1, 0, 2))
        lc = mc.LineCollection(ls, linewidths=3, colors=(0,0,0,0))
        col = ax.add_collection(lc)
        cols.append(col)

    ax.set_title("t={}".format(0))
    ax.set_xlim(left=x_lims[0], right=x_lims[1])
    ax.set_ylim(top=y_lims[0], bottom=y_lims[1])
    ax.set_xlabel('Location ($x_1$)')
    ax.set_ylabel('Location ($x_2$)')

    def update_plot(i):
        for col in cols:
            col.set_color([(0,0,0, decay**(i-j)) if j <= i else (0,0,0,0) for j in range(ns)])
        ax.set_title("t={}".format(i))
        return col,

    print("Creating animation...")
    ani = animation.FuncAnimation(fig, update_plot, frames=range(ns), interval=20, blit=True)
    plt.close()
    return ani