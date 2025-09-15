from matplotlib import pyplot as plt
import numpy as np
from dyn.model import Model


def plot_nominal_trajectory(m: Model, X, ax=None):
    """
    :param X: nominal trajectory
    :return: plot the nominal trajectory
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # plot the nominal trajectory
    colors = plt.cm.viridis(np.linspace(0, 1, m.nx + 2))
    time = np.arange(0, X.shape[1]) * m.dt
    for i in range(m.nx):
        ax.plot(time, X[i, :], color=colors[i + 1])

    return ax


def plot_input_nominal_trajectory(m: Model, U, ax=None):
    """
    :param X: nominal trajectory
    :return: plot the nominal trajectory
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    colors = plt.cm.viridis(np.linspace(0, 1, m.nu + 2))
    time = np.arange(0, U.shape[1]) * m.dt
    for i in range(m.nu):
        ax.plot(time, U[i, :], color=colors[i + 1])
    return ax


def plot_tube(m: Model, backoff, center, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # transpose the matrices if they are not in the right shape: (nx, N+1)
    if not backoff.shape[0] == m.nx:
        backoff = backoff.T
    if not center.shape[0] == m.nx:
        center = center.T

    time = np.arange(0, center.shape[1]) * m.dt

    colors = plt.cm.viridis(np.linspace(0, 1, m.nx + 2))
    for i in range(m.nx):
        lower_bound = center[i] - backoff[i]
        upper_bound = center[i] + backoff[i]
        ax.fill_between(time, lower_bound, upper_bound, color=colors[i + 1], alpha=0.5, label='Bounds')

    return ax


def plot_input_tube(m: Model, backoff, center, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    if not backoff.shape[0] == m.nu:
        backoff = backoff.T
    if not center.shape[0] == m.nu:
        center = center.T

    time = np.arange(0, center.shape[1]) * m.dt

    colors = plt.cm.viridis(np.linspace(0, 1, m.nu + 2))
    for i in range(m.nu):
        lower_bound = center[i] - backoff[i]
        upper_bound = center[i] + backoff[i]
        ax.fill_between(time, lower_bound, upper_bound, color=colors[i + 1], alpha=0.5, label='Bounds')
    return ax