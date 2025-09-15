from dyn.model import Model
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt


class Pendulum(Model):
    def __init__(self):
        self.nx = 4
        self.nu = 1
        self.dt = 0.05

        self.G = ca.vertcat(np.eye(5), -np.eye(5))
        x_max = np.array([10, 10, 10, 10])
        u_max = np.array([5])

        self.g = np.concatenate((x_max, u_max, x_max, u_max))
        self.ni = 10
        self.Gf = ca.vertcat(np.eye(4), -np.eye(4))
        self.gf = np.concatenate((x_max, x_max))
        self.ni_f = 8

        self.E = 0.1 * np.eye(4)
        self.nw = 4

    def ode(self, X, u):
        # taken from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9849119
        x, x_dot, theta, theta_dot = ca.vertsplit(X)

        # Constants for the cart-pole system
        m1 = 1  # mass of the cart
        m2 = 0.1  # mass of the pole
        l = 0.5  # length of the pole
        g = 9.81  # acceleration due to gravity

        # Equations of motion
        x_ddot = (u + m2 * l * theta_dot ** 2 * ca.sin(theta) - m2 * g * ca.sin(theta) * ca.cos(theta)) / (
                m1 + m2 * (1 - ca.cos(theta) ** 2))
        theta_ddot = (-u * ca.cos(theta) - m2 * l * theta_dot ** 2 * ca.sin(theta) * ca.cos(theta) + (
                m1 + m2) * g * ca.sin(
            theta)) / (l * (m1 + m2 * (1 - ca.cos(theta) ** 2)))
        Xd = ca.vertcat(x_dot, x_ddot, theta_dot, theta_ddot)

        return Xd

    def plot_nominal_trajectory(self, X, time = None, ax=None):
        """
        :param X: nominal trajectory
        :return: plot the nominal trajectory
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        x_max = self.g[0]
        # assume all constraints are the same

        # plot horizontal line
        ax.axhline(y=-x_max, color='k')
        ax.axhline(y=x_max, color='k')
        # time vector
        if time is None:
            # if time is not provided, create a time vector based on the shape of X
            time = np.arange(0, X.shape[1]) * self.dt

        nx = 4
        colors = plt.cm.viridis(np.linspace(0, 1, nx + 2))
        for i in range(nx):
            # plot the nominal trajectory
            ax.plot(time, X[i, :], color=colors[i + 1])

        return ax

    def plot_input_nominal_trajectory(self, U, time = None, ax=None):
        """
        :param X: nominal trajectory
        :return: plot the nominal trajectory
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        u_max = self.g[4]

        ax.axhline(y=-u_max, color='k')
        ax.axhline(y=u_max, color='k')
        U = U.reshape(-1, 1)
        if time is None:
            # if time is not provided, create a time vector based on the shape of U
            time = np.arange(0, U.shape[0]) * self.dt
        color = plt.cm.viridis(np.linspace(0, 1, 2))
        # plot the nominal trajectory
        ax.plot(time, U, color=color[0])

        return ax

    def plot_tube(self, backoff, center, time=None, ax=None):
        """
        :param backoff: backoff matrix (shape: (nx, N+1) or (N+1, nx))
        :param center: center matrix (shape: (nx, N+1) or (N+1, nx))
        :param time: time vector (optional, if not provided, it will be created based on the shape of center)
        :param ax: matplotlib axis to plot on (optional, if not provided, a new figure will be created)
        :return: matplotlib axis with the tube plot
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # transpose the matrices if they are not in the right shape: (nx, N+1)
        if not backoff.shape[0] == 4:
            backoff = backoff.T

        if not center.shape[0] == 4:
            center = center.T

        if time is None:
            # if time is not provided, create a time vector based on the shape of center
            time = np.arange(0, center.shape[1]) * self.dt

        nx = 4
        colors = plt.cm.viridis(np.linspace(0, 1, nx + 2))
        margin = 0.000001
        for i in range(nx):
            lower_bound = center[i] - backoff[i] + margin
            upper_bound = center[i] + backoff[i] - margin
            ax.fill_between(time, lower_bound, upper_bound, color=colors[i + 1], alpha=0.5, label='Bounds')

        return ax

    def plot_input_tube(self, backoff, center, time=None, ax=None):

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        backoff = backoff.reshape(-1)
        center = center.reshape(-1)
        if time is None:
            # if time is not provided, create a time vector based on the shape of center
            time = np.arange(0, center.shape[0]) * self.dt

        margin = 0.001
        lower_bound = center - backoff + margin
        upper_bound = center + backoff - margin
        color = plt.cm.viridis(np.linspace(0, 1, 2))
        ax.fill_between(time, lower_bound, upper_bound, alpha=0.5, label='Bounds', color=color[0])

        return ax

    def replace_constraints(self, x_max, x_min, u_max, u_min, x_max_f, x_min_f):
        self.g = np.hstack((x_max, u_max, -x_min, -u_min))
        self.gf = np.hstack((x_max_f, -x_min_f))
