import numpy as np


class Model:
    def __init__(self):
        self.dt = None
        self.nx = None
        self.nu = None
        self.nw = None
        self.ni = None
        self.ni_f = None
        self.ode = None
        self.discretization_method = 'rk4'  # Default to RK4, can be set to 'euler'

    def ddyn(self, x, u, h=.05):
        ode = self.ode
        self.dt = h

        # Default to RK4 if discretization_method is not set
        discretization_method = getattr(self, 'discretization_method', 'rk4')

        if discretization_method == 'euler':
            # Euler integration: x_k+1 = x_k + h * f(x_k, u_k)
            xdot = ode(x, u)
            x_p = x + h * xdot
        else:
            # Default RK4 integration
            k_1 = ode(x, u)
            k_2 = ode(x + 0.5 * h * k_1, u)
            k_3 = ode(x + 0.5 * h * k_2, u)
            k_4 = ode(x + h * k_3, u)
            x_p = x + (1 / 6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4) * h
        
        return x_p

    def remove_constraints(self):
        # todo add check on constraint existence
        m = self

        # remove constraints
        m.G = np.zeros((0, m.nx + m.nu))
        m.g = np.zeros((0, 1))
        m.Gf = np.zeros((0, m.nx))
        m.gf = np.zeros((0, 1))
        m.ni = 0
        m.ni_f = 0
