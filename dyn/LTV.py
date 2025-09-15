from dyn.model import Model
import numpy as np


class LTV(Model):
    def __init__(self):
        super().__init__()
        self.N = 0
        self.A_list = []
        self.B_list = []
        self.E_list = []
        self.g_list = []
        self.G = None
        self.Gf = None
        self.gf = None

    def __init__(self, m, N):
        # m is a nonlinear model
        self.N = N
        self.nx = m.nx
        self.nu = m.nu
        self.nw = m.nw
        self.G = m.G
        self.ni = m.ni
        self.Gf = m.Gf
        self.gf = m.gf
        self.ni_f = m.ni_f
        self.A_list = [np.ones((self.nx, self.nx)) for _ in range(N)]
        self.B_list = [np.ones((self.nx, self.nu)) for _ in range(N)]
        self.E_list = [np.ones((self.nx, self.nw)) for _ in range(N + 1)]
        self.g_list = [np.ones(self.ni) for _ in range(N)]
        self.g_list.append(np.ones(self.ni_f))

    def ddyn(self, x, u, k):
        return self.A_list[k] @ x + self.B_list[k] @ u

    def update_model(self, new_list_A, new_list_B, new_list_E, new_list_g):
        """
        Initialize the LTV dynamics.
        :return:
        """
        self.A_list = new_list_A
        self.B_list = new_list_B
        self.E_list = new_list_E
        self.g_list = new_list_g

    def update_constraints(self, G, g, Gf, gf):
        self.G = G
        if not isinstance(g, list):
            self.g_list = [g for _ in range(self.N)]
        else:
            self.g_list = g
        self.g_list.append(gf)
        self.Gf = Gf
        self.gf = gf

        self.assign_dimensions()

    def assign_dimensions(self):
        self.nx = self.A_list[0].shape[0]
        self.nu = self.B_list[0].shape[1]
        self.nw = self.E_list[0].shape[1]

        self.ni = self.G.shape[0]
        self.ni_f = self.Gf.shape[0]
