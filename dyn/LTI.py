from dyn.model import Model
import numpy as np

class LTI(Model):
    def __init__(self):
        super().__init__()
        self.A = None
        self.B = None
        self.E = None
        self.Kf = None

    def ddyn(self, x, u, k):
        return self.A @ x + self.B @ u

    def assign_dimensions(self):
        self.nx = self.A.shape[0]
        self.nu = self.B.shape[1]
        self.nw = self.E.shape[1]

        self.ni = self.G.shape[0]
        self.ni_f = self.Gf.shape[0]

    def build_G_constraints(self, Hx, hx, Hu, hu):
        self.G = np.block([
            [Hx, np.zeros((Hx.shape[0], Hu.shape[1]))],
            [np.zeros((Hu.shape[0], Hx.shape[1])), Hu]
        ])

        self.g = np.concatenate([hx, hu])

    def build_controller(self, Q, R):
        K, S, E = control.lqr(self.A, self.B, Q, R)
        self.Kf = K
        return K
