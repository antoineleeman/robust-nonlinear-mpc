from dyn.model import Model

from dyn.LTV import LTV
import numpy as np


class LTV_OF(LTV):
    def __init__(self):
        super().__init__()
        self.C_list = []
        self.F_list = []
        self.nv = None
        self.ny = None

    def __init__(self, m, N):
        super().__init__(m, N)
        ny = m.ny
        nv = m.nv
        self.nv = nv
        self.ny = ny
        self.C_list = [np.zeros((self.ny, self.nx)) for _ in range(N)]
        self.F_list = [np.zeros((self.ny, self.nv)) for _ in range(N)]

    def update_model(self, new_list_A, new_list_B, new_list_E, new_list_g, new_list_C, new_list_F):
        super().update_model(new_list_A, new_list_B, new_list_E, new_list_g)
        self.C_list = new_list_C
        self.F_list = new_list_F

    def measurement(self, x, k):
        return self.C_list[k] @ x

    def assign_dimensions(self):
        super().assign_dimensions()
        self.ny = self.C_list[0].shape[0]
        self.nv = self.F_list[0].shape[1]
