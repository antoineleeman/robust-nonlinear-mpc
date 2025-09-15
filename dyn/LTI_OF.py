from dyn.model import Model
import numpy as np

from dyn.LTI import LTI


class LTI_OF(LTI):
    def __init__(self):
        self.nv = None
        self.ny = None
        self.C = None  # measurement matrix y = Cx
        self.F = None  # measurement noise y = Cx + Fw
        super().__init__()

    def measurement(self, x):
        return self.C @ x

    def assign_dimensions(self):
        super().assign_dimensions()
        self.ny = self.C.shape[0]
        self.nv = self.F.shape[1]
