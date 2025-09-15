import numpy as np
from dyn.LTI import LTI
from dyn.LTV import LTV
from dyn.LTI_OF import LTI_OF
from dyn.LTV_OF import LTV_OF

class OCP:
    def __init__(self, N, Q, R, m, Qf, Q_reg=None, R_reg=None, Q_reg_f=None):
        self.N = N
        self.Q = Q
        self.R = R
        self.m = m
        self.xf = np.zeros((m.nx, 1))
        self.Qf = Qf
        if Q_reg is not None:
            self.Q_reg = Q_reg
        else:
            self.Q_reg = np.eye(Q.shape[0])
        if R_reg is not None:
            self.R_reg = R_reg
        else:
            self.R_reg = np.eye(R.shape[0])
        if Q_reg_f is not None:
            self.Q_reg_f = Q_reg_f
        else:
            self.Q_reg_f = np.eye(Qf.shape[0])

        self.CONV_EPS = 1e-6

        self.A_list = None
        self.B_list = None
        self.C_list = None
        self.c_offset_list = None

        self.E_list = None
        self.F_list = None

        self.g_list = None

    # -----------------------------
    # Standard stage-wise packing helpers
    # -----------------------------
    def pack_primal_nominal(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Pack (X,U) into a single vector y in stage-wise order:
        y = [x0; u0; x1; u1; ...; x_{N-1}; u_{N-1}; xN]
        Returns a 1-D numpy array of length (nx+nu)*N + nx.
        """
        nx, nu, N = self.m.nx, self.m.nu, self.N
        assert X.shape == (nx, N + 1), f"X must be {(nx, N+1)}, got {X.shape}"
        assert U.shape == (nu, N), f"U must be {(nu, N)}, got {U.shape}"
        parts: list[np.ndarray] = []
        for k in range(N):
            parts.append(np.asarray(X[:, k]).reshape(-1))
            parts.append(np.asarray(U[:, k]).reshape(-1))
        parts.append(np.asarray(X[:, N]).reshape(-1))
        return np.concatenate(parts, axis=0)

    def unpack_primal_nominal(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Unpack stage-wise y into (X,U).
        Expects y ordered as [x0; u0; ...; x_{N-1}; u_{N-1}; xN].
        Returns X with shape (nx, N+1) and U with shape (nu, N).
        """
        nx, nu, N = self.m.nx, self.m.nu, self.N
        yv = np.asarray(y).reshape(-1)
        expected = (nx + nu) * N + nx
        assert yv.size == expected, f"y must be size {expected}, got {yv.size}"
        X = np.zeros((nx, N + 1))
        U = np.zeros((nu, N))
        p = 0
        for k in range(N):
            X[:, k] = yv[p:p + nx]; p += nx
            U[:, k] = yv[p:p + nu]; p += nu
        X[:, N] = yv[p:p + nx]
        return X, U

    def initialize_list_dynamics(self):
        """
        Initialize the linear dynamics. Assume time-invariant cost and constraints
        :return:
        """
        m = self.m
        # check if m is an instance of LTI, where LTI is a class
        if isinstance(m, LTI):
            self.A_list = [m.A for _ in range(self.N)]
            self.B_list = [m.B for _ in range(self.N)]
            self.E_list = [m.E for _ in range(self.N)]
            self.E_list.insert(0, m.E)
            self.g_list = [m.g for _ in range(self.N)]
            self.g_list.append(m.gf)
            self.c_offset_list = [np.zeros((m.nx, 1)) for _ in range(self.N)]

        elif isinstance(m, LTV):
            self.A_list = m.A_list
            self.B_list = m.B_list
            self.E_list = m.E_list
            self.g_list = m.g_list
            #todo: add here c_offset_list if needed
        else:
            raise ValueError('Model type not supported')

    #todo: not consistent: A, B should only be for linear dynamics

    @staticmethod
    def riccati_step(A, B, Cx, Cu, Sk):
        x = B.T @ Sk
        y = A.T @ Sk
        K = -np.linalg.solve(Cu + x @ B, x @ A)
        S = Cx + y @ A + y @ B @ K
        return K, S

    @staticmethod
    def riccati_step_cholesky(A, B, Cx, Cu, Sk):
        x = B.T @ Sk
        y = A.T @ Sk
        L = np.linalg.cholesky(Cu + x @ B)
        M = np.linalg.solve(L, x @ A)
        K = -np.linalg.solve(L.T, M)
        S = Cx + y @ A + y @ B @ K
        return K, S
