import casadi as ca
import numpy as np

from solver.ocp import OCP
from typing import Optional, Tuple


class NLPSoftConstraints(OCP):
    """
    NLP with soft inequality constraints:
      Stage:   G @ [x_k; u_k] - g <= gamma_k,   gamma_k >= 0
      Terminal: Gf @ x_N - gf <= gamma_f,       gamma_f >= 0
    Penalize slacks with a large weight rho_soft in the objective.
    """

    def __init__(self, N, Q, R, m, Qf, rho_soft: float = 1e6, rho_soft_l1: Optional[float] = None):
        super().__init__(N, Q, R, m, Qf)
        self.nlp_solver_name = 'ipopt'
        self.solution = {}
        self.solver_params = {}
        self.solver_nominal = None
        self.rho_soft = float(rho_soft)
        # If not provided, use the same weight as rho_soft
        self.rho_soft_l1 = float(rho_soft if rho_soft_l1 is None else rho_soft_l1)

        # Solver options (quiet by default)
        self.opts_ipopt = {
            'ipopt.print_level': 0,
            'print_time': False,
            'ipopt.sb': 'yes'
        }
        self.verbose = False

        # Internal bookkeeping for slack dimensions
        self._mg: int = 0       # number of stage inequalities (rows of G)
        self._mgf: int = 0      # number of terminal inequalities (rows of Gf)

        self.initialize_solver()

    def solve(self, x0, x_guess: Optional[np.ndarray] = None, u_guess: Optional[np.ndarray] = None,
              gamma_guess: Optional[np.ndarray] = None):
        """
        Solve the soft-constrained NLP with optional initial guess.
        Args:
          x0: np.ndarray of shape (nx,) initial state parameter.
          x_guess: optional state trajectory guess of shape (nx, N+1).
          u_guess: optional input trajectory guess of shape (nu, N).
          gamma_guess: optional slack guess of shape (mg*N + mgf,).
        """
        # Update initial guess if provided
        if x_guess is not None or u_guess is not None or gamma_guess is not None:
            self.set_initial_guess(x_guess=x_guess, u_guess=u_guess, gamma_guess=gamma_guess)

        y0_local = self.solver_params['y0']
        lbg = self.solver_params['lbg']
        ubg = self.solver_params['ubg']

        try:
            sol = self.solver_nominal(x0=y0_local, p=x0, lbg=lbg, ubg=ubg)  # type: ignore[operator]

            # Check solver status
            solver_success = self._check_solver_status(sol)

            if solver_success:
                self.solution['success'] = True
                self._post_process_solution(sol)
            else:
                self.solution['success'] = False

        except Exception:
            self.solution['success'] = False
            return None

        return self.solution

    def set_initial_guess(self, x_guess: Optional[np.ndarray] = None, u_guess: Optional[np.ndarray] = None,
                          gamma_guess: Optional[np.ndarray] = None):
        """
        Set the initial guess for the NLP decision vector in stage-wise order.
        Accepts X (nx,N+1), U (nu,N), and Gamma (mg*N+mgf,) packed via _pack_initial_guess.
        """
        nx, nu, N = self.m.nx, self.m.nu, self.N

        # If nothing provided, keep existing stored y0
        if x_guess is None and u_guess is None and gamma_guess is None:
            return

        # Default guesses
        if x_guess is None:
            x_guess = np.zeros((nx, N + 1))
        if u_guess is None:
            u_guess = np.zeros((nu, N))
        if gamma_guess is None:
            gamma_guess = np.zeros((self._mg * N + self._mgf,))

        x_guess = np.asarray(x_guess)
        u_guess = np.asarray(u_guess)
        gamma_guess = np.asarray(gamma_guess).flatten()

        assert x_guess.shape == (nx, N + 1), f"x_guess must be {(nx, N+1)}, got {x_guess.shape}"
        assert u_guess.shape == (nu, N), f"u_guess must be {(nu, N)}, got {u_guess.shape}"
        assert gamma_guess.shape == (self._mg * N + self._mgf,), \
            f"gamma_guess must be {(self._mg * N + self._mgf,)}, got {gamma_guess.shape}"

        y0_vec = self._pack_initial_guess(x_guess, u_guess, gamma_guess).reshape((-1, 1))
        self.solver_params['y0'] = y0_vec

    def _pack_initial_guess(self, X: np.ndarray, U: np.ndarray, Gamma: np.ndarray) -> np.ndarray:
        # pack (X,U,Gamma) into a single vector y in the same order as decision vars
        x_vec = X.flatten(order='F')
        u_vec = U.flatten(order='F')
        g_vec = Gamma.flatten(order='F')
        return np.hstack([x_vec, u_vec, g_vec])

    def _check_solver_status(self, sol):
        """
        Check the solver status and return True if successful, False otherwise.
        """
        try:
            stats = self.solver_nominal.stats()  # type: ignore[attr-defined]
            if 'success' in stats:
                return bool(stats['success'])
            elif 'return_status' in stats:
                status = stats['return_status']
                return status in ['Solve_Succeeded', 'Solved_To_Acceptable_Level']
            fval = float(sol['f']) if sol.get('f', None) is not None else np.nan
            return sol.get('x', None) is not None and not np.isnan(fval)
        except Exception:
            return False

    def _post_process_solution(self, sol):
        sol_vec = np.array(sol['x']).flatten()
        primal_x, primal_u, primal_gamma, primal_vec = self._unpack_y(sol_vec)

        self.solution['primal_vec'] = primal_vec
        self.solution['primal_x'] = primal_x
        self.solution['primal_u'] = primal_u
        self.solution['primal_gamma'] = primal_gamma
        self.solution['dual_vec'] = np.array(sol['lam_g'])
        self.solution['cost'] = sol['f']

    def _unpack_y(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, ca.MX]:
        """
        Given y = [ vec(Z); vec(V); Gamma_all ], recover:
          Z of shape (nx, N+1), V of shape (nu, N), Gamma_all of shape (mg*N+mgf,)
        and the stage-wise primal vector [Z0; V0; ... ; Z_{N-1}; V_{N-1}; Z_N; gamma_all].
        """
        nx, nu, N = self.m.nx, self.m.nu, self.N
        mg, mgf = self._mg, self._mgf

        y_arr = np.array(y).flatten()

        n_x = nx * (N + 1)
        n_u = nu * N
        n_g = mg * N + mgf

        sol_vec_x = y_arr[:n_x]
        sol_vec_u = y_arr[n_x:n_x + n_u]
        sol_vec_gamma = y_arr[n_x + n_u:n_x + n_u + n_g]

        primal_x = sol_vec_x.reshape(nx, N + 1, order='F')
        primal_u = sol_vec_u.reshape(nu, N, order='F')
        primal_gamma = sol_vec_gamma

        # Build stage-wise vector for convenience (without gammas inside stages to avoid shape mismatch)
        elems = []
        for i in range(N):
            elems.append(primal_x[:, i])
            elems.append(primal_u[:, i])
        elems.append(primal_x[:, N])  # final state
        primal_vec = ca.vertcat(*elems)

        return primal_x, primal_u, primal_gamma, primal_vec

    def initialize_solver(self):
        """
        Initialize the soft-constraint NLP.
        """
        N = self.N
        m = self.m

        G = self.m.G
        Gf = self.m.Gf
        g = self.m.g
        gf = self.m.gf

        # Determine inequality sizes (allow 0-sized gracefully)
        self._mg = int(G.shape[0]) if hasattr(G, 'shape') else 0
        self._mgf = int(Gf.shape[0]) if hasattr(Gf, 'shape') else 0

        Z = ca.MX.sym('state', self.m.nx, self.N + 1)
        V = ca.MX.sym('input', self.m.nu, self.N)
        p = ca.MX.sym('p', self.m.nx)

        # Slack variables
        Gamma_stg = ca.MX.sym('gamma', self._mg, self.N) if self._mg > 0 else ca.MX(0, self.N)
        Gamma_fin = ca.MX.sym('gammaf', self._mgf, 1) if self._mgf > 0 else ca.MX(0, 1)

        # Constraints
        g_eq = ca.vertcat()
        g_ineq_soft = ca.vertcat()       # Gz - g - gamma <= 0
        g_gamma_nonneg = ca.vertcat()    # -gamma <= 0

        # Dynamics
        for i in range(N):
            g_eq = ca.vertcat(g_eq, Z[:, i + 1] - m.ddyn(Z[:, i], V[:, i], self.m.dt))
        # Initial state
        g_eq = ca.vertcat(g_eq, Z[:, 0] - p)

        # Stage inequalities with slacks
        if self._mg > 0:
            for i in range(N):
                zu = ca.vertcat(Z[:, i], V[:, i])
                ineq = G @ zu - g - Gamma_stg[:, i]
                g_ineq_soft = ca.vertcat(g_ineq_soft, ineq)
                # gamma >= 0 -> -gamma <= 0
                g_gamma_nonneg = ca.vertcat(g_gamma_nonneg, -Gamma_stg[:, i])

        # Terminal inequality with slack
        if self._mgf > 0:
            ineq_f = Gf @ Z[:, -1] - gf - Gamma_fin
            g_ineq_soft = ca.vertcat(g_ineq_soft, ineq_f)
            g_gamma_nonneg = ca.vertcat(g_gamma_nonneg, -Gamma_fin)

        # Objective: nominal quadratic + large quadratic and L1 penalty on slacks
        f = Z[:, -1].T @ self.Qf @ Z[:, -1]
        for i in range(N):
            f += Z[:, i].T @ self.Q @ Z[:, i] + V[:, i].T @ self.R @ V[:, i]
        if self._mg > 0:
            # Quadratic penalty
            f += self.rho_soft * ca.sumsqr(Gamma_stg)
            # L1 penalty (slacks are constrained nonnegative, so sum == L1 norm)
            f += self.rho_soft_l1 * ca.sum1(ca.sum2(Gamma_stg))
        if self._mgf > 0:
            f += self.rho_soft * ca.sumsqr(Gamma_fin)

        # Bounds for constraints
        n_ineq_soft = g_ineq_soft.shape[0]
        lbg_ineq_soft = [-ca.inf] * n_ineq_soft
        ubg_ineq_soft = np.zeros(n_ineq_soft)

        n_gamma_nonneg = g_gamma_nonneg.shape[0]
        lbg_gamma_nonneg = [-ca.inf] * n_gamma_nonneg
        ubg_gamma_nonneg = np.zeros(n_gamma_nonneg)

        n_eq = g_eq.shape[0]
        lbg_eq = np.zeros(n_eq)
        ubg_eq = np.zeros(n_eq)

        # Full constraint vector
        g_all = ca.vertcat(g_eq, g_ineq_soft, g_gamma_nonneg)

        # Decision vector: stage-wise Z, V, then all gammas
        y = ca.vertcat(
            ca.reshape(Z, (N + 1) * self.m.nx, 1),
            ca.reshape(V, N * self.m.nu, 1),
            ca.reshape(Gamma_stg, self._mg * N, 1) if self._mg > 0 else ca.MX(),
            ca.reshape(Gamma_fin, self._mgf, 1) if self._mgf > 0 else ca.MX()
        )

        # Define the NLP problem
        nlp = {
            'x': y,
            'f': f,
            'p': p,
            'g': g_all
        }

        # Store bounds and initial guess
        self.solver_params['lbg'] = ca.vertcat(lbg_eq, lbg_ineq_soft, lbg_gamma_nonneg)
        self.solver_params['ubg'] = ca.vertcat(ubg_eq, ubg_ineq_soft, ubg_gamma_nonneg)
        n_y = (self.m.nx + self.m.nu) * N + self.m.nx + self._mg * N + self._mgf
        self.solver_params['y0'] = np.zeros((n_y, 1))

        # Create solver
        self.solver_nominal = ca.nlpsol('solver_soft', self.nlp_solver_name, nlp, self.opts_ipopt)
