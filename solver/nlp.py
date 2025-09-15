import casadi as ca
import numpy as np

from solver.ocp import OCP
from typing import Optional


class NLP(OCP):
    def __init__(self, N, Q, R, m, Qf):
        super().__init__(N, Q, R, m, Qf)
        self.nlp_solver_name = 'ipopt'
        self.solution = {}
        self.solver_params = {}
        self.solver_nominal = None
        # Solver options
        self.opts_ipopt = {
            'ipopt.print_level': 0,  # Minimal output
            'print_time': False,  # Disable timing info
            'ipopt.sb': 'yes'  # Silent barrier
        }
        # self.opts_ipopt = {
        #     'ipopt.print_level': 5,  # Minimal output
        #     'print_time': False,  # Disable timing info
        #     'ipopt.sb': 'no'  # Silent barrier
        # }
        self.verbose = False

        self.initialize_solver()

    def solve(self, x0, x_guess: Optional[np.ndarray] = None, u_guess: Optional[np.ndarray] = None):
        """
        Solve the nominal trajectory NLP with optional initial guess.
        Args:
          x0: np.ndarray of shape (nx,) initial state parameter.
          x_guess: optional state trajectory guess of shape (nx, N+1).
          u_guess: optional input trajectory guess of shape (nu, N).
        Precedence: if y0 is provided, it is used as-is; else if x_guess/u_guess provided, they are packed; otherwise the internal stored y0 is used.
        """
        # Update initial guess if provided
        if x_guess is not None or u_guess is not None:
            self.set_initial_guess(x_guess=x_guess, u_guess=u_guess)

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

        except Exception as e:
            self.solution['success'] = False
            return None

        return self.solution

    def set_initial_guess(self, x_guess: Optional[np.ndarray] = None, u_guess: Optional[np.ndarray] = None):
        """
        Set the initial guess for the NLP decision vector in stage-wise order.
        Accepts X (nx,N+1) and U (nu,N) and packs via pack_primal_nominal.
        """
        nx, nu, N = self.m.nx, self.m.nu, self.N

        # If separate guesses provided, validate or zero-fill
        if x_guess is None and u_guess is None:
            return  # keep existing stored y0

        if x_guess is None:
            x_guess = np.zeros((nx, N + 1))
        if u_guess is None:
            u_guess = np.zeros((nu, N))

        x_guess = np.asarray(x_guess)
        u_guess = np.asarray(u_guess)
        assert x_guess.shape == (nx, N + 1), f"x_guess must be {(nx, N+1)}, got {x_guess.shape}"
        assert u_guess.shape == (nu, N), f"u_guess must be {(nu, N)}, got {u_guess.shape}"

        y0_vec = self._pack_initial_guess(x_guess, u_guess).reshape((-1, 1))
        self.solver_params['y0'] = y0_vec


    def _pack_initial_guess(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        # pack (X,U) into a single vector y NOT in stage-wise order:
        x_vec = X.flatten(order='F')  # column-major flatten
        u_vec = U.flatten(order='F')
        return np.concatenate([x_vec, u_vec], axis=0)

    def _check_solver_status(self, sol):
        """
        Check the solver status and return True if successful, False otherwise.
        """
        try:
            stats = self.solver_nominal.stats()  # type: ignore[attr-defined]

            # Check if solution exists and solver converged
            if 'success' in stats:
                return bool(stats['success'])
            elif 'return_status' in stats:
                status = stats['return_status']
                # IPOPT successful statuses
                return status in ['Solve_Succeeded', 'Solved_To_Acceptable_Level']
            
            # Fallback: check if we have a valid solution
            fval = float(sol['f']) if sol.get('f', None) is not None else np.nan
            return sol.get('x', None) is not None and not np.isnan(fval)
        except Exception:
            # If we can't get stats, assume failure
            return False

    def _post_process_solution(self,sol):
        sol_vec = np.array(sol['x'])
        primal_x, primal_u, primal_vec = self._unpack_y(sol_vec)

        self.solution['primal_vec'] = primal_vec
        self.solution['primal_x'] = primal_x
        self.solution['primal_u'] = primal_u
        self.solution['dual_vec'] = np.array(sol['lam_g'])
        self.solution['cost'] = sol['f']

    def _unpack_y(self, y):
        """
        Given the stacked vector y = [ Z[:,0]; V[:,0];  … ; Z[:,N-1]; V[:,N-1]; Z[:,N] ],
        recover Z of shape (nx, N+1) and V of shape (nu, N).

        y may be a CasADi DM or a 1‑D numpy array/flat list.
        """
        # make sure it’s a flat numpy array
        nx = self.m.nx
        nu = self.m.nu
        N = self.N

        y_arr = np.array(y).flatten()

        sol_vec_x = y_arr[:nx * (N + 1)]
        sol_vec_u = y_arr[nx * (N + 1):]

        primal_x = sol_vec_x.reshape(nx, -1, order='F')
        primal_u = sol_vec_u.reshape(nu, -1, order='F')

        elems = []
        for i in range(N):
            elems.append(primal_x[:, i])
            elems.append(primal_u[:, i])
        elems.append(primal_x[:, N])  # final state

        primal_vec =  ca.vertcat(*elems)

        return primal_x, primal_u, primal_vec

    def initialize_solver(self):
        """
        This method initializes the nlp solver for nominal trajectory optimization
        :return:
        """
        N = self.N
        m = self.m

        G = self.m.G
        Gf = self.m.Gf
        g = self.m.g
        gf = self.m.gf

        Z = ca.MX.sym('state', self.m.nx, self.N + 1)
        V = ca.MX.sym('input', self.m.nu, self.N)
        p = ca.MX.sym('p', self.m.nx)

        g_eq = ca.vertcat()
        g_ineq = ca.vertcat()

        for i in range(N):  # Python uses 0-based indexing
            g_eq = ca.vertcat(g_eq, Z[:, i + 1] - m.ddyn(Z[:, i], V[:, i],self.m.dt))

        g_eq = ca.vertcat(g_eq, Z[:, 0] - p)

        for i in range(N):
            ineq = G @ ca.vertcat(Z[:, i], V[:, i]) - g
            g_ineq = ca.vertcat(g_ineq, ineq)

        ineq = Gf @ Z[:, -1] - gf
        g_ineq = ca.vertcat(g_ineq, ineq)

        f = Z[:, -1].T @ self.Qf @ Z[:, -1]
        for i in range(N):
            f += Z[:, i].T @ self.Q @ Z[:, i] + V[:, i].T @ self.R @ V[:, i]

        n_ineq = g_ineq.shape[0]
        lbg_ineq = [-ca.inf] * n_ineq
        ubg_ineq = np.zeros(n_ineq)

        n_eq = g_eq.shape[0]
        lbg_eq = np.zeros(n_eq)
        ubg_eq = np.zeros(n_eq)

        g = ca.vertcat(g_eq, g_ineq)
        y = ca.vertcat(ca.reshape(Z, (N + 1) * self.m.nx, 1), ca.reshape(V, N * self.m.nu, 1))

        # Define the NLP problem
        nlp = {
            'x': y,  # Decision variables (stage-wise)
            'f': f,  # Objective function
            'p': p,
            'g': g
        }

        self.solver_params['lbg'] = ca.vertcat(lbg_eq, lbg_ineq)
        self.solver_params['ubg'] = ca.vertcat(ubg_eq, ubg_ineq)
        self.solver_params['y0'] = np.zeros(((self.m.nx + self.m.nu) * N + self.m.nx, 1))

        self.solver_nominal = ca.nlpsol('solver', self.nlp_solver_name, nlp, self.opts_ipopt)

