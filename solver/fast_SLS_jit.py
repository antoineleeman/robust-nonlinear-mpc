import numpy as np
from numba import njit, prange
import casadi as ca

from solver.ocp import OCP
from solver.qp_jit import QP
from util.SLS import SLS

from pathlib import Path
import sys, glob, importlib.util
import time
from prettytable import PrettyTable, NONE, HEADER

# compute absolute path: <repo_root>/build/osqp_fast
# this assumes your fast_sls file is in <repo_root>/solver/...
codegen_dir = (Path(__file__).resolve().parents[1] / "build" / "osqp_fast")

# 1) try adding the folder to sys.path
if codegen_dir.is_dir():
    sys.path.insert(0, str(codegen_dir))

try:
    import osqp_generated
except ModuleNotFoundError:
    # 2) fallback: load the compiled extension directly
    matches = (
        glob.glob(str(codegen_dir / "osqp_generated*.so"))
        + glob.glob(str(codegen_dir / "osqp_generated*.pyd"))
    )
    if not matches:
        raise  # extension not built / wrong folder
    spec = importlib.util.spec_from_file_location("osqp_generated", matches[0])
    osqp_generated = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(osqp_generated)

PRIMAL_INFEASIBILITY = 0.0
fastmath = False
# -----------------------------
# JIT-accelerated kernels
# -----------------------------

@njit(fastmath=fastmath)
def riccati_step_njit(A, B, Cx, Cu, Sk):
    # Shapes: A(nx,nx), B(nx,nu), Cx(nx,nx), Cu(nu,nu), Sk(nx,nx)
    # x = Bᵀ S_k, y = Aᵀ S_k
    x = B.T @ Sk  # (nu, nx)
    y = A.T @ Sk  # (nx, nx)

    H = Cu + x @ B  # (nu, nu)
    F = x @ A  # (nu, nx)

    # Solve H * Z = F  →  K = -Z
    Z = np.linalg.solve(H, F)  # (nu, nx)
    K = -Z

    # S = Cx + y @ (A + B K)
    S = Cx + y @ (A + B @ K)

    # Optional: symmetrize to control drift
    S = 0.5 * (S + S.T)
    return K, S


@njit(parallel=True, fastmath=fastmath)
def _backward_solve_numba(N, nx, nu,
                          A, B, G, Gf, eta, eta_f,
                          Q_reg, R_reg, Q_reg_f):
    S = np.zeros((N + 1, N + 1, nx, nx))
    K = np.zeros((N, N + 1, nu, nx))

    for jj in prange(N + 1):  # columns are independent
        C_fj = Gf.T @ (eta_f[jj].reshape(-1, 1) * Gf)
        S[N, jj] = C_fj + Q_reg_f

        for kk in range(N - 1, jj - 1, -1):
            C_kj = G.T @ (eta[kk, jj].reshape(-1, 1) * G)  # (nx+nu, nx+nu)
            C_xx = C_kj[:nx, :nx] + Q_reg
            C_uu = C_kj[nx:, nx:] + R_reg

            Kkk, Skk = riccati_step_njit(A[kk], B[kk], C_xx, C_uu, S[kk + 1, jj])
            K[kk, jj] = Kkk
            S[kk, jj] = Skk
    return S, K


@njit(cache=True, fastmath=fastmath, parallel=True)
def _propagate(A, B, E, K):
    """Forward-propagate Phi_x, Phi_u using batched matmuls.
    Shapes:
      A: (N, nx, nx)
      B: (N, nx, nu)
      E: (N+1, nx, nw)
      K: (N, N+1, nu, nx)  lower-triangular in time (K[k,j]=0 for j>k)
    Returns:
      Phi_x: (N+1, N+1, nx, nw)
      Phi_u: (N,   N+1, nu, nw)
    """
    N = A.shape[0]
    nx = A.shape[1]
    nu = B.shape[2]
    nw = E.shape[2]

    Phi_x = np.zeros((N+1, N+1, nx, nw))
    Phi_u = np.zeros((N,   N+1, nu, nw))

    # set diagonal Phi_x[j,j] = E[j]
    for j in range(N+1):
        Phi_x[j, j] = E[j]

    for kk in range(N):
        # jj loops are independent given Phi_x[kk, jj]
        for jj in prange(kk+1):
            Phi_u[kk, jj] = K[kk, jj] @ Phi_x[kk, jj]
            Acl = A[kk] + B[kk] @ K[kk, jj]
            Phi_x[kk+1, jj] = Acl @ Phi_x[kk, jj]
    return Phi_x, Phi_u


@njit(cache=True, fastmath=fastmath, parallel=True)
def _backoff_from_phi(Phi_x, Phi_u, Gx, Gu, Gf, epsilon):
    """Compute beta, beta_f, and backoffs from Phi_x/Phi_u.
    Shapes:
      Phi_x: (N+1, N+1, nx, nw)
      Phi_u: (N,   N+1, nu, nw)
      Gx: (ni, nx)
      Gu: (ni, nu)
      Gf: (ni_f, nx)
    Returns:
      beta:   (N, N, ni)
      beta_f: (N+1, ni_f)
      backoff:   (N, ni)
      backoff_f: (ni_f,)
    """
    N = Phi_u.shape[0]
    ni = Gx.shape[0]
    ni_f = Gf.shape[0]
    nw = Phi_x.shape[3]

    beta = np.zeros((N, N, ni))
    beta_f = np.zeros((N+1, ni_f))

    # stage constraints
    for kk in prange(N):
        for jj in range(kk+1):
            # Z = Gx@Phi_x + Gu@Phi_u -> (ni,nw)
            Zx = Gx @ Phi_x[kk, jj]
            Zu = Gu @ Phi_u[kk, jj]
            Z = Zx + Zu
            # row-wise 2-norm squared
            for i in range(ni):
                s = 0.0
                for w in range(nw):
                    v = Z[i, w]
                    s += v * v
                if s < epsilon:
                    s = epsilon
                beta[kk, jj, i] = s

    # terminal constraints
    for jj in prange(N+1):
        Zf = Gf @ Phi_x[N, jj]  # (ni_f,nw)
        for i in range(ni_f):
            s = 0.0
            for w in range(nw):
                v = Zf[i, w]
                s += v * v
            if s < epsilon:
                s = epsilon
            beta_f[jj, i] = s

    # sqrt + sums
    backoff = np.zeros((N, ni))
    for kk in prange(N):
        for i in range(ni):
            acc = 0.0
            for jj in range(kk+1):
                acc += np.sqrt(beta[kk, jj, i])
            backoff[kk, i] = acc

    backoff_f = np.zeros(ni_f)
    for i in prange(ni_f):
        acc = 0.0
        for jj in range(N+1):
            acc += np.sqrt(beta_f[jj, i])
        backoff_f[i] = acc

    return beta, beta_f, backoff, backoff_f


# -----------------------------
# Optimized Fast-SLS
# -----------------------------

class fast_SLS(OCP):
    """Optimized Fast-SLS:
    - NumPy batched matmuls + Numba JIT kernels
    - No per-iteration history; only final solution returned
    - Minimal allocations; no NaNs
    """

    def __init__(self, N, Q, R, m, Qf, Q_reg=None, R_reg=None, Q_reg_f=None):
        super().__init__(N, Q, R, m, Qf, Q_reg, R_reg, Q_reg_f)

        self.epsilon_backoff = 1e-10
        self.MAX_ITER = 30
        self.nominal_solver_options = {'printLevel': 'none'}
        self.verbose = True
        # configurable indentation for all console prints
        tab_indent = 25
        self._tab = "\t" * max(0, int(tab_indent))

        # RTI-like iteration cap (None or <=0 disables; >0 runs exactly that many iterations)
        self.rti_steps = None

        # working state for current iteration (trimmed)
        self.current_iteration = {
            'success': False,
            'cost_nominal': np.nan,
            'cost_tube': np.nan,
            'cost': np.nan,
            # timings (ms)
            't_qp_ms': np.nan,
            't_backward_ms': np.nan,
        }
        self.convergence_data = {}

        # Use compiled solver
        self.solver_forward = QP(
            self.N, self.Q, self.R, self.m, self.Qf,
            backend="osqp",
            codegen_module="osqp_generated"  # matches the module you just built
        )
        self.nominal_lbg = self.solver_forward.lbg

        # initialize dynamics lists from OCP base
        self.initialize_list_dynamics()

        # small buffers used across iterations
        self.initialize_backoff()
        self.initialize_solver()

    def set_rti_steps(self, steps: int | None):
        """Set RTI-like cap on iterations. None or <=0 disables; >0 runs exactly that many iterations."""
        if steps is None or steps <= 0:
            self.rti_steps = None
        else:
            self.rti_steps = int(steps)

    def update_dynamics_list(self, new_list_A, new_list_B, new_list_E=None, new_list_g=None, c_offset_list=None):
        """
        Update the linear dynamics lists and propagate changes to the forward QP solver.
        Mirrors fast_SLS.update_dynamics_list.
        """
        # update the dynamics
        self.A_list = new_list_A
        self.B_list = new_list_B

        if new_list_E is not None:
            self.E_list = new_list_E

        if new_list_g is not None:
            self.g_list = new_list_g

        # update the solver_forward object with the new dynamics
        self.solver_forward.update_dynamics(self.A_list, self.B_list, self.E_list, self.g_list)

        # optional offsets for equality (dynamics) constraints
        if c_offset_list is not None:
            self.c_offset_list = c_offset_list
            self.solver_forward.offset_constraints(np.hstack(c_offset_list))

        return

    # -----------------------------
    # High-level solve loop
    # -----------------------------
    def solve(self, x0):
        # RTI-like mode: run exactly rti_steps iterations and return current iterate (even if not converged)
        if self.rti_steps is not None and self.rti_steps > 0:
            self.initialize_backoff()
            table = self.printHeader() if self.verbose else None
            last_infeasible = False
            for i in range(self.rti_steps):
                state = self._step(x0)
                if state is False:  # infeasible forward solve
                    last_infeasible = True
                    break
                if self.verbose and table is not None:
                    self.printLine(i, table)
            # Always finish on a QP solve if we didn’t end on infeasibility
            if not last_infeasible:
                _ = self.forward_solve(x0)
            # Mark success as long as we didn’t hit infeasibility (even if not converged)
            self.current_iteration['success'] = not last_infeasible or bool(self.current_iteration.get('success', False))
            return self.post_processing_solution()

        # Default: iterate until convergence (or MAX_ITER safety cap)
        self.initialize_backoff()
        table = self.printHeader() if self.verbose else None
        for i in range(self.MAX_ITER):
            state = self._step(x0)
            if state is False:  # infeasible
                return self._finish_failure(i, infeasible=True)
            if state is True:   # converged (already preceded by a QP solve in this iteration)
                return self._finish_success(i)
            # ongoing iteration -> print line
            if self.verbose and table is not None:
                self.printLine(i, table)
        # Hit safety cap without convergence: perform one last QP solve with latest tightened bounds
        _ = self.forward_solve(x0)
        return self._finish_failure(self.MAX_ITER - 1)

    def _step(self, x0):
        if not self.forward_solve(x0):
            return False
        self.evaluate_dual_eta()
        if self.check_convergence_socp():
            self.current_iteration['success'] = True
            return True
        self.backward_solve()
        self.update_tightening()
        self.current_iteration['cost'] = (
            self.current_iteration['cost_nominal'] + self.current_iteration['cost_tube']
        )
        self.current_iteration['iteration_number'] += 1
        return None

    def _finish_success(self, i):
        if self.verbose:
            print(f"{self._tab}Fast-SLS: Converged in {i} iterations")
        return self.post_processing_solution()

    def _finish_failure(self, i, infeasible=False):
        self.current_iteration['success'] = False
        if self.verbose:
            msg = "infeasible forward solve" if infeasible else f"did not converge in {i} iters"
            print(f"{self._tab}Fast‑SLS: {msg}")
        sol = self.post_processing_solution()
        self.reset_solver_to_zeros()
        return sol

    # -----------------------------
    # Table utilities (prettytable) – mirrored from fast_SLS
    # -----------------------------
    def printHeader(self):
        fixed_width = 10
        headers = [
            "it (SLS)", "primal", "dual", "cost nom.", "cost tube", "cost",
            "t_qp [ms]", "t_bwd [ms]"
        ]
        formatted_headers = [f"{h:>{fixed_width}}" for h in headers]
        table = PrettyTable()
        table.field_names = formatted_headers
        table.hrules = HEADER
        table.border = True

        # align and width
        table.align[headers[0]] = "right"
        for h in headers[1:]:
            table.align[h] = "right"
        for h in headers:
            table.max_width[h] = fixed_width

        header_str = table.get_string(end=0)
        print("\n".join(f"{self._tab}" + line for line in header_str.splitlines()))
        table.hrules = NONE
        return table

    def printLine(self, i, table):
        fixed_width = 10
        # compute metrics defensively
        try:
            primal = float(np.max(np.abs(self.current_iteration.get('primal_vec', np.array([np.nan])))))
        except Exception:
            primal = np.nan
            # prefer eta if available; else dual_vec
        try:
            if 'eta' in self.current_iteration and self.current_iteration['eta'] is not None:
                dual_val = float(np.max(np.abs(np.nan_to_num(self.current_iteration['eta']))))
            else:
                dual_val = float(np.max(np.abs(np.nan_to_num(self.current_iteration.get('dual_vec', np.array([np.nan]))))))
        except Exception:
            dual_val = np.nan
        cost_nom = float(self.current_iteration.get('cost_nominal', np.nan))
        cost_tube = float(self.current_iteration.get('cost_tube', np.nan))
        cost_tot = float(self.current_iteration.get('cost', np.nan))
        t_qp = self.current_iteration.get('t_qp_ms', np.nan)
        t_bwd = self.current_iteration.get('t_backward_ms', np.nan)

        row = [
            f"{i:>{fixed_width}}",
            f"{primal:>{fixed_width}.2e}",
            f"{dual_val:>{fixed_width}.2e}",
            f"{cost_nom:>{fixed_width}.2e}",
            f"{cost_tube:>{fixed_width}.2e}",
            f"{cost_tot:>{fixed_width}.2e}",
            f"{t_qp:>{fixed_width}.2e}",
            f"{t_bwd:>{fixed_width}.2e}",
        ]
        table.add_row(row)
        row_str = table.get_string(start=len(table._rows) - 1, end=len(table._rows), header=False)
        print("\n".join(f"{self._tab}" + line for line in row_str.splitlines()))

    # -----------------------------
    # Minimal state init/reset
    # -----------------------------
    def initialize_solver(self):
        N = self.N
        ni = self.m.ni
        ni_f = self.m.ni_f
        self.current_iteration = {}
        self.current_iteration.update({
            'primal_vec': np.zeros(1),
            'dual_vec': np.zeros(1),
            'eta': np.zeros((N, N, ni)),
            'eta_f': np.zeros((N + 1, ni_f)),
            'iteration_number': 0,
            # reset timing fields each solve
            't_qp_ms': np.nan,
            't_backward_ms': np.nan,
        })

    def reset_solver_to_zeros(self):
        # initialize dynamics lists from OCP base
        self.initialize_list_dynamics()
        self.A_list = []
        self.B_list = []
        self.E_list = []
        self.c_offset_list = []
        self.g_list = []

        # small buffers used across iterations
        self.initialize_backoff()
        self.initialize_solver()
        self.solver_forward.reset_ubg()
        self.solver_forward.reset_lbg()
        self.solver_forward.reset_q_cost_lin()
        self.solver_forward.A_list = []
        self.solver_forward.B_list = []
        self.E_list = []
        self.initialize_backoff()

    def initialize_backoff(self):
        N = self.N
        nx, nu = self.m.nx, self.m.nu
        self.current_iteration['beta'] = np.full((N, N, self.m.ni), self.epsilon_backoff)
        self.current_iteration['beta_f'] = np.full((N + 1, self.m.ni_f), self.epsilon_backoff)
        self.current_iteration['sqrt_beta'] = np.sqrt(self.current_iteration['beta'])
        self.current_iteration['sqrt_beta_f'] = np.sqrt(self.current_iteration['beta_f'])
        self.current_iteration['backoff'] = self.current_iteration['sqrt_beta'].sum(axis=1)
        self.current_iteration['backoff_f'] = self.current_iteration['sqrt_beta_f'].sum(axis=0)
        self.current_iteration['backoff_x'] = np.zeros((N + 1, nx))
        self.current_iteration['backoff_u'] = np.zeros((N, nu))

    # -----------------------------
    # Forward/Backward passes
    # -----------------------------
    def forward_solve(self, x0):
        sol = self.solver_forward.solve(x0)
        if sol['success'] is False:
            if self.verbose:
                print(f"{self._tab}Infeasible forward solution. Try another initial condition.")
            return False
        self.current_iteration['t_qp_ms'] = sol.get('time_ms', np.nan)
        self.current_iteration['primal_vec'] = sol['primal_vec']
        self.current_iteration['primal_x'] = sol['primal_x']
        self.current_iteration['primal_u'] = sol['primal_u']
        self.current_iteration['dual_vec'] = sol['dual_vec']
        self.current_iteration['dual_mu'] = sol['dual_mu']
        self.current_iteration['dual_mu_f'] = sol['dual_mu_f']
        self.current_iteration['cost_nominal'] = sol['cost']
        return True

    def evaluate_dual_eta(self):
        N = self.N
        ni, ni_f = self.m.ni, self.m.ni_f
        beta = np.maximum(self.current_iteration['beta'], self.epsilon_backoff)
        beta_f = np.maximum(self.current_iteration['beta_f'], self.epsilon_backoff)

        eta = self.current_iteration['eta']
        for jj in range(N):
            for kk in range(jj, N):
                eta[kk, jj] = self.current_iteration['dual_mu'][:, kk] / (2.0 * np.sqrt(beta[kk, jj]))
        eta_f = self.current_iteration['eta_f']
        for jj in range(N + 1):
            eta_f[jj] = self.current_iteration['dual_mu_f'] / (2.0 * np.sqrt(beta_f[jj]))

    def backward_solve(self):
        m, N = self.m, self.N
        nx, nu = m.nx, m.nu

        # Ensure contiguous float64 arrays (Numba likes this)
        A = np.ascontiguousarray(np.stack(self.A_list).astype(np.float64))  # (N, nx, nx)
        B = np.ascontiguousarray(np.stack(self.B_list).astype(np.float64))  # (N, nx, nu)
        G = np.ascontiguousarray(m.G.full().astype(np.float64))  # (p, nx+nu)
        Gf = np.ascontiguousarray(m.Gf.full().astype(np.float64))  # (pf, nx)
        eta = np.ascontiguousarray(self.current_iteration['eta'].astype(np.float64))  # (N, N+1, p)
        eta_f = np.ascontiguousarray(self.current_iteration['eta_f'].astype(np.float64))  # (N+1, pf)
        Q_reg = np.ascontiguousarray(self.Q_reg.astype(np.float64))  # (nx, nx)
        R_reg = np.ascontiguousarray(self.R_reg.astype(np.float64))  # (nu, nu)
        Q_reg_f = np.ascontiguousarray(self.Q_reg_f.astype(np.float64))  # (nx, nx)

        t0 = time.perf_counter()
        S, K = _backward_solve_numba(N, nx, nu, A, B, G, Gf, eta, eta_f, Q_reg, R_reg, Q_reg_f)
        t1 = time.perf_counter()
        self.current_iteration['t_backward_ms'] = (t1 - t0) * 1e3
        if np.isnan((t1 - t0) * 1e3):
            print("x is NaN")

        self.current_iteration['S'] = S
        self.current_iteration['K'] = K

    # -----------------------------
    # Tightening (vectorized + JIT)
    # -----------------------------
    def update_tightening(self):
        N = self.N
        nx, nu, nw = self.m.nx, self.m.nu, self.m.nw
        ni, ni_f = self.m.ni, self.m.ni_f

        G = self.m.G
        Gf = self.m.Gf
        gf = self.m.gf

        A = np.ascontiguousarray(self.A_list)
        B = np.ascontiguousarray(self.B_list)
        E = np.ascontiguousarray(self.E_list)
        K = np.ascontiguousarray(self.current_iteration['K'])

        # split G = [Gx | Gu]
        Gx = np.ascontiguousarray(G[:, :nx])
        Gu = np.ascontiguousarray(G[:, nx:])
        Gf = np.ascontiguousarray(Gf)

        # forward propagate (Numba)
        Phi_x, Phi_u = _propagate(A, B, E, K)

        # cost of tube (use SLS utilities on dense matrices)
        self.current_iteration['cost_tube'] = SLS.eval_cost(
            N, self.Q_reg, self.R_reg, self.Q_reg_f,
            SLS.convert_tensor_to_matrix(Phi_x),
            SLS.convert_tensor_to_matrix(Phi_u)
        )

        # compute backoffs (Numba)
        beta, beta_f, backoff, backoff_f = _backoff_from_phi(
            Phi_x, Phi_u, Gx, Gu, Gf, float(self.epsilon_backoff)
        )

        self.current_iteration['beta'] = beta
        self.current_iteration['beta_f'] = beta_f
        self.current_iteration['backoff'] = backoff
        self.current_iteration['backoff_f'] = backoff_f

        # backoff_x/backoff_u (assume symmetric constraints as before)
        self.current_iteration['backoff_x'] = np.vstack((backoff[:, :nx], backoff_f[:nx]))
        self.current_iteration['backoff_u'] = backoff[:, nx:nx+nu]

        # update inequality UB with new backoff
        g = self.g_list
        absolute_backoff_table = np.squeeze(g[:-1]) - backoff  # (N, ni)

        c_offset_mat = np.hstack(self.c_offset_list) - PRIMAL_INFEASIBILITY

        new_ubg_table = np.vstack([-c_offset_mat, absolute_backoff_table.T])
        new_ubg_without_terminal = np.reshape(new_ubg_table, (N * (ni + nx)), order='F')
        new_ubg = np.concatenate([new_ubg_without_terminal, gf - backoff_f])
        self.solver_forward.update_ubg(new_ubg)

        return
    # -----------------------------
    # Misc
    # -----------------------------
    def update_linear_cost(self, q_cost_lin):
        self.solver_forward.update_q_cost_lin(q_cost_lin)

    def add_linear_cost(self, q_cost_lin):
        self.solver_forward.add_q_cost_lin(q_cost_lin)

    def check_convergence_socp(self):
        prev_p = getattr(self, '_prev_primal_vec', None)
        prev_eta = getattr(self, '_prev_eta', None)

        cur_p = self.current_iteration['primal_vec']
        cur_eta = self.current_iteration['eta']

        # first iteration guard: store and continue
        if prev_p is None or prev_eta is None:
            self._prev_primal_vec = cur_p.copy()
            self._prev_eta = cur_eta.copy()
            return False

        ok_p = (np.max(np.abs(cur_p - prev_p)) <= 1e-3)
        ok_d = (np.max(np.abs(cur_eta - prev_eta)) <= 1e-3)

        self._prev_primal_vec = cur_p.copy()
        self._prev_eta = cur_eta.copy()

        return ok_p

    def post_processing_solution(self):
        # Convert K/Phi to block matrices only once at the end
        if 'K' in self.current_iteration:
            K_mat = SLS.convert_tensor_to_matrix(self.current_iteration['K'])
            K_mat = np.nan_to_num(K_mat)
            self.current_iteration['K_mat'] = K_mat

        if 'Phi_u' in self.current_iteration and 'Phi_x' in self.current_iteration:
            Phi_u = np.nan_to_num(self.current_iteration['Phi_u'])
            Phi_x = np.nan_to_num(self.current_iteration['Phi_x'])
            self.current_iteration['Phi_u_mat'] = SLS.convert_tensor_to_matrix(Phi_u)
            self.current_iteration['Phi_x_mat'] = SLS.convert_tensor_to_matrix(Phi_x)

        out = {
            'iteration_number': self.current_iteration['iteration_number'],
            'success': self.current_iteration['success'],
            'cost_nominal': self.current_iteration['cost_nominal'],
            'cost_tube': np.nan,
            'cost': np.nan,
            'primal_x': self.current_iteration.get('primal_x', None),
            'primal_u': self.current_iteration.get('primal_u', None),
            'primal_vec': self.current_iteration.get('primal_vec', None),
            'dual_vec': self.current_iteration.get('dual_vec', None),
            'dual_mu': self.current_iteration.get('dual_mu', None),
            'dual_mu_f': self.current_iteration.get('dual_mu_f', None),
            'eta': self.current_iteration.get('eta', None),
            'eta_f': self.current_iteration.get('eta_f', None),
            'K': self.current_iteration.get('K', None),
            'K_mat': self.current_iteration.get('K_mat', None),
            'Phi_x': self.current_iteration.get('Phi_x', None),
            'Phi_u': self.current_iteration.get('Phi_u', None),
            'Phi_x_mat': self.current_iteration.get('Phi_x_mat', None),
            'Phi_u_mat': self.current_iteration.get('Phi_u_mat', None),
            'beta': self.current_iteration.get('beta', None),
            'beta_f': self.current_iteration.get('beta_f', None),
            'backoff': self.current_iteration.get('backoff', None),
            'backoff_f': self.current_iteration.get('backoff_f', None),
            'backoff_x': self.current_iteration.get('backoff_x', None),
            'backoff_u': self.current_iteration.get('backoff_u', None),
            't_qp_ms': self.current_iteration.get('t_qp_ms', None),
            't_backward_ms': self.current_iteration.get('t_backward_ms', None),
        }
        if np.isnan(self.current_iteration.get('t_backward_ms', None)):
            print('[Pendulum] t_backward_ms is NaN')
        return out

    # Kept for API compatibility (no-op / simplified)
    def initialize_tightening_fun(self, A_fun, B_fun, E_fun):
        G = ca.DM(self.m.G)
        Gf = ca.DM(self.m.Gf)
        nu = self.m.nu
        nw = self.m.nw
        nx = self.m.nx
        ni = self.m.ni
        N = self.N

        Z = ca.SX.sym('Z', nx, (N + 1))
        V = ca.SX.sym('V', nu,  N)

        Phi_u = [[ca.SX.sym(f"Phi_u_{k}_{j}", nu, nw) for j in range(N + 1)] for k in range(N)]
        A = [A_fun(Z[:,k], V[:,k]) for k in range(N)]
        B = [B_fun(Z[:,k], V[:,k]) for k in range(N)]
        E = [E_fun(Z[:,j]) for j in range(N + 1)]

        Phi_x = [[None for _ in range(N + 1)] for _ in range(N + 1)]
        for jj in range(N + 1):
            Phi_x[jj][jj] = E[jj]
            for kk in range(jj, N):
                Phi_x[kk + 1][jj] = A[kk] @ Phi_x[kk][jj] + B[kk] @ Phi_u[kk][jj]

        beta = [[None for _ in range(N)] for _ in range(N)]
        for jj in range(N):
            for kk in range(jj, N):
                XU = ca.vertcat(Phi_x[kk][jj], Phi_u[kk][jj])
                tmp = G @ XU
                beta[kk][jj] = ca.sum2(tmp**2)
        for kk in range(N):
            for jj in range(kk + 1, N):
                beta[kk][jj] = ca.DM.zeros(ni, 1)

        beta_f = [ca.sum2((Gf @ Phi_x[N][j]) ** 2) for j in range(N + 1)]
        beta_vec = ca.vertcat(*[ca.vertcat(*row) for row in beta])
        beta_f_vec = ca.vertcat(*beta_f)

        inputs = [Z] + [V] + sum(Phi_u, [])
        outputs = [beta_vec, beta_f_vec]
        return ca.Function('tighten', inputs, outputs, {'jit': False})
