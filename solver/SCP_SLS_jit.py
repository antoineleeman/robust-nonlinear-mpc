import casadi as ca
import numpy as np

from solver.ocp import OCP
from dyn.LTV import LTV
from solver.fast_SLS_jit import fast_SLS
from solver.nlp import NLP

from prettytable import PrettyTable, NONE, HEADER
from util.SLS import SLS
from scipy.sparse import csc_matrix
from scipy.linalg import solve_discrete_are  # added for LQR computation
import time

class SCP_SLS(OCP):
    """
    This class is an implementation of the Algorithm 2 in the paper: https://arxiv.org/pdf/2005.13297.pdf

    RTI option:
      - rti == -1 (default): iterate until convergence (subject to MAX_ITER_SCP safety cap)
      - rti > 0: perform exactly rti SCP iterations and return the current iterate
    """

    def __init__(self, N, Q, R, m, Qf, Q_reg=None, R_reg=None, Q_reg_f=None, **kwargs):
        super().__init__(N, Q, R, m, Qf, Q_reg, R_reg, Q_reg_f)

        self.c_offset_dynamics_fun = None
        self.H_mat = None
        self.epsilon_convergence = 1e-10
        self.current_iteration_scp = {}  # structure that contains current iteration data
        self.convergence_data_scp = {}  # timings, number of iterations, convergence, etc.
        self.it_data = {}  # structure that contains iteration data
        self.save_it_data = kwargs.get("save_it_data", True)
        self.verbose = kwargs.get("verbose", True)
        # RTI iterations: -1 => until convergence; >0 => run exactly that many iterations
        self.rti = kwargs.get("rti", -1)
        # Toggle: enable/disable warm-start inequality ubg shifting
        self.warm_start_ubg_shift_enabled = kwargs.get("warm_start_ubg_shift", True)

        # Pass-through: cap inner fast_SLS_jit iterations (None or <=0 disables)
        self.fast_sls_rti_steps = kwargs.get("fast_sls_rti_steps", None)

        # This flag indicates whether the linearization error is used. Only with an overbound of the linearization error, we can guarantee robust constraint satisfaction for the uncertain nonlinear dynamics.
        self.linearization_error = kwargs.get("linearization_error", False)

        self.nominal_trajectory_solver = None
        self.solver_nominal_params = {}
        self.initialize_nominal_trajectory_solver()

        self.MAX_ITER_SCP = 100
        self.A_fun = None
        self.B_fun = None
        self.E_fun = None
        self.jac_all = None
        self.jac_all_map = None
        self.initialize_jacobian_Function()
        # Pre-build mapped function for horizon length N
        # self.make_mapped(self.N)

        self.fast_SLS_solver = None
        self.initialize_fast_SLS_solver()
        self.fast_SLS_solver.verbose = False
        self.fast_SLS_solver.save_it_data = False

    def solve(self, x0):
        """
        Implementation of an SCP algorithm. Each iteration of the SCP consists of the fast-SLS algorithm.
        This algorithm is an implementation of the Algorithm 2 in the paper: https://arxiv.org/pdf/2005.13297.pdf
        """
        table = None
        if self.verbose:
            table = self.printHeader()
        # nominal trajectory initialization (only if none exists yet, e.g., after warm start we skip)
        has_nominal = ('primal_x' in self.current_iteration_scp) and ('primal_u' in self.current_iteration_scp)
        if not has_nominal:
            if not self.solve_nominal_trajectory(x0):
                return {'success': False}

        # Push current dynamics/cost once; then apply pending warm bounds shift to QP
        self.update_jacobian()
        # If a warm-start inequality ubg shift was prepared, apply it now (after dynamics/cost are up-to-date)
        shift_ubg = False
        if shift_ubg:
            if 'pending_ubg_shift_ineq' in self.current_iteration_scp:
                try:
                    base = np.array(self.fast_SLS_solver.solver_forward.ubg, dtype=float).reshape(-1)
                    nx, ni, ni_f, N = self.m.nx, self.m.ni, self.m.ni_f, self.N
                    block = nx + ni
                    total_stage = block * N
                    if base.size >= total_stage:
                        stage_mat = np.reshape(base[:total_stage], (block, N), order='F')
                        # replace only inequality rows with shifted ones
                        ineq_shift = self.current_iteration_scp['pending_ubg_shift_ineq']
                        stage_mat[nx:, :] = np.reshape(ineq_shift, (ni, N), order='F')
                        new_ubg = np.concatenate([
                            np.reshape(stage_mat, (total_stage,), order='F'),
                            base[total_stage:total_stage + ni_f]  # keep terminal unchanged
                        ])
                        self.fast_SLS_solver.solver_forward.update_ubg(new_ubg)
                finally:
                    # consume it exactly once
                    self.current_iteration_scp.pop('pending_ubg_shift_ineq', None)

        # Determine iteration policy (RTI)
        if self.rti is not None and self.rti > 0:
            max_iters = int(self.rti)
            run_until_converged = False
        else:
            max_iters = int(self.MAX_ITER_SCP)
            run_until_converged = True  # -1 => until convergence (safety cap still applies)

        last_iter = 0
        last_success = False
        for ii in range(max_iters):
            last_iter = ii
            # solve fast-SLS algorithm
            step_ok = self.socp_step(x0)
            last_success = bool(step_ok)
            if not step_ok:
                break
            if self.verbose and table is not None:
                self.printLine(ii, table)

            if run_until_converged and self.check_convergence_scp():
                self.current_iteration_scp['success'] = True
                self.current_iteration_scp['iterations'] = ii
                if self.verbose:
                    print('SCP-SLS: Solution found! Converged in {} iterations'.format(ii))
                sol_ipopt = self.nominal_trajectory_solver.solve(x0,
                                                                 x_guess=self.current_iteration_scp['primal_x'],
                                                                 u_guess=self.current_iteration_scp['primal_u'])
                print("Refinement IPOPT: success = {}, cost = {:.6e}".format(sol_ipopt['success'], float(sol_ipopt['cost'])))
                return self.current_iteration_scp

            if self.save_it_data:
                self.it_data[ii] = self.current_iteration_scp.copy()

            # Update linearization and costs for next SOC iteration
            self.update_jacobian()

        # Exit policy
        self.current_iteration_scp['iterations'] = last_iter
        if run_until_converged:
            # RTI disabled: report non-convergence
            self.current_iteration_scp['success'] = False
            if self.verbose:
                print('SCP did not converge in {} iterations'.format(last_iter))
        else:
            # RTI enabled: report whether last step solved
            self.current_iteration_scp['success'] = last_success
        solution = self.post_processing_solution()
        return solution

    def initialize_nominal_trajectory_solver(self):
        """
        This method initializes the nominal trajectory solver
        :return:
        """
        self.nominal_trajectory_solver = NLP(self.N, self.Q, self.R, self.m, self.Qf)

    def solve_nominal_trajectory(self, x0):
        """
        This method provides an initial guess for the SCP algorithm by solving the nominal trajectory optimization problem.
        It also initializes the Phi_u matrix at zeros.
        :param x0:
        :return:
        """
        sol = self.nominal_trajectory_solver.solve(x0)

        if not sol['success']:
            if self.verbose:
                print('SCP-SLS: Initial guess nominal trajectory did not converge!')
            return sol['success']

        self.current_iteration_scp['primal_x'] = sol['primal_x']
        self.current_iteration_scp['primal_u'] = sol['primal_u']
        self.current_iteration_scp['primal_vec'] = self.pack_primal_nominal(sol['primal_x'], sol['primal_u'])
        self.current_iteration_scp['dual_vec'] = sol['dual_vec']
        self.current_iteration_scp['cost'] = sol['cost']
        self.current_iteration_scp['Phi_u'] = np.zeros((self.N, self.N + 1, self.m.nx, self.m.nu))
        self.current_iteration_scp['dual_eta'] = np.zeros((self.N, self.N, self.m.ni))
        self.current_iteration_scp['dual_eta_f'] = np.zeros((self.N + 1, self.m.ni_f))

        if self.verbose:
            print('SCP-SLS: Initial guess nominal trajectory converged!')
            print('  cost: {:.6e}'.format(float(sol['cost'])))

        return sol['success']

    def initialize_jacobian_Function(self):
        """
        This method initializes the Jacobian functions and a batched (mapped) factory for fast evaluation.
        :return:
        """
        nx = self.m.nx
        nu = self.m.nu

        # Use SX for speed when differentiating small systems
        xS = ca.SX.sym('x', nx)
        uS = ca.SX.sym('u', nu)
        xpS = ca.SX.sym('xp', nx)

        fS = self.m.ddyn(xS, uS)
        A = ca.jacobian(fS, xS)
        B = ca.jacobian(fS, uS)
        c_offset = fS - xpS

#        opts = {'jit': True, 'jit_options': {'flags': ['-O3']}}
        opts = {
            'jit': False,
            'jit_options': None
        }

        # Keep per-stage functions for compatibility elsewhere
        self.A_fun = ca.Function('A_fun', [xS, uS], [A], opts)
        self.B_fun = ca.Function('B_fun', [xS, uS], [B], opts)
        self.E_fun = ca.Function('E_fun', [xS], [self.m.E], opts)
        self.c_offset_dynamics_fun = ca.Function('c_offset_dynamics_fun', [xS, uS, xpS], [c_offset], opts)

        # Batched factory: maps (nx,N),(nu,N),(nx,N) -> (nx,nx,N),(nx,nu,N),(nx,N)
        self.jac_all = ca.Function(
            'jac_all',
            [xS, uS, xpS],
            [A, B, c_offset],
            ['x', 'u', 'xp'],
            ['A', 'B', 'c'],
            opts
        )

        # E_traj includes the uncertainty of the initial conditions
        self.current_iteration_scp['E_traj'] = np.zeros((self.m.nx, self.m.nw, self.N + 1))

        # the first element is the uncertainty of the initial conditions
        self.current_iteration_scp['E_traj'][:, :, 0] = self.m.E

        # all the other E_traj must correspond to the same value, which is the value of E
        for i in range(1, self.N + 1):
            self.current_iteration_scp['E_traj'][:, :, i] = self.m.E

        # initialize the list of constraints
        self.current_iteration_scp['g_list'] = [self.m.g for _ in range(self.N)]
        self.current_iteration_scp['g_list'].append(self.m.gf)  # final state constraint

    def make_mapped(self, N):
        """Create a column-wise mapped version of jac_all over horizon N."""
        if self.jac_all is None:
            return
        # Vectorize over the 3rd dimension (columns interpreted as batch items)
        self.jac_all_map = self.jac_all.map(N)

    def update_jacobian(self):
        """
        This method updates the Jacobian of the dynamics function based on the current nominal trajectory.
        :return:
        """
        nx = self.m.nx
        nu = self.m.nu
        N = self.N

        primal_x = self.current_iteration_scp['primal_x']
        primal_u = self.current_iteration_scp['primal_u']

        if self.linearization_error:
            raise NotImplementedError("Linearization error is not implemented yet")
        else:
            E_list = [self.m.E for _ in range(N + 1)]

        t0_j = time.perf_counter()
        if self.jac_all_map is None:
            # Fallback to per-stage evaluation
            A_list = [np.zeros((nx, nx)) for _ in range(N)]
            B_list = [np.zeros((nx, nu)) for _ in range(N)]
            c_offset_list = [np.zeros((nx, 1)) for _ in range(N)]
            for i in range(N):
                A_list[i] = self.A_fun(primal_x[:, i], primal_u[:, i])
                B_list[i] = self.B_fun(primal_x[:, i], primal_u[:, i])
                c_offset_list[i] = self.c_offset_dynamics_fun(primal_x[:, i], primal_u[:, i], primal_x[:, i + 1])
        else:

            # Prepare stacked inputs (column-wise batches)
            X = primal_x[:, :N]
            U = primal_u[:, :N]
            XP = primal_x[:, 1:N + 1]

            raise NotImplementedError("jac_all_map is not implemented yet")
            # Batched evaluation: outputs may be stacked in different ways depending on CasADi version
            t0 = time.perf_counter()
            A_stack, B_stack, c_stack = self.jac_all_map(X, U, XP)
            t1 = time.perf_counter()
            elapsed = t1 - t0
            print(f"time: {elapsed * 1e3:6.3f} ms (jac_all_map)")
            # ---- A: expect (nx,nx) per stage ----
            A_list = []
            a_r, a_c = A_stack.shape
            if a_r == nx and a_c == nx * N:
                # block-column concatenation: [A0 | A1 | ... | A_{N-1}]
                for i in range(N):
                    A_list.append(A_stack[:, i * nx:(i + 1) * nx])
            elif a_r == nx * nx and a_c == N:
                # vectorized columns: each col is vec(A_k)
                for i in range(N):
                    A_list.append(ca.reshape(A_stack[:, i], nx, nx))
            else:
                # fallback: try reshaping to (nx*nx, N) then split
                A_vec = ca.reshape(A_stack, nx * nx, N)
                for i in range(N):
                    A_list.append(ca.reshape(A_vec[:, i], nx, nx))

            # ---- B: expect (nx,nu) per stage ----
            B_list = []
            b_r, b_c = B_stack.shape
            if b_r == nx and b_c == nu * N:
                for i in range(N):
                    B_list.append(B_stack[:, i * nu:(i + 1) * nu])
            elif b_r == nx * nu and b_c == N:
                for i in range(N):
                    B_list.append(ca.reshape(B_stack[:, i], nx, nu))
            else:
                B_vec = ca.reshape(B_stack, nx * nu, N)
                for i in range(N):
                    B_list.append(ca.reshape(B_vec[:, i], nx, nu))

            # ---- c_offset: expect (nx,1) per stage ----
            c_offset_list = []
            c_r, c_c = c_stack.shape
            if c_r == nx and c_c == N:
                for i in range(N):
                    c_offset_list.append(c_stack[:, i])
            elif c_r == nx * N and c_c == 1:
                c_mat = ca.reshape(c_stack, nx, N)
                for i in range(N):
                    c_offset_list.append(c_mat[:, i])
            else:
                # generic fallback
                c_mat = ca.reshape(c_stack, nx, N)
                for i in range(N):
                    c_offset_list.append(c_mat[:, i])

        t1_j = time.perf_counter()
        # export last Jacobian evaluation time (ms)
        self.current_iteration_scp['t_jac_ms'] = float((t1_j - t0_j) * 1e3)

        self.current_iteration_scp['A_list'] = A_list
        self.current_iteration_scp['B_list'] = B_list
        self.current_iteration_scp['c_offset_list'] = c_offset_list
        self.current_iteration_scp['E_list'] = E_list

        ni = self.m.ni
        g_list = [np.zeros(ni) for _ in range(N)]
        # update the dynamics list of constraints
        for i in range(N):
            z = ca.vertcat(primal_x[:, i], primal_u[:, i])
            g_list[i] = self.m.g - self.m.G @ z
        # the last element of g_list is the final state constraint
        z = ca.vertcat(primal_x[:, -1])
        g_list.append(self.m.gf - self.m.Gf @ z)

        self.current_iteration_scp['g_list'] = g_list
        self.fast_SLS_solver.update_dynamics_list(A_list, B_list, E_list, g_list, c_offset_list)

        # update the linear part of the cost
        H_mat = self.H_mat
        H_dm = np.array(self.scipy_csc_to_casadi_dm(H_mat))
        y_nom = np.array(ca.DM(self.current_iteration_scp['primal_vec']))
        q_lin_cost = 2* (H_dm @ y_nom)
        self.fast_SLS_solver.update_linear_cost(q_lin_cost)

    @staticmethod
    def scipy_csc_to_casadi_dm(A: csc_matrix) -> ca.DM:
        # CasADi uses the same CSC layout (indptr = col offsets, indices = row ids)
        sp = ca.Sparsity(A.shape[0], A.shape[1], A.indptr.tolist(), A.indices.tolist())
        return ca.DM(sp, A.data)

    def initialize_fast_SLS_solver(self):
        """
        This method initializes the fast-SLS solver
        :return:
        """
        init_LTV = LTV(self.m, self.N)
        self.fast_SLS_solver = fast_SLS(self.N, self.Q, self.R, init_LTV, self.Qf)
        # Apply inner RTI cap if requested
        try:
            self.fast_SLS_solver.set_rti_steps(self.fast_sls_rti_steps)
        except Exception:
            pass
        self.fast_SLS_solver.Q_reg = self.Q_reg
        self.fast_SLS_solver.R_reg = self.R_reg
        self.fast_SLS_solver.Q_reg_f = self.Q_reg_f
        self.H_mat = self.fast_SLS_solver.solver_forward.P_mat_csc

        # the matrix H_mat is the same as in the fast-SLS solver.
        # We assume here the cost is quadratic, so we don't need to linearize/quadratize it at each iteration.

    def set_fast_sls_rti_steps(self, steps: int | None):
        """Configure the inner fast_SLS_jit to stop after exactly `steps` iterations per SCP call."""
        self.fast_sls_rti_steps = steps
        if hasattr(self, 'fast_SLS_solver') and self.fast_SLS_solver is not None:
            try:
                self.fast_SLS_solver.set_rti_steps(steps)
            except Exception:
                pass


    def socp_step(self, x0):
        """ This method performs one step of the SCP algorithm, which is a fast-SLS step.
        :return: True if the step was successful, False otherwise
        """
        x0 = np.asarray(x0).reshape(-1)
        x_nom0 = np.asarray(self.current_iteration_scp['primal_x'][:, 0]).reshape(-1)
        solution = self.fast_SLS_solver.solve(x_nom0 - x0)

        if solution['success']:
            # export last inner timings (ms)
            self.current_iteration_scp['t_qp_ms'] = float(solution.get('t_qp_ms', np.nan))
            self.current_iteration_scp['t_backward_ms'] = float(solution.get('t_backward_ms', np.nan))
            # extract nominal trajectory
            delta_x = solution['primal_x']
            delta_u = solution['primal_u']
            # stage-wise delta
            dual_vec = solution['dual_vec']
            delta_vec = self.pack_primal_nominal(delta_x, delta_u)

            ret = self.eval_deviation_mismatch(delta_x, delta_u)

            x_debb, u_debb = self.unpack_primal_nominal(np.array(self.current_iteration_scp['primal_vec']))
            self.current_iteration_scp['primal_x'] = self.current_iteration_scp['primal_x'] + delta_x
            self.current_iteration_scp['primal_u'] = self.current_iteration_scp['primal_u'] + delta_u
            self.current_iteration_scp['dual_vec'] = dual_vec
            self.current_iteration_scp['delta_dual_vec'] = dual_vec - self.current_iteration_scp.get('dual_vec', 0)
            self.current_iteration_scp['primal_vec'] += delta_vec
            self.current_iteration_scp['dual_mu'] = solution['dual_mu']
            self.current_iteration_scp['dual_mu_f'] = solution['dual_mu_f']
            self.current_iteration_scp['cost_QP'] = solution['cost_nominal']
            self.current_iteration_scp['SOCP_steps'] = solution['iteration_number']
            H_mat = self.H_mat
            H_dm = np.array(self.scipy_csc_to_casadi_dm(H_mat))
            y_nom = np.array(ca.DM(self.current_iteration_scp['primal_vec']))
            cost_nlp = 0
            for i in range(self.N):
                x_i = self.current_iteration_scp['primal_x'][:, i].reshape(-1, 1)
                u_i = self.current_iteration_scp['primal_u'][:, i].reshape(-1, 1)
                cost_nlp += x_i.T @ self.Q @ x_i + u_i.T @ self.R @ u_i
            x_N = self.current_iteration_scp['primal_x'][:, self.N].reshape(-1, 1)

            cost_nlp += x_N.T @ self.Qf @ x_N
            cost_nlp_alt = y_nom.T @ H_dm @ y_nom
            x_deb, u_deb = self.unpack_primal_nominal(y_nom)

            self.current_iteration_scp['cost'] = solution['cost_nominal'] + cost_nlp

            # arr = np.array(self.current_iteration_scp['c_offset_list'])
            arr = np.array([self.m.ddyn(self.current_iteration_scp['primal_x'][:, i].reshape(-1),
                                  self.current_iteration_scp['primal_u'][:, i].reshape(-1)) -
                      self.current_iteration_scp['primal_x'][:, i + 1].reshape(-1) for i in range(self.N)])[:, :, 0]
            max_val = np.max(arr)
            self.current_iteration_scp['primal_infeasibility'] = max_val
            self.current_iteration_scp['dual_eta'] = solution['eta']
            self.current_iteration_scp['dual_eta_f'] = solution['eta_f']

            self.current_iteration_scp['delta_vec'] = delta_vec
            self.current_iteration_scp['backoff'] = solution['backoff']
            self.current_iteration_scp['backoff_x'] = solution['backoff_x']
            self.current_iteration_scp['backoff_u'] = solution['backoff_u']
            self.current_iteration_scp['Phi_x'] = solution.get('Phi_x')
            self.current_iteration_scp['Phi_u'] = solution.get('Phi_u')

        # reset fast_SLS solver
        # self.fast_SLS_solver.reset_solver_to_zeros()

        if self.verbose:
            if not solution['success']:
                print('SCP-SLS: Fast-SLS did not converge!')
        return solution['success']
    def post_processing_solution(self):
        """
        This method post-processes the solution
        :return:
        """
        solution = self.current_iteration_scp.copy()
        solution.update(self.convergence_data_scp)
        solution['it_data'] = self.it_data.copy()

        return solution


    def reset(self):
        """
        This method resets the SCP_SLS solver to its initial state.
        :return:
        """
        self.current_iteration_scp = {}
        self.convergence_data_scp = {}
        self.it_data = {}
        self.fast_SLS_solver.reset_solver_to_zeros()
        self.initialize_jacobian_Function()
        #self.make_mapped(self.N)
        self.initialize_fast_SLS_solver()
        self.initialize_nominal_trajectory_solver()

    def reset_warm_start(self):
        """Prepare warm start by shifting x/u one step and resetting internal buffers.
        Do not push updates to fast_SLS/QP here to preserve update order; instead, set a flag so
        solve() can shift the bounds after pushing new dynamics and linear cost.
        """
        N = self.N

        # Nominal trajectory shift:
        X = np.array(self.current_iteration_scp['primal_x']).copy()
        U = np.array(self.current_iteration_scp['primal_u']).copy()

        # Shift trajectories: x_k <- x_{k+1}, u_k <- u_{k+1}
        X_new = X.copy()
        U_new = U.copy()
        X_new[:, :N] = X[:, 1:N+1]
        if N >= 2:
            U_new[:, :N-1] = U[:, 1:N]
        U_new[:, N-1] = U[:, N-1]
        X_new[:, N] =np.array( self.m.ddyn(X[:, N],U[:, N-1].reshape(-1,1))).reshape(-1)

        # Extract current ubg and prepare shifted inequality-only blocks
        ubg = np.array(self.fast_SLS_solver.solver_forward.ubg, dtype=float).reshape(-1)
        nx, ni, ni_f, N = self.m.nx, self.m.ni, self.m.ni_f, self.N
        block = nx + ni
        total_stage = block * N
        if ubg.size >= total_stage:
            stage_mat = np.reshape(ubg[:total_stage], (block, N), order='F')
            ineq_mat = stage_mat[nx:, :].copy()  # shape (ni, N)
            # Shift inequality blocks by one; duplicate penultimate for the last; keep terminal separate later
            if N >= 2:
                ineq_mat[:, :-1] = ineq_mat[:, 1:]
                ineq_mat[:, -1] = ineq_mat[:, -2]
            # else N==1 -> unchanged
            ubg_shifted_ineq = np.reshape(ineq_mat, (ni * N,), order='F')
        else:
            ubg_shifted_ineq = None

        # Reset internal state and keep the shifted inequality ubg to be applied at next solve()
        self.current_iteration_scp = {}
        self.convergence_data_scp = {}
        self.it_data = {}
        self.fast_SLS_solver.reset_solver_to_zeros()

        # Stage-wise primal vector
        primal_vec = self.pack_primal_nominal(X_new, U_new)
        # Store shifted state in SCP cache
        self.current_iteration_scp['primal_x'] = X_new
        self.current_iteration_scp['primal_u'] = U_new
        self.current_iteration_scp['primal_vec'] = primal_vec
        # Stash pending ineq-only ubg shift for application in solve()
        if ubg_shifted_ineq is not None:
            self.current_iteration_scp['pending_ubg_shift_ineq'] = ubg_shifted_ineq


    @staticmethod
    def printHeader():
        fixed_width = 10
        # Format headers to have fixed width and right alignment
        headers = ["it (SCP)", "∆ primal", "∆ dual", "∆ cost", "cost nom.", "p. infeas.", "SOCP it"]
        formatted_headers = [f"{h:>{fixed_width}}" for h in headers]
        table = PrettyTable()
        table.field_names = formatted_headers
        table.hrules = HEADER  # Horizontal line after header
        table.border = True

        # Align columns
        table.align["it"] = "right"
        table.align["primal"] = "right"
        table.align["dual"] = "right"
        table.align["cost"] = "right"
        table.align["cost nom."] = "right"
        table.align["p. infeas."] = "right"
        table.align["SOCP it"] = "right"
        # Set a fixed width for all columns
        fixed_width = 10
        # Set fixed widths for each column individually
        table.max_width["it"] = fixed_width
        table.max_width["primal"] = fixed_width
        table.max_width["dual"] = fixed_width
        table.max_width["cost"] = fixed_width
        table.max_width["cost nom."] = fixed_width
        table.max_width["p. infeas."] = fixed_width
        table.max_width["SOCP it"] = fixed_width

        print(table.get_string(end=0))
        table.hrules = NONE

        return table

    def printLine(self, i, table):

        fixed_width = 10
        primal = np.max(self.current_iteration_scp['delta_vec'])
        dual = np.max(self.current_iteration_scp['delta_dual_vec'])
        cost_QP = self.current_iteration_scp['cost_QP']
        cost = self.current_iteration_scp['cost']
        primal_infeasibility = self.current_iteration_scp['primal_infeasibility']
        socp_steps_number = self.current_iteration_scp['SOCP_steps']

        iteration_str = f"{i:>{fixed_width}}"
        primal_val = f"{float(primal):>{fixed_width}.2e}"
        dual_val = f"{float(dual):>{fixed_width}.2e}"
        cost_QP_val = f"{float(cost_QP):>{fixed_width}.2e}"
        cost = f"{float(cost):>{fixed_width}.2e}"
        primal_infeasibility = f"{float(primal_infeasibility):>{fixed_width}.2e}"
        socp_steps_number = f"{socp_steps_number:>{fixed_width}}"

        table.add_row([iteration_str, primal_val, dual_val, cost_QP_val, cost, primal_infeasibility, socp_steps_number])
        print(table.get_string(start=len(table._rows) - 1, end=len(table._rows), header=False))

    def check_convergence_scp(self):
        """
        This method checks the convergence of the SCP algorithm
        :return: True if the SCP algorithm has converged
        """
        # check if the value delta_x is already assigned
        if 'delta_vec' in self.current_iteration_scp:
            delta_vec = self.current_iteration_scp['delta_vec']
            conv = np.max(abs(delta_vec))
            if conv < self.epsilon_convergence:
                return True

        return False

    def generate_lqr_controller(self):
        """
        Generate an infinite-horizon discrete LQR controller from the linearization at the origin.
        Uses self.Q and self.R and returns the feedback gain K and a controller callable u = -K x.

        Returns:
            dict with keys:
                - K: (nu, nx) LQR gain
                - P: (nx, nx) DARE solution
                - A: (nx, nx) linearized A at (0,0)
                - B: (nx, nu) linearized B at (0,0)
                - controller: function mapping x -> u
        """
        # Ensure Jacobian functions are available
        if self.A_fun is None or self.B_fun is None:
            self.initialize_jacobian_Function()

        nx, nu = self.m.nx, self.m.nu
        x0 = np.zeros(nx)
        u0 = np.zeros(nu)

        # Linearize dynamics at the origin
        A = np.array(self.A_fun(x0, u0))
        B = np.array(self.B_fun(x0, u0))

        # Solve discrete-time ARE for infinite-horizon LQR
        P = solve_discrete_are(A, B, self.Q, self.R)
        K = np.linalg.solve(self.R + B.T @ P @ B, B.T @ P @ A)

        def controller(x: np.ndarray) -> np.ndarray:
            return -K @ x

        self.K = K
        self.Qf = P

        return {"K": K, "P": P, "A": A, "B": B, "controller": controller}

    def eval_deviation_mismatch(self, e: np.ndarray, d: np.ndarray) -> dict:
        """
        Evaluate the deviation modeling error across the horizon.
        Predicted: e_{k+1} = A_k e_k + B_k d_k + r_k,  with r_k := f(z_k, v_k) - z_{k+1}
        Actual roll-out: e^{roll}_{k+1} = f(z_k + e_k, v_k + d_k) - z_{k+1}

        Inputs:
          - e: (nx, N+1) array of state deviations (includes terminal e_N)
          - d: (nu, N)  array of input deviations
        Returns:
          dict with keys:
            mismatch: (nx, N) array Δ_{k+1} = e^{roll}_{k+1} - (A e_k + B d_k + r_k)
            pred:     (nx, N) predicted e_{k+1}
            roll:     (nx, N) rolled e^{roll}_{k+1}
            r:        (nx, N) residual r_k
            norms:    (N,)    2-norm of mismatch per step
        """
        z = np.array(self.current_iteration_scp['primal_x'])   # (nx, N+1)
        v = np.array(self.current_iteration_scp['primal_u'])   # (nu, N)
        N = self.N
        nx = self.m.nx
        nu = self.m.nu

        assert e.shape == (nx, N + 1), f"e must be {(nx, N+1)}, got {e.shape}"
        assert d.shape == (nu, N), f"d must be {(nu, N)}, got {d.shape}"

        # Use current linearization (A, B)
        if 'A_list' not in self.current_iteration_scp or 'B_list' not in self.current_iteration_scp:
            self.update_jacobian()
        A_list = self.current_iteration_scp['A_list']
        B_list = self.current_iteration_scp['B_list']

        pred = np.zeros((nx, N))
        roll = np.zeros((nx, N))
        r = np.zeros((nx, N))
        mismatch = np.zeros((nx, N))

        for k in range(N):
            # residual r_k = f(z_k, v_k) - z_{k+1}
            fk = np.array(self.m.ddyn(z[:, k], v[:, k])).reshape(-1)
            rk = fk - z[:, k + 1]
            r[:, k] = rk

            # predicted next deviation
            pred[:, k] = (np.array(A_list[k]) @ e[:, k] + np.array(B_list[k]) @ d[:, k] + rk)

            # roll the true system from deviated point
            f_roll = np.array(self.m.ddyn(z[:, k] + e[:, k], v[:, k] + d[:, k])).reshape(-1)
            roll[:, k] = f_roll - z[:, k + 1]

            mismatch[:, k] = roll[:, k] - pred[:, k]

        norms = np.linalg.norm(mismatch, axis=0)
        return {
            'mismatch': mismatch,
            'pred': pred,
            'roll': roll,
            'r': r,
            'norms': norms,
        }
