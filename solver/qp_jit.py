import numpy as np
import casadi as ca
import scipy.sparse as sp
from scipy.io import savemat
from pathlib import Path
import shutil
import time

from solver.ocp import OCP
from dyn.LTV import LTV
from dyn.LTI import LTI

try:
    import osqp  # optional: native OSQP path + codegen
    _HAS_OSQP = True
except Exception:
    _HAS_OSQP = False

EPSILON = 1e-10


class QP(OCP):
    """
    Minimal, fast QP front-end for fast-SLS with three interchangeable backends:
      • "casadi_osqp"   — CasADi conic using OSQP plugin (no codegen)
      • "osqp"          — Native OSQP Python API (fast updates)
      • "osqp_codegen"  — Compiled OSQP Python extension (generated with parameters='matrices')

    Contract with fast-SLS:
      - build once; then only q (linear term), l/u (bounds), and numeric A/P update per iteration
      - expose: solve(x0), update_dynamics(...), update_ubg(...), reset_ubg(),
                update_q_cost_lin(...), add_q_cost_lin(...), reset_q_cost_lin()
    """

    def __init__(self, N, Q, R, m, Qf, *, backend="casadi_osqp", codegen_module=None, verbose=True,
                 export_standard_QP: bool = False, export_dir: str | Path | None = None):
        super().__init__(N, Q, R, m, Qf)
        assert backend in {"casadi_osqp", "osqp", "osqp_codegen"}
        self.backend = backend
        self.codegen_module = codegen_module
        self.verbose = verbose

        # export options
        self.export_standard_qp = bool(export_standard_QP)
        self.export_dir = Path(export_dir) if export_dir is not None else Path("build/quadprog_exports")
        self._export_counter = 0

        # structures
        self.A_mat_csc: sp.csc_matrix | None = None
        self.P_mat_csc: sp.csc_matrix | None = None
        self.A_mat_cas: ca.DM | None = None
        self.P_mat_cas: ca.DM | None = None

        # vectors
        self.lbg = None
        self.ubg = None
        self.nominal_ubg = None
        self.nominal_lbg = None
        self.q_cost_lin = None

        # casadi helpers
        self.lbx_fun = None
        self.ubx_fun = None

        # solver objects
        self.solver_qp = None     # CasADi conic
        self._osqp_prob = None    # osqp.OSQP() or codegen module/object

        # build once
        self.initialize_list_dynamics()
        self._assemble_structures()
        self._init_backend()

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------
    def _assemble_structures(self):
        nx, nu, ni, ni_f, N = self.m.nx, self.m.nu, self.m.ni, self.m.ni_f, self.N

        rows = []
        cols = []
        data = []

        def put(r0, c0, M):
            C = sp.coo_matrix(M)
            rows.extend(r0 + C.row)
            cols.extend(c0 + C.col)
            data.extend(C.data)

        def put_dense_block(r0, c0, M):
            """Insert a dense block M (including explicit zeros) to freeze sparsity pattern."""
            M = np.asarray(M)
            nr, nc = M.shape
            # generate all coordinates for the block
            rr = r0 + np.repeat(np.arange(nr), nc)
            cc = c0 + np.tile(np.arange(nc), nr)
            rows.extend(rr)
            cols.extend(cc)
            data.extend(M.reshape(-1))

        r = 0
        for k in range(N):
            Ak = self.A_list[k]
            Bk = self.B_list[k]
            # dynamics: [A B -I] on (x_k, u_k, x_{k+1})
            cx = k * (nx + nu)
            cu = cx + nx
            cxp = (k + 1) * (nx + nu)
            # Use dense placement for A and B so their sparsity never changes across updates
            put_dense_block(r, cx, Ak)
            put_dense_block(r, cu, Bk)
            put(r, cxp, -np.eye(nx))
            r += nx
            # ineq: G[x;u] <= g_k
            G = self.m.G
            put(r, cx, G[:, :nx])
            put(r, cu, G[:, nx:])
            r += ni
        # terminal ineq: Gf x_N <= g_f
        put(r, N * (nx + nu), self.m.Gf)
        r += ni_f

        A = sp.csc_matrix((data, (rows, cols)), shape=(r, (nx + nu) * N + nx))

        # Quadratic cost: blkdiag(Q,R,...,Qf)
        Qs = sp.csc_matrix(self.Q)
        Rs = sp.csc_matrix(self.R)
        Qf = sp.csc_matrix(self.Qf)
        QR = sp.bmat([[Qs, None], [None, Rs]], format='csc')
        P = sp.block_diag([QR for _ in range(N)] + [Qf], format='csc')

        self.A_mat_csc = A
        self.P_mat_csc = P
        self.A_mat_cas = ca.DM(A)
        self.P_mat_cas = ca.DM(P)

        # --- freeze CSC ordering for OSQP/codegen ---
        self.A_mat_csc.sort_indices()
        self.P_mat_csc.sort_indices()

        # bounds lbg/ubg
        if isinstance(self.m, LTI):
            ub = ca.kron(ca.DM.ones(N, 1), ca.vertcat(ca.DM.zeros(nx, 1), self.m.g))
            ub = ca.vertcat(ub, self.m.gf )
        elif isinstance(self.m, LTV):
            pieces = [ca.vertcat(ca.DM.zeros(nx, 1), g ) for g in self.m.g_list[:-1]]
            ub = ca.vertcat(*pieces)
            ub = ca.vertcat(ub, self.m.g_list[-1] )
        else:
            raise ValueError("Model must be LTI or LTV")
        self.nominal_ubg = ub
        self.ubg = ub

        lb = ca.kron(ca.DM.ones(N, 1), ca.vertcat(ca.DM.zeros(nx, 1), -ca.DM.inf(self.m.ni, 1)))
        self.lbg = ca.vertcat(lb, -ca.DM.inf(self.m.ni_f, 1))
        self.lbg_ZEROS_INF_NOMINAL = self.lbg

        # casadi-only variable box bounds for x0 (others free)
        x0 = ca.SX.sym('x0', nx)
        # Enforce x(0) + x0 = 0  => x(0) = -x0 (tight box around -x0)
        self.ubx_fun = ca.Function(
            'ubx_fun', [x0],
            [ca.vertcat(-x0 + EPSILON, ca.DM.inf(N * (nx + nu), 1))],
            {'jit': False, 'compiler': 'shell'}
        )
        self.lbx_fun = ca.Function(
            'lbx_fun', [x0],
            [ca.vertcat(-x0 - EPSILON, -ca.DM.inf(N * (nx + nu), 1))],
            {'jit': False, 'compiler': 'shell'}
        )

        self.reset_q_cost_lin()

        # fingerprints to ensure sparsity unchanged when updating numerics
        self._A_sig = (tuple(self.A_mat_csc.indices), tuple(self.A_mat_csc.indptr))
        self._P_sig = (tuple(self.P_mat_csc.indices), tuple(self.P_mat_csc.indptr))

        # --- extra rows to pin x0 in OSQP ---
        nx, nu, N = self.m.nx, self.m.nu, self.N
        nv = (nx + nu) * N + nx

        I0 = sp.hstack([sp.eye(nx, format='csc'),
                        sp.csc_matrix((nx, nv - nx))], format='csc')  # acts on the first nx vars (x0)

        self.A_mat_csc_osqp = sp.vstack([self.A_mat_csc, I0], format='csc')
        self.A_mat_csc_osqp.sort_indices()

        # base l/u without x0 values yet (filled per-solve)
        self.lbg_osqp_base = np.concatenate([np.array(self.lbg).astype(float).ravel(),
                                             np.zeros(nx)])
        self.ubg_osqp_base = np.concatenate([np.array(self.ubg).astype(float).ravel(),
                                             np.zeros(nx)])

    def _reassemble_numeric_same_sparsity(self):
        """Recompute numeric values of A and P given current lists; allow arbitrary (full) matrices.
        This rebuilds the constraint matrix and cost blocks from scratch and refreshes OSQP buffers.
        """
        nx, nu, ni, ni_f, N = self.m.nx, self.m.nu, self.m.ni, self.m.ni_f, self.N

        rows = []
        cols = []
        data = []

        def put(r0, c0, M):
            C = sp.coo_matrix(np.asarray(M))
            rows.extend(r0 + C.row)
            cols.extend(c0 + C.col)
            data.extend(C.data)

        def put_dense_block(r0, c0, M):
            """Insert a dense block M (including explicit zeros) to freeze sparsity pattern."""
            M = np.asarray(M, dtype=float)
            nr, nc = M.shape
            rr = r0 + np.repeat(np.arange(nr), nc)
            cc = c0 + np.tile(np.arange(nc), nr)
            rows.extend(rr)
            cols.extend(cc)
            data.extend(M.reshape(-1))

        r = 0
        for k in range(N):
            Ak = np.asarray(self.A_list[k], dtype=float)
            Bk = np.asarray(self.B_list[k], dtype=float)
            cx = k * (nx + nu)
            cu = cx + nx
            cxp = (k + 1) * (nx + nu)
            # Use dense placement to keep pattern fixed
            put_dense_block(r, cx, Ak)
            put_dense_block(r, cu, Bk)
            put(r, cxp, -np.eye(nx))
            r += nx
            G = self.m.G
            put(r, cx, G[:, :nx])
            put(r, cu, G[:, nx:])
            r += ni
        put(r, N * (nx + nu), self.m.Gf)
        r += ni_f

        A_new = sp.csc_matrix((data, (rows, cols)), shape=(r, (nx + nu) * N + nx))
        A_new.sort_indices()
        self.A_mat_csc = A_new
        self.A_mat_cas = ca.DM(A_new)

        # OSQP variant (with extra x0 rows)
        nv = self.A_mat_csc.shape[1]
        I0 = sp.hstack([sp.eye(self.m.nx, format='csc'),
                        sp.csc_matrix((self.m.nx, nv - self.m.nx))], format='csc')
        self.A_mat_csc_osqp = sp.vstack([self.A_mat_csc, I0], format='csc')
        self.A_mat_csc_osqp.sort_indices()

        # refresh OSQP base bounds for extra rows
        self.lbg_osqp_base = np.concatenate([np.array(self.lbg).astype(float).ravel(),
                                             np.zeros(self.m.nx)])
        self.ubg_osqp_base = np.concatenate([np.array(self.ubg).astype(float).ravel(),
                                             np.zeros(self.m.nx)])

        # Rebuild cost (P) unconditionally (sizes unchanged)
        Qs = sp.csc_matrix(self.Q)
        Rs = sp.csc_matrix(self.R)
        Qf = sp.csc_matrix(self.Qf)
        QR = sp.bmat([[Qs, None], [None, Rs]], format='csc')
        P_new = sp.block_diag([QR for _ in range(N)] + [Qf], format='csc')
        P_new.sort_indices()
        self.P_mat_csc = P_new
        self.P_mat_cas = ca.DM(P_new)
        self.P_mat_csc.sort_indices()

        # ubg:
        pieces = [ca.vertcat(ca.DM.zeros(nx, 1), g) for g in self.g_list[:-1]]
        ub = ca.vertcat(*pieces)
        ub = ca.vertcat(ub, self.m.g_list[-1])
        self.update_ubg(ub)
        self.reset_lbg()

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------
    def _init_backend(self):
        if self.backend == "casadi_osqp":
            opts = {'osqp': {'verbose': False, 'polish': True}}
            self.solver_qp = ca.conic('solver', 'osqp',
                                      {'a': self.A_mat_cas.sparsity(), 'h': self.P_mat_cas.sparsity()},
                                      opts)
        elif self.backend == "osqp":
            if not _HAS_OSQP:
                raise RuntimeError("OSQP not installed. pip install osqp")
            prob = osqp.OSQP()
            prob.setup(
                P=2 * self.P_mat_csc,  # match CasADi’s h = 2*H
                q=np.zeros(self.P_mat_csc.shape[0]),
                A=self.A_mat_csc_osqp,  # includes x0 rows
                l=self.lbg_osqp_base,  # placeholders; updated per solve
                u=self.ubg_osqp_base,
                # --- Accuracy / robustness ---
                eps_abs=1e-9,  # tighten as needed (1e-9 if stable)
                eps_rel=1e-9,
                scaled_termination=False,  # check tolerances on ORIGINAL (unscaled) problem
                max_iter=50000,
                check_termination=1,  # check every iteration
                # --- Extras that help ---
                polish=True,
                polish_refine_iter=20,
                warm_start=False,  # reuse solution for the 2nd pass (below)
                adaptive_rho=True,  # keep OSQP’s automatic rho updates
                verbose=False
            )
            self._osqp_prob = prob
        elif self.backend == "osqp_codegen":
            if self.codegen_module is None:
                raise ValueError("backend='osqp_codegen' needs codegen_module")
            mod = __import__(self.codegen_module)
            self._osqp_mod = mod

            # Always compute and store the reference CSC patterns used at init
            P_ut = sp.triu(2 * self.P_mat_csc, format='csc')
            P_ut.sort_indices()
            A_osqp = self.A_mat_csc_osqp
            A_osqp.sort_indices()
            self._store_cg_signatures(P_ut, A_osqp)

            if hasattr(mod, 'OSQP'):
                self._osqp_prob = mod.OSQP()

                # Build matrices exactly as used at codegen time
                self._osqp_prob.setup(
                    P=P_ut,
                    q=np.zeros(P_ut.shape[0]),
                    A=A_osqp,
                    l=self.lbg_osqp_base,
                    u=self.ubg_osqp_base,
                    polishing=True, warm_starting=False, verbose=False
                )
            else:
                # Module-level API (functions like update_data_mat/update_data_vec/solve)
                self._osqp_prob = mod

        else:
            raise ValueError(f"Unknown backend {self.backend}")

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    def solve(self, x0):
        nx, nu, ni, ni_f, N = self.m.nx, self.m.nu, self.m.ni, self.m.ni_f, self.N

        if self.backend == "casadi_osqp":
            try:
                sol = self.solver_qp(a=self.A_mat_cas,
                                     h=2 * self.P_mat_cas,  # CasADi/OSQP: 1/2 x' P x + q' x
                                     lba=self.lbg,
                                     uba=self.ubg,
                                     g=self.q_cost_lin,
                                     lbx=self.lbx_fun(x0),
                                     ubx=self.ubx_fun(x0))
            except Exception as e:
                if self.verbose:
                    print(e)
                    print('QP: infeasible (CasADi/OSQP).')
                return {'success': False}
            return self._pack_solution(sol, nx, nu, N, ni_f)

        if self.backend == "osqp":
            prob = self._osqp_prob

            # linear term
            qv = np.array(self.q_cost_lin).astype(float).ravel()

            # bounds with x0 box: last nx rows correspond to I*x0
            nx = self.m.nx
            l_curr = np.array(self.lbg, dtype=float).ravel()
            u_curr = np.array(self.ubg, dtype=float).ravel()
            # append placeholders for x0 equality rows
            l = np.concatenate([l_curr, np.zeros(nx)])
            u = np.concatenate([u_curr, np.zeros(nx)])

            x0v = np.array(x0, dtype=float).ravel()
            # Enforce x(0) + x0 = 0  => A*y = -x0
            l[-nx:] = -x0v - EPSILON
            u[-nx:] = -x0v + EPSILON

            # sanitize infinities to large magnitudes for OSQP
            BIG = 1e20
            l[np.isneginf(l)] = -BIG
            l[np.isposinf(l)] = BIG
            u[np.isposinf(u)] = BIG
            u[np.isneginf(u)] = -BIG

            # ensure latest dynamics values are used (pattern unchanged)
            A_osqp = self.A_mat_csc_osqp
            A_osqp.sort_indices()
            prob.update(Ax=A_osqp.data, q=qv, l=l, u=u)
            t0 = time.perf_counter()
            res = prob.solve()
            t1 = time.perf_counter()
            time_qp = (t1 - t0) * 1e3

            if res.info.status_val not in (1, 2):
                if self.verbose:
                    print(f"QP(OSQP): {res.info.status}")
                return {'success': False}
            sol = {'x': res.x, 'lam_a': res.y, 'cost': res.info.obj_val, 'success': True, 'time_ms': time_qp}
            return self._pack_solution(sol, nx, nu, N, ni_f)

        if self.backend == "osqp_codegen":
            cg = self._osqp_prob
            nx = self.m.nx

            # ---- vectors (with x0 box in last nx rows) ----
            qv = np.ascontiguousarray(np.array(self.q_cost_lin, dtype=np.float64).ravel())
            l_curr = np.array(self.lbg, dtype=float).ravel()
            u_curr = np.array(self.ubg, dtype=float).ravel()
            l = np.ascontiguousarray(np.concatenate([l_curr, np.zeros(nx)]), dtype=np.float64)
            u = np.ascontiguousarray(np.concatenate([u_curr, np.zeros(nx)]), dtype=np.float64)

            x0v = np.asarray(x0, dtype=np.float64).ravel()
            # Enforce x(0) + x0 = 0  => A*y = -x0
            l[-nx:] = -x0v - EPSILON
            u[-nx:] = -x0v + EPSILON

            BIG = 1e20
            l[np.isneginf(l)] = -BIG
            l[np.isposinf(l)] = BIG
            u[np.isposinf(u)] = BIG
            u[np.isneginf(u)] = -BIG

            # ---- matrices: must match codegen CSC pattern and order ----
            P_ut = sp.triu(2 * self.P_mat_csc, format='csc')
            P_ut.sort_indices()
            A_osqp = self.A_mat_csc_osqp
            A_osqp.sort_indices()

            # Verify CSC pattern and ordering are identical to codegen setup
            if hasattr(self, "_P_cg_sig") and hasattr(self, "_A_cg_sig"):
                self._assert_same_csc_pattern(P_ut, self._P_cg_sig, "P_ut")
                self._assert_same_csc_pattern(A_osqp, self._A_cg_sig, "A_osqp")

            # Take data arrays in the exact frozen order
            Px = np.ascontiguousarray(P_ut.data.astype(np.float64))
            Ax = np.ascontiguousarray(A_osqp.data.astype(np.float64))

            # ---- sanity sizes ----
            n = P_ut.shape[0]
            m = A_osqp.shape[0]
            assert qv.size == n and l.size == m and u.size == m
            assert Px.size == P_ut.nnz
            assert Ax.size == A_osqp.nnz

            # ---- push numerics ----
            self._cg_push_updates(cg, Px, Ax, qv, l, u)

            # ---- solve & normalize result ----
            t0 = time.perf_counter()
            res = cg.solve()
            t1 = time.perf_counter()
            time_qp = (t1 - t0) * 1e3
            ok = False
            if isinstance(res, tuple):
                x = res[0] if len(res) >= 1 else None
                y = res[1] if len(res) >= 2 else None
                # osqp_generated returns: (x, y, status_code, iter, run_time)
                status_code = int(res[2]) if len(res) >= 3 else -1
                ok = (status_code == 0) and (x is not None) and (y is not None)
                status_val = status_code
            else:
                x = getattr(res, "x", None)
                y = getattr(res, "y", None)
                info = getattr(res, "info", None)
                status_val = int(getattr(info, "status_val", 0) if info is not None else 0)
                ok = (status_val in (1, 2)) and (x is not None) and (y is not None)

            if ok:
                cost = float(x @ (self.P_mat_csc @ x) + qv @ x)
                sol = {'x': x, 'lam_a': y, 'cost': cost, 'success': True, 'time_ms': time_qp}
                ret = self._pack_solution(sol, self.m.nx, self.m.nu, self.N, self.m.ni_f)
                if self.export_standard_qp and ret.get('success', True):
                    try:
                        self._export_quadprog(x0, ret)
                    except Exception as ex:
                        if self.verbose:
                            print(f"Export (quadprog) failed: {ex}")
                return ret

            if self.verbose:
                print(f"QP(osqp_codegen): status={status_val} (failed)")
            return {'success': False}

    def _pack_solution(self, sol, nx, nu, N, ni_f):
        # ---- primal ----
        y = np.asarray(sol['x']).copy().reshape(-1)
        primal_x, primal_u = self.unpack_primal_nominal(y)

        # ---- duals ----
        lam = np.asarray(sol['lam_a']).copy().reshape(-1)

        # OSQP backend has extra last-nx duals for x0 equality rows -> remove them
        if self.backend in ("osqp", "osqp_codegen"):
            lam = lam[:-nx]

        dual_mu_f = lam[-ni_f:]  # terminal ineq duals
        dual_non_term = lam[:-ni_f].reshape(N, nx + self.m.ni)
        dual_mu_stage = dual_non_term[:, nx:].T  # shape (ni, N)

        return {
            'success': True,
            'primal_vec': y,
            'primal_x': primal_x,
            'primal_u': primal_u,
            'dual_vec': lam,
            'dual_mu': dual_mu_stage,
            'dual_mu_f': dual_mu_f,  # 1D; make .reshape(1,-1) if you prefer row
            'cost': float(sol['cost']),
            'time_ms': sol.get('time_ms', None),
        }

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------
    def update_dynamics(self, new_A_list, new_B_list, new_E_list=None, new_g_list=None):
        assert len(new_A_list) == self.N and len(new_B_list) == self.N
        self.A_list = new_A_list
        self.B_list = new_B_list
        self.m.A_list = new_A_list
        self.m.B_list = new_B_list
        if new_E_list is not None:
            assert len(new_E_list) == self.N + 1
            self.E_list = new_E_list
            self.m.E_list = new_E_list
        if new_g_list is not None:
            assert len(new_g_list) == self.N + 1
            self.g_list = new_g_list
            self.m.g_list = new_g_list

        # Always rebuild numerics to allow arbitrary (dense) A/B values
        self._reassemble_numeric_same_sparsity()

        # Backend handling
        if self.backend == "osqp":
            # Rebuild native OSQP problem to accept new sparsity structure
            prob = osqp.OSQP()
            prob.setup(
                P=2 * self.P_mat_csc,
                q=np.zeros(self.P_mat_csc.shape[0]),
                A=self.A_mat_csc_osqp,
                l=self.lbg_osqp_base,
                u=self.ubg_osqp_base,
                polishing=True, warm_starting=False, verbose=False
            )
            self._osqp_prob = prob
        elif self.backend == "osqp_codegen":
            # If CSC pattern/order no longer matches codegen, fall back to native OSQP
            try:
                P_ut = sp.triu(2 * self.P_mat_csc, format='csc')
                P_ut.sort_indices()
                A_osqp = self.A_mat_csc_osqp
                A_osqp.sort_indices()
                self._assert_same_csc_pattern(P_ut, getattr(self, "_P_cg_sig", None), "P_ut")
                self._assert_same_csc_pattern(A_osqp, getattr(self, "_A_cg_sig", None), "A_osqp")
            except Exception as err:
                if self.verbose:
                    print("QP(osqp_codegen): pattern mismatch detected; details below:")
                    print(str(err))
                    print("Falling back to native OSQP backend.")
                self.backend = "osqp"
                prob = osqp.OSQP()
                prob.setup(
                    P=2 * self.P_mat_csc,
                    q=np.zeros(self.P_mat_csc.shape[0]),
                    A=self.A_mat_csc_osqp,
                    l=self.lbg_osqp_base,
                    u=self.ubg_osqp_base,
                    polishing=True, warm_starting=False, verbose=False
                )
                self._osqp_prob = prob
        else:
            # casadi_osqp uses CasADi sparsity; nothing to do here (handled at solve())
            pass

    def update_ubg(self, new_ubg):
        self.ubg = new_ubg

        self.ubg_osqp_base = np.concatenate([np.array(self.ubg).astype(float).ravel(),
                                             np.zeros(self.m.nx)])
    def reset_ubg(self):
        self.ubg = self.nominal_ubg

        self.ubg_osqp_base = np.concatenate([np.array(self.ubg).astype(float).ravel(),
                                             np.zeros(self.m.nx)])

    def reset_lbg(self):
        self.lbg = self.lbg_ZEROS_INF_NOMINAL

        self.lbg_osqp_base = np.concatenate([np.array(self.lbg).astype(float).ravel(),
                                             np.zeros(self.m.nx)])

    def offset_constraints(self, offset_equality_constraints):
        """
        Offset the equality constraints by a given amount, and keeps the inequality constraints unchanged.
        :param offset_equality_constraints:
        :return:
        """
        assert offset_equality_constraints.shape == (self.m.nx, self.N)
        off = ca.DM.zeros((0, 1))
        for k in range(self.N):
            off = ca.vertcat(off, ca.DM(offset_equality_constraints[:, k]), ca.DM.zeros((self.m.ni, 1)))
        off = ca.vertcat(off, ca.DM.zeros((self.m.ni_f, 1)))
        self.ubg = self.ubg - off + EPSILON
        self.lbg = self.lbg_ZEROS_INF_NOMINAL
        self.lbg = self.lbg - off - EPSILON

        return

    def update_q_cost_lin(self, q_cost_lin):
        expected = (self.m.nx + self.m.nu) * self.N + self.m.nx
        assert q_cost_lin.shape[0] == expected
        if not isinstance(q_cost_lin, ca.DM):
            q_cost_lin = ca.DM(q_cost_lin)
        self.q_cost_lin = q_cost_lin

    def add_q_cost_lin(self, q_cost_lin):
        expected = (self.m.nx + self.m.nu) * self.N + self.m.nx
        assert q_cost_lin.shape[0] == expected
        if not isinstance(q_cost_lin, ca.DM):
            q_cost_lin = ca.DM(q_cost_lin)
        self.q_cost_lin += q_cost_lin

    def reset_q_cost_lin(self):
        nx, nu, N = self.m.nx, self.m.nu, self.N
        self.q_cost_lin = ca.DM.zeros((nx + nu) * N + nx)

    def debug_codegen_pattern(self):
        """
        Return a dict with CSC pattern diagnostics comparing current P_ut and A_osqp
        against stored codegen signatures. Useful to pinpoint where the pattern changed.
        """
        info = {}
        if not hasattr(self, "_P_cg_sig") or not hasattr(self, "_A_cg_sig"):
            info["has_signatures"] = False
            return info
        info["has_signatures"] = True

        # current matrices
        P_ut = sp.triu(2 * self.P_mat_csc, format='csc'); P_ut.sort_indices()
        A_osqp = self.A_mat_csc_osqp; A_osqp.sort_indices()

        # summaries
        info["P_now_shape"] = P_ut.shape
        info["A_now_shape"] = A_osqp.shape
        info["P_now_nnz"] = int(P_ut.nnz)
        info["A_now_nnz"] = int(A_osqp.nnz)
        info["P_ref_nnz"] = len(self._P_cg_sig[0]) if self._P_cg_sig else None
        info["A_ref_nnz"] = len(self._A_cg_sig[0]) if self._A_cg_sig else None

        # matches?
        try:
            self._assert_same_csc_pattern(P_ut, self._P_cg_sig, "P_ut")
            info["P_match"] = True
            info["P_report"] = "OK"
        except Exception as e:
            info["P_match"] = False
            info["P_report"] = str(e)
        try:
            self._assert_same_csc_pattern(A_osqp, self._A_cg_sig, "A_osqp")
            info["A_match"] = True
            info["A_report"] = "OK"
        except Exception as e:
            info["A_match"] = False
            info["A_report"] = str(e)

        return info

    def _cg_push_updates(self, cg, Px, Ax, qv, l, u):
        """
        Push P (upper-tri data), A (data), and vectors q,l,u to the codegen module,
        """

        # -------- matrices: P then A --------
        # Use explicit kwargs so our 2 arrays are mapped to the correct slots
        pushed_AP = cg.update_data_mat(P_x=Px, A_x=Ax)
        # todo: only update the diagonal values ... (leverage sparsity)

        # -------- vectors: q, l, u --------
        pushed_ulq = cg.update_data_vec(qv, l, u)

        # Return code 0 means success in OSQP C API
        if pushed_AP != 0:
            # Enrich error with pattern diagnostics if available
            diag = {}
            if hasattr(self, "_P_cg_sig") and hasattr(self, "_A_cg_sig"):
                try:
                    P_ut = sp.triu(2 * self.P_mat_csc, format='csc'); P_ut.sort_indices()
                    A_osqp = self.A_mat_csc_osqp; A_osqp.sort_indices()
                    diag["P"] = self._csc_diff_report(P_ut, self._P_cg_sig, "P_ut")
                    diag["A"] = self._csc_diff_report(A_osqp, self._A_cg_sig, "A_osqp")
                except Exception as e:
                    diag["error"] = f"diag collection failed: {e}"
            raise RuntimeError(f"update_data_mat failed with code {pushed_AP}. diag={diag}")
        if pushed_ulq != 0:
            raise RuntimeError(f"update_data_vec failed with code {pushed_ulq}")

    def _store_cg_signatures(self, P_ut: sp.csc_matrix, A_osqp: sp.csc_matrix):
        # Both must be sorted before storing
        P_ut.sort_indices()
        A_osqp.sort_indices()
        self._P_cg_sig = (tuple(P_ut.indices), tuple(P_ut.indptr))
        self._A_cg_sig = (tuple(A_osqp.indices), tuple(A_osqp.indptr))

    def _assert_same_csc_pattern(self, M: sp.csc_matrix, sig, name: str):
        M.sort_indices()
        # If no reference signature, nothing to compare against
        if sig is None or sig[0] is None or sig[1] is None:
            return
        if (tuple(M.indices), tuple(M.indptr)) != sig:
            # Build a detailed diff to aid debugging
            raise RuntimeError(self._csc_diff_report(M, sig, name))

    def _csc_diff_report(self, M: sp.csc_matrix, sig, name: str) -> str:
        """
        Create a detailed human-readable report of how M's CSC pattern differs from the
        stored reference signature (indices, indptr). Highlights first differing column,
        per-column nnz and row-index mismatches, and overall stats.
        """
        idx_ref, ptr_ref = sig
        idx_now = tuple(M.indices)
        ptr_now = tuple(M.indptr)
        ncols = M.shape[1]
        cols_ref = len(ptr_ref) - 1
        cols_now = len(ptr_now) - 1
        nnz_ref = len(idx_ref)
        nnz_now = len(idx_now)

        lines = []
        lines.append(f"{name} CSC pattern/order changed vs. codegen")
        lines.append(f"  shape_now={M.shape}, ncols_now={ncols}")
        lines.append(f"  ref: cols={cols_ref}, nnz={nnz_ref}")
        lines.append(f"  now: cols={cols_now}, nnz={nnz_now}")

        # Column pointer differences
        max_cols = min(cols_ref, cols_now)
        first_col_ptr_diff = None
        for j in range(max_cols + 1):  # include last pointer
            pr = ptr_ref[j] if j < len(ptr_ref) else None
            pn = ptr_now[j] if j < len(ptr_now) else None
            if pr != pn:
                first_col_ptr_diff = j
                break
        if first_col_ptr_diff is not None:
            j = first_col_ptr_diff
            pr = ptr_ref[j] if j < len(ptr_ref) else None
            pn = ptr_now[j] if j < len(ptr_now) else None
            lines.append(f"  first indptr mismatch at j={j}: ref={pr}, now={pn}")
        else:
            lines.append("  indptr arrays identical")

        # If we can align columns, find first column with row-index diff
        first_col_idx_diff = None
        j_limit = min(cols_ref, cols_now)
        for j in range(j_limit):
            r0_ref, r1_ref = ptr_ref[j], ptr_ref[j + 1]
            r0_now, r1_now = ptr_now[j], ptr_now[j + 1]
            col_idx_ref = idx_ref[r0_ref:r1_ref]
            col_idx_now = idx_now[r0_now:r1_now]
            if tuple(col_idx_ref) != tuple(col_idx_now):
                first_col_idx_diff = j
                break
        if first_col_idx_diff is not None:
            j = first_col_idx_diff
            r0_ref, r1_ref = ptr_ref[j], ptr_ref[j + 1]
            r0_now, r1_now = ptr_now[j], ptr_now[j + 1]
            col_idx_ref = tuple(idx_ref[r0_ref:r1_ref])
            col_idx_now = tuple(idx_now[r0_now:r1_now])
            lines.append(f"  first column row-index mismatch at j={j}:")
            lines.append(f"    ref rows[{r0_ref}:{r1_ref}] len={len(col_idx_ref)} -> {list(col_idx_ref)[:20]}")
            lines.append(f"    now rows[{r0_now}:{r1_now}] len={len(col_idx_now)} -> {list(col_idx_now)[:20]}")
        else:
            lines.append("  per-column row indices identical for compared columns")

        # Also include a short checksum of arrays for quick compare
        import hashlib
        def _hs(x):
            h = hashlib.sha1(bytes(str(x), 'utf-8')).hexdigest()[:12]
            return h
        lines.append(f"  checksum idx_ref={_hs(idx_ref)}, idx_now={_hs(idx_now)}")
        lines.append(f"  checksum ptr_ref={_hs(ptr_ref)}, ptr_now={_hs(ptr_now)}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Codegen export helpers
    # ------------------------------------------------------------------


    def export_osqp_solver(self, out_dir, *, extension_name="osqp_generated"):
        out = Path(out_dir)
        # nuke existing folder to avoid copytree errors (simplest + robust)
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)
        self.P_mat_csc.sort_indices()
        self.A_mat_csc_osqp.sort_indices()

        P = 2 * self.P_mat_csc
        A = self.A_mat_csc_osqp
        q = np.zeros(P.shape[0])
        l = self.lbg_osqp_base
        u = self.ubg_osqp_base
        self.P_mat_csc.sort_indices()
        self.A_mat_csc_osqp.sort_indices()

        P_ut = sp.triu(2 * self.P_mat_csc, format='csc')
        P_ut.sort_indices()

        prob = osqp.OSQP()
        prob.setup(P=P_ut, q=q, A=self.A_mat_csc_osqp, l=l, u=u, polish=True, warm_start=True, verbose=False)
        # prob.setup(
        #     P=2 * self.P_mat_csc,  # match CasADi’s h = 2*H
        #     q=np.zeros(self.P_mat_csc.shape[0]),
        #     A=self.A_mat_csc_osqp,  # includes x0 rows
        #     l=self.lbg_osqp_base,  # placeholders; updated per solve
        #     u=self.ubg_osqp_base,
        #     polish=True, warm_start=False, verbose=False
        # )

        # import os, sys
        # os.environ["PYBIND11_FINDPYTHON"] = "NEW"
        # os.environ["CMAKE_ARGS"] = f"-DPython3_EXECUTABLE={sys.executable}"

        # OSQP ≥ 1.0 API
        prob.codegen(
            str(out),
            parameters="matrices",
            extension_name=extension_name,
            include_codegen_src=True,
            force_rewrite=True,  # <-- key
            compile=False,  # builds the Python module
            profiling_enable=False, # !!! profiling is not supported in codegen mode
        )
        if self.verbose:
            print(f"Built {extension_name} in {out}. You can now `import {extension_name}`.")

    # ------------------------------------------------------------------
    # Quadprog export helpers
    # ------------------------------------------------------------------
    def _convert_bounds_to_quadprog(self, A: sp.csc_matrix, l: np.ndarray, u: np.ndarray, tol_eq: float = 1e-8):
        """
        Convert l <= A x <= u into quadprog's A,b,Aeq,beq.
        Returns (A_ineq, b_ineq, A_eq, b_eq) as CSR matrices/vectors.
        """
        A = A.tocsr()
        m, n = A.shape
        idx = np.arange(m)

        is_l_fin = np.isfinite(l)
        is_u_fin = np.isfinite(u)
        is_eq = is_l_fin & is_u_fin & (np.abs(u - l) <= tol_eq)

        # Equality rows
        Aeq = A[is_eq]
        beq = 0.5 * (u[is_eq] + l[is_eq])

        # Inequalities
        # upper-only: A x <= u
        mask_up = (~is_l_fin) & is_u_fin
        A1 = A[mask_up]
        b1 = u[mask_up]

        # lower-only: -A x <= -l
        mask_lo = is_l_fin & (~is_u_fin)
        A2 = (-A[mask_lo])
        b2 = (-l[mask_lo])

        # double-bounded (not equal): both A x <= u and -A x <= -l
        mask_db = is_l_fin & is_u_fin & (~is_eq)
        A3a = A[mask_db]
        b3a = u[mask_db]
        A3b = (-A[mask_db])
        b3b = (-l[mask_db])

        # stack
        A_ineq = sp.vstack([A1, A2, A3a, A3b], format='csr') if (A1.shape[0] + A2.shape[0] + A3a.shape[0] + A3b.shape[0]) > 0 else sp.csr_matrix((0, n))
        b_ineq = np.concatenate([b1, b2, b3a, b3b]) if A_ineq.shape[0] > 0 else np.zeros((0,))

        return A_ineq, b_ineq, Aeq, beq

    def _export_quadprog(self, x0, solve_ret: dict):
        """
        Build quadprog inputs (H,f,A,b,Aeq,beq,lb,ub) from the current QP and save to .mat.
        Includes the solution vector and trajectories.
        """
        # Ensure export dir exists
        self.export_dir.mkdir(parents=True, exist_ok=True)

        # Matrices and vectors
        H = (2 * self.P_mat_csc).tocsc()  # quadprog uses 1/2 x' H x
        f = np.asarray(self.q_cost_lin, dtype=float).ravel()

        # Constraints in OSQP form (without x0 helper rows)
        A = self.A_mat_csc.tocsc()
        l = np.asarray(self.lbg, dtype=float).ravel()
        u = np.asarray(self.ubg, dtype=float).ravel()

        # Convert to quadprog form
        A_ineq, b_ineq, Aeq_base, beq_base = self._convert_bounds_to_quadprog(A, l, u)

        # Append x0 equality: I * x(0) + x0 = 0  => I*y = -x0 on first nx vars
        nx, nu, N = self.m.nx, self.m.nu, self.N
        nv = (nx + nu) * N + nx
        I0 = sp.hstack([sp.eye(nx, format='csr'), sp.csr_matrix((nx, nv - nx))], format='csr')
        Aeq = sp.vstack([Aeq_base, I0], format='csr') if Aeq_base.shape[0] > 0 else I0
        beq = np.concatenate([beq_base, -np.asarray(x0, dtype=float).ravel()]) if Aeq_base.shape[0] > 0 else (-np.asarray(x0, dtype=float).ravel())

        # Variable bounds: leave free (±inf) — x0 equality is already in Aeq
        lb = -np.inf * np.ones(nv)
        ub = +np.inf * np.ones(nv)

        # Solution info
        y = np.asarray(solve_ret.get('primal_vec')).ravel()
        x_traj = np.asarray(solve_ret.get('primal_x'))
        u_traj = np.asarray(solve_ret.get('primal_u'))
        cost = float(solve_ret.get('cost', np.nan))

        # Save
        k = self._export_counter
        self._export_counter += 1
        out_path = self.export_dir / f"qp_export_{k:06d}.mat"
        savemat(str(out_path), {
            'H': H,
            'f': f,
            'A': A_ineq,
            'b': b_ineq,
            'Aeq': Aeq,
            'beq': beq,
            'lb': lb,
            'ub': ub,
            'x0': np.asarray(x0, dtype=float).ravel(),
            'x_sol': y,
            'x_traj': x_traj,
            'u_traj': u_traj,
            'cost': cost,
            'backend': np.array(self.backend),
            'dimensions': np.array([nx, nu, N], dtype=np.int32),
        })
        if self.verbose:
            print(f"Saved quadprog export to {out_path}")
