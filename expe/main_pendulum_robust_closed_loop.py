from dyn.pendulum import Pendulum

import numpy as np
import matplotlib.pyplot as plt
import os
from util.footnote import add_footnote_time
import argparse
from datetime import datetime

# Prefer the JIT solver (with warm-start support); fallback to non-JIT if not available
try:
    from solver.SCP_SLS_jit import SCP_SLS as SCP_SLS_impl
    _HAS_JIT = True
except Exception:
    from solver.SCP_SLS import SCP_SLS as SCP_SLS_impl
    _HAS_JIT = False

folder_name = "pendulum_robust_closed_loop"
os.makedirs(folder_name, exist_ok=True)

# fix the random seed for reproducibility
np.random.seed(0)

def generate(N: int | None = None):
    name = "main_pendulum_robust_closed_loop"
    m = Pendulum()
    Q = np.eye(m.nx)
    R = np.eye(m.nu)
    Qf = 10*np.eye(m.nx)
    N = int(N) if N is not None else 15
    m.E = 0.003 * np.eye(m.nx)
    m.dt = 0.05

    x_max = 10 * np.ones(m.nx)
    x_min = -10 * np.ones(m.nx)
    u_max = 5 * np.ones(m.nu)
    u_min = -5 * np.ones(m.nu)
    x_max_f = 10 * np.ones(m.nx)
    x_min_f = -10 * np.ones(m.nx)
    m.replace_constraints(x_max, x_min, u_max, u_min, x_max_f, x_min_f)

    scp_sls_solver = SCP_SLS_impl(N, Q, R, m, Qf,
                                  Q_reg = 1e3 * np.eye(m.nx),
                                  R_reg = 1e3 * np.eye(m.nu),
                                  Q_reg_f = 1e4 * np.eye(m.nx),
                                  rti = 3,
                                  fast_sls_rti_steps=2,
                                  )

    scp_sls_solver.epsilon_convergence = 1e-10
    scp_sls_solver.fast_SLS_solver.verbose = True
    scp_sls_solver.fast_SLS_solver.solver_forward.verbose = False
    rng = np.random.default_rng()
    # derive state bounds from m.g
    ub_x = m.g[:m.nx]
    lb_x = -m.g[m.nx + m.nu: m.nx + m.nu + m.nx]
    x0 = rng.uniform(lb_x / 10, ub_x / 10)
    x0 = np.array([0.5, 0.5, 0.0, 0.0])  # specific initial condition

    simulation_time_steps = 60
    state_trajectory = np.zeros((m.nx, simulation_time_steps))
    state_trajectory[:, 0] = x0
    input_trajectory = np.zeros((m.nu, simulation_time_steps - 1))

    backoff_trajectory_x = np.zeros((m.nx, N + 1, simulation_time_steps))
    backoff_trajectory_u = np.zeros((m.nu, N, simulation_time_steps))
    nominal_trajectory_x = np.zeros((m.nx, N + 1, simulation_time_steps))
    nominal_trajectory_u = np.zeros((m.nu, N, simulation_time_steps))
    t_jac = np.zeros((simulation_time_steps,1))
    t_qp = np.zeros((simulation_time_steps,1))
    t_riccati = np.zeros((simulation_time_steps,1))


    for i in range(simulation_time_steps):
        if i > 0 and hasattr(scp_sls_solver, 'reset_warm_start'):
            scp_sls_solver.reset_warm_start()
        print(f"[Pendulum] Step {i+1}/{simulation_time_steps}: solving ...")
        solution_scp_sls = scp_sls_solver.solve(x0)
        tj = float(solution_scp_sls.get('t_jac_ms', np.nan))
        tq = float(solution_scp_sls.get('t_qp_ms', np.nan))
        tb = float(solution_scp_sls.get('t_backward_ms', np.nan))
        t_jac[i] = tj
        t_qp[i] = tq
        t_riccati[i] = tb
        if np.isnan(tb):
            print("x is NaN")
        print(f"[Pendulum]   -> done (Δt: jac={solution_scp_sls.get('t_jac_ms', 0.0):.2f} ms, qp={solution_scp_sls.get('t_qp_ms', 0.0):.2f} ms, riccati={solution_scp_sls.get('t_backward_ms', 0.0):.2f} ms)")
        backoff_trajectory_x[:, :, i] = solution_scp_sls['backoff_x'].T
        backoff_trajectory_u[:, :, i] = solution_scp_sls['backoff_u'].T
        nominal_trajectory_x[:, :, i] = solution_scp_sls['primal_x']
        nominal_trajectory_u[:, :, i] = solution_scp_sls['primal_u']


        if i < simulation_time_steps - 1:
            input_trajectory[:, i] = solution_scp_sls['primal_u'][:, 0]
            x0 = m.ddyn(x0, solution_scp_sls['primal_u'][:, 0])
            state_trajectory[:, i + 1] = np.array(x0).reshape(-1)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(folder_name, f"pendulum_robust_closed_loop_{timestamp}.npz")
    np.savez(
        filename,
        state_trajectory=state_trajectory,
        input_trajectory=input_trajectory,
        nominal_trajectory_x=nominal_trajectory_x,
        nominal_trajectory_u=nominal_trajectory_u,
        backoff_trajectory_x=backoff_trajectory_x,
        backoff_trajectory_u=backoff_trajectory_u,
        dt=m.dt,
        g=m.g,
        nx=m.nx,
        nu=m.nu,
        simulation_time_steps=simulation_time_steps,
        N=N,
        # timings (ms)
        t_jac=t_jac,
        t_qp=t_qp,
        t_riccati=t_riccati,
    )
    print(f"Results saved to {filename}")
    return filename

def plot():
    files = [f for f in os.listdir(folder_name) if f.endswith('.npz')]
    if not files:
        print("No data files found in the directory.")
        return
    latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(folder_name, x)))
    solution = np.load(os.path.join(folder_name, latest_file))
    m = Pendulum()
    fig, ax = plt.subplots(1, 2, figsize=(15, 8))
    m.plot_nominal_trajectory(solution['state_trajectory'], ax=ax[0])
    ax[0].grid(True)
    ax[0].set_title('State trajectory')
    m.plot_input_nominal_trajectory(solution['input_trajectory'], ax=ax[1])
    ax[1].grid(True)
    ax[1].set_title('Input trajectory')
    add_footnote_time(plt)
    plt.tight_layout()
    out_png = os.path.join(folder_name, "trajectory.png")
    plt.savefig(out_png, dpi=300)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pendulum robust closed-loop experiment")
    parser.add_argument('--run', action='store_true', help='Run the closed-loop simulation (generate data)')
    parser.add_argument('--N', type=int, default=None, help='Horizon length (overrides default if provided)')
    args = parser.parse_args()
    if args.run:
        generate(args.N)
    else:
        plot()