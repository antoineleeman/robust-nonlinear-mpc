from dyn.rockETH.rockETH import RockETH
from solver.SCP_SLS_jit import SCP_SLS as SCP_SLS_impl
from solver.nlp_soft_constraints import NLPSoftConstraints

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from itertools import cycle
from matplotlib.lines import Line2D

folder_name = "rockETH_robust_vs_soft_closed_loop"
os.makedirs(folder_name, exist_ok=True)


def _affine_to_unit(x, lb, ub):
    denom = (ub - lb) if np.all(ub - lb != 0) else 1.0
    return 2.0 * (x - lb) / denom - 1.0


def compact_dual_legend(ax, title=None, ncol=3):
    h, l = ax.get_legend_handles_labels()
    hv = [hi for hi, li in zip(h, l) if "(robust)" in li]
    lv = [li.replace(" (robust)", "") for li in l if "(robust)" in li]

    leg1 = ax.legend(
        hv, lv, title=title, loc="upper left", ncol=ncol,
        handlelength=1.0, handletextpad=0.3, columnspacing=0.6,
        labelspacing=0.2, borderpad=0.3, framealpha=0.8,
    )
    ax.add_artist(leg1)

    ax.legend(
        [Line2D([], [], linestyle='-', color='0.3'),
         Line2D([], [], linestyle='--', color='0.3')],
        ["robust", "soft"],
        loc="upper right", frameon=True, handlelength=1.2, handletextpad=0.3,
    )


def _configure_model(dt):
    m = RockETH()
    m.dt = dt
    # Disturbance scaling consistent with both scripts
    sigma_theta = np.deg2rad(2.0)
    q_vec_std = 0.5 * sigma_theta
    q_w_std = 0.1 * q_vec_std
    m.E = m.dt * np.diag([
        0.20, 0.20, 0.20,  # pos [m]
        0.20, 0.20, 0.20,  # vel [m/s]
        q_vec_std, q_vec_std, q_vec_std, q_w_std,  # quat
        0.20, 0.20, 0.20,  # angular vel [rad/s]
        0.8,  # thrust magnitude
        0.2,  # torque x
        0.04, 0.04  # servo angles [rad]
    ])
    return m


def _cost_matrices():
    Q = np.diag([
        10.0, 10.0, 10.0,    # x, y, z
        1.0, 1.0, 1.0,       # vx, vy, vz
        1.0, 1.0, 1.0, 1.0,  # qx, qy, qz, qw
        1.0, 5.0, 5.0,       # omega_x, omega_y, omega_z
        1.0, 1.0, 1.0, 1.0   # dT, dtorque, dservo1, dservo2
    ])
    R = np.diag([1.0, 1.0, 1.0, 1.0])
    Qf = 10 * Q
    return Q, R, Qf


def _fixed_initial_condition():
    # Use the same fixed x0 as in both originals to enforce identical starts
    return np.array([1.75729,
                    4.15951,
                    4.72757,
                    -0.18913,
                    -0.38367,
                    -0.08697,
                    -0.79487,
                    0.00768,
                    -0.21110,
                    -0.56883,
                    -0.12752,
                    -0.58026,
                    -0.76542,
                    0.20555,
                    0.54610,
                    -0.40116,
                    -0.35401])


def _compute_closed_loop_cost(X_all, U_all, Q, R, Qf):
    # X_all: (nx, T), U_all: (nu, T-1)
    T = X_all.shape[1]
    J = 0.0
    for t in range(T - 1):
        J += float(X_all[:, t].T @ Q @ X_all[:, t]) + float(U_all[:, t].T @ R @ U_all[:, t])
    J_terminal = float(X_all[:, -1].T @ Qf @ X_all[:, -1])
    return J, J_terminal, J + J_terminal


def _run_robust(N, Q, R, Qf, dt, x0, W):
    m = _configure_model(dt)
    nx, nu = m.nx, m.nu

    solver = SCP_SLS_impl(
        N, Q, R, m, Qf,
        Q_reg=1e4 * np.eye(m.nx),
        R_reg=1e4 * np.eye(m.nu),
        Q_reg_f=1e4 * np.eye(m.nx),
        rti=1,
        fast_sls_rti_steps=1,
    )
    # Quiet inner solver by default; toggle if needed
    try:
        solver.fast_SLS_solver.solver_forward.export_standard_qp = False
        solver.fast_SLS_solver.verbose = False
        solver.fast_SLS_solver.solver_forward.verbose = False
        solver.fast_SLS_solver.CONV_EPS = 1e-4
    except Exception:
        pass
    solver.verbose = False
    solver.epsilon_convergence = 1e-3

    # steps disturbances => T = steps + 1 states
    steps = W.shape[0]
    T = steps + 1
    X = np.zeros((nx, T))
    U = np.zeros((nu, steps))

    # For logging nominal/backoff trajectories per MPC step
    Xn = np.zeros((nx, N + 1, steps))
    Un = np.zeros((nu, N, steps))
    backoff_x = np.zeros((nx, N + 1, steps))
    backoff_u = np.zeros((nu, N, steps))

    x = x0.copy().reshape(-1)
    X[:, 0] = x

    for i in range(steps):
        if i > 0 and hasattr(solver, 'reset_warm_start'):
            solver.reset_warm_start()

        sol = solver.solve(x)
        # Store nominal/backoff for this MPC step
        Xn[:, :, i] = sol['primal_x']
        Un[:, :, i] = sol['primal_u']
        if 'backoff_x' in sol:
            backoff_x[:, :, i] = sol['backoff_x'].T
        if 'backoff_u' in sol:
            backoff_u[:, :, i] = sol['backoff_u'].T

        # Apply first control and propagate
        u0 = sol['primal_u'][:, 0]
        U[:, i] = u0
        x = np.array(m.ddyn(x, u0) + m.E @ W[i]).reshape(-1)
        X[:, i + 1] = x

    return dict(
        state_trajectory=X,
        input_trajectory=U,
        nominal_trajectory_x=Xn,
        nominal_trajectory_u=Un,
        backoff_trajectory_x=backoff_x,
        backoff_trajectory_u=backoff_u,
        g=m.g,
        nx=m.nx,
        nu=m.nu,
        dt=m.dt,
        N=N,
    )


def _run_soft(N, Q, R, Qf, dt, x0, W):
    m = _configure_model(dt)
    nx, nu = m.nx, m.nu

    solver = NLPSoftConstraints(N, Q, R, m, Qf, rho_soft=1e6, rho_soft_l1=1e6)

    # steps disturbances => T = steps + 1 states
    steps = W.shape[0]
    T = steps + 1
    X = np.zeros((nx, T))
    U = np.zeros((nu, steps))

    Xn = np.zeros((nx, N + 1, steps))
    Un = np.zeros((nu, N, steps))
    backoff_x = np.zeros((nx, N + 1, steps))  # zeros for compatibility
    backoff_u = np.zeros((nu, N, steps))

    x = x0.copy().reshape(-1)
    X[:, 0] = x

    for i in range(steps):
        sol = solver.solve(x)
        if not sol or not sol.get('success', False):
            raise RuntimeError(f"Soft-constrained NLP failed at step {i}.")

        Xn[:, :, i] = sol['primal_x']
        Un[:, :, i] = sol['primal_u']

        # Apply first control and propagate
        u0 = sol['primal_u'][:, 0]
        U[:, i] = u0
        x = np.array(m.ddyn(x, u0) + m.E @ W[i]).reshape(-1)
        X[:, i + 1] = x

    return dict(
        state_trajectory=X,
        input_trajectory=U,
        nominal_trajectory_x=Xn,
        nominal_trajectory_u=Un,
        backoff_trajectory_x=backoff_x,
        backoff_trajectory_u=backoff_u,
        g=m.g,
        nx=m.nx,
        nu=m.nu,
        dt=m.dt,
        N=N,
    )


def generate():
    # Common setup
    N = 15
    dt = 0.05
    T = 30
    Q, R, Qf = _cost_matrices()

    # Identical initial condition
    x0 = _fixed_initial_condition()

    # Identical disturbance sequence for both controllers
    rng = np.random.default_rng(123)
    # Uniform disturbances in [-1, 1] for each state at each step
    # Shape (T, nx). We'll infer nx from RockETH once to make W.
    nx_tmp = RockETH().nx
    #W = rng.uniform(-1.0, 1.0, size=(T - 1, nx_tmp))  # T-1 disturbances between steps
    W = -0.8*np.ones((T - 1, nx_tmp))

    # Run controllers
    robust_res = _run_robust(N, Q, R, Qf, dt, x0, W)
    soft_res = _run_soft(N, Q, R, Qf, dt, x0, W)

    # Costs
    Jr_s, Jr_T, Jr = _compute_closed_loop_cost(
        robust_res['state_trajectory'], robust_res['input_trajectory'], Q, R, Qf
    )
    Js_s, Js_T, Js = _compute_closed_loop_cost(
        soft_res['state_trajectory'], soft_res['input_trajectory'], Q, R, Qf
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(folder_name, f"rockETH_compare_closed_loop_{timestamp}.npz")

    np.savez(
        filename,
        # robust
        r_state_trajectory=robust_res['state_trajectory'],
        r_input_trajectory=robust_res['input_trajectory'],
        r_nominal_trajectory_x=robust_res['nominal_trajectory_x'],
        r_nominal_trajectory_u=robust_res['nominal_trajectory_u'],
        r_backoff_trajectory_x=robust_res['backoff_trajectory_x'],
        r_backoff_trajectory_u=robust_res['backoff_trajectory_u'],
        # soft
        s_state_trajectory=soft_res['state_trajectory'],
        s_input_trajectory=soft_res['input_trajectory'],
        s_nominal_trajectory_x=soft_res['nominal_trajectory_x'],
        s_nominal_trajectory_u=soft_res['nominal_trajectory_u'],
        s_backoff_trajectory_x=soft_res['backoff_trajectory_x'],
        s_backoff_trajectory_u=soft_res['backoff_trajectory_u'],
        # common
        dt=dt,
        g=robust_res['g'],
        nx=robust_res['nx'],
        nu=robust_res['nu'],
        simulation_time_steps=T,
        N=N,
        x0=x0,
        W=W,
        # costs
        Jr_stage=Jr_s, Jr_terminal=Jr_T, Jr_total=Jr,
        Js_stage=Js_s, Js_terminal=Js_T, Js_total=Js,
    )

    print(f"Saved comparison results to {filename}")


def plot():
    # Load latest comparison file and overlay closed-loop trajectories
    files = [f for f in os.listdir(folder_name) if f.endswith('.npz')]
    if not files:
        print("No data files found in the directory.")
        return

    latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(folder_name, x)))
    data = np.load(os.path.join(folder_name, latest_file))

    g = data['g']
    nx = int(data['nx'])
    nu = int(data['nu'])
    dt = float(data['dt'])
    N = int(data['N'])
    T = int(data['simulation_time_steps'])

    Xr = data['r_state_trajectory']
    Ur = data['r_input_trajectory']
    Xs = data['s_state_trajectory']
    Us = data['s_input_trajectory']

    ub_x = g[:nx]
    ub_u = g[nx:nx + nu]
    lb_x = -g[nx + nu: nx + nu + nx]
    lb_u = -g[nx + nu + nx: nx + nu + nx + nu]

    plt.rcParams.update({
        "font.size": 18,
        "axes.labelsize": 18,
        "legend.fontsize": 15,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    })

    fig, axs = plt.subplots(2, 3, figsize=(20, 10), sharex=True)
    axs = axs.flatten()

    labels = [
        [r"$x$", r"$y$", r"$z$"],
        [r"$v_{x}$", r"$v_{y}$", r"$v_{z}$"],
        [r"$q_{x}$", r"$q_{y}$", r"$q_{z}$", r"$q_{w}$"],
        [r"$\omega_{x}$", r"$\omega_{y}$", r"$\omega_{z}$"],
        [r"$T$", r"$\tau$", r"$\theta_{1}$", r"$\theta_{2}$"]
    ]
    indices = [
        range(0, 3),
        range(3, 6),
        range(6, 10),
        range(10, 13),
        range(13, 17),
    ]
    input_labels = [r"$T_{in}$", r"$\tau_{in}$", r"$\theta_{1,in}$", r"$\theta_{2,in}$"]

    viridis = plt.cm.viridis
    grid_kw = dict(alpha=0.3, linestyle="--")

    time_all =np.arange(Xr.shape[1]) * dt
    time_u = time_all[:-1]
    time_cons = np.arange(Xr.shape[1] + N) * dt

    # States panels (overlay robust vs soft)
    for k, (ax, idxs, lbls) in enumerate(zip(axs[:5], indices, labels)):
        colors = viridis(np.linspace(0.3, 0.7, len(idxs)))
        ls_r = cycle(["-"] * 4)
        ls_s = cycle(["--"] * 4)

        if k == 4:  # normalized inputs-like states (actuators)
            for idx, label, color in zip(idxs, lbls, colors):
                ax.plot(time_all, _affine_to_unit(Xr[idx], lb_x[idx], ub_x[idx]),
                        label=f"{label} (robust)", linewidth=2.5, color=color, linestyle=next(ls_r))
                ax.plot(time_all, _affine_to_unit(Xs[idx], lb_x[idx], ub_x[idx]),
                        label=f"{label} (soft)", linewidth=2.5, color=color, linestyle=next(ls_s))
            ax.hlines([-1, 1], time_cons[0], time_cons[-1], colors='red', linestyles=[':'], linewidth=2.5)
            ax.set_ylim(-1.1, 1.1)
        else:
            for idx, label, color in zip(idxs, lbls, colors):
                ax.plot(time_all, Xr[idx], label=f"{label} (robust)", linewidth=2.5, color=color, linestyle=next(ls_r))
                ax.plot(time_all, Xs[idx], label=f"{label} (soft)", linewidth=2.5, color=color, linestyle=next(ls_s))
                if k not in (0, 2):
                    ax.hlines([lb_x[idx], ub_x[idx]], time_cons[0], time_cons[-1], colors="red",
                              linestyles=[':'], linewidth=2.5)
        ax.set_ylabel(["Position [m]", "Velocity [m/s]", "Quaternion [-]",
                       r"Angular vel. [rad/s]", "Thrust/torques (norm.) [-]"][k])
        ax.grid(True, **grid_kw)
        ax.legend(loc="best", ncol=2)

    # Inputs (normalized), overlay
    ax_u = axs[-1]
    colors_u = viridis(np.linspace(0.3, 0.7, len(input_labels)))
    ls_r = cycle(["-"] * 4)
    ls_s = cycle(["--"] * 4)
    for j, (label, color) in enumerate(zip(input_labels, colors_u)):
        ax_u.plot(time_u, _affine_to_unit(Ur[j], lb_u[j], ub_u[j]),
                  label=f"{label} (robust)", linewidth=2.5, color=color, linestyle=next(ls_r))
        ax_u.plot(time_u, _affine_to_unit(Us[j], lb_u[j], ub_u[j]),
                  label=f"{label} (soft)", linewidth=2.5, color=color, linestyle=next(ls_s))
    ax_u.hlines([-1, 1], time_u[0], time_u[-1], colors='red', linestyles=[ ':'], linewidth=2.5)
    ax_u.set_ylim(-1.1, 1.1)
    ax_u.set_ylabel("Inputs (norm.) [-]")
    ax_u.grid(True, **grid_kw)
    ax_u.legend(loc="best", ncol=2)

    axs[4].set_xlabel("Time [s]")
    axs[5].set_xlabel("Time [s]")

    plt.tight_layout(pad=1.4)
    out_pdf = os.path.join(folder_name, "trajectory_plot_compare_closed_loop.pdf")
    plt.savefig(out_pdf, format="pdf", dpi=600, bbox_inches="tight")
    plt.show()

    # Print costs summary
    print("Robust total cost:", float(data['Jr_total']))
    print("  stage:", float(data['Jr_stage']), "terminal:", float(data['Jr_terminal']))
    print("Soft total cost:", float(data['Js_total']))
    print("  stage:", float(data['Js_stage']), "terminal:", float(data['Js_terminal']))


def plot_vel_omega_inputs():
    """Plot velocity (v) and angular velocity (omega) for robust vs soft, with compact dual legends."""
    files = [f for f in os.listdir(folder_name) if f.endswith('.npz')]
    if not files:
        print("No data files found in the directory.")
        return

    latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(folder_name, x)))
    data = np.load(os.path.join(folder_name, latest_file))

    g = data['g']
    nx = int(data['nx'])
    nu = int(data['nu'])
    dt = float(data['dt'])
    N = int(data['N'])

    Xr = data['r_state_trajectory']
    Ur = data['r_input_trajectory']
    Xs = data['s_state_trajectory']
    Us = data['s_input_trajectory']

    ub_x = g[:nx]
    ub_u = g[nx:nx + nu]
    lb_x = -g[nx + nu: nx + nu + nx]
    lb_u = -g[nx + nu + nx: nx + nu + nx + nu]

    plt.rcParams.update({
        "font.size": 18,
        "axes.labelsize": 18,
        "legend.fontsize": 14,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    })

    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=False)

    viridis = plt.cm.viridis
    grid_kw = dict(alpha=0.3, linestyle="--")

    time_all = np.arange(Xr.shape[1]) * dt

    # Panel 1: Velocity (vx, vy, vz)
    ax_v = axs[0]
    colors_v = viridis(np.linspace(0.3, 0.7, 3))
    ls_r = cycle(["-"] * 3)
    ls_s = cycle(["--"] * 3)
    vel_labels = [r"$v_x$", r"$v_y$", r"$v_z$"]
    for j, idx in enumerate(range(3, 6)):
        color = colors_v[j]
        label = vel_labels[j]
        ax_v.plot(time_all, Xr[idx], label=f"{label} (robust)", linewidth=2.5, color=color, linestyle=next(ls_r))
        ax_v.plot(time_all, Xs[idx], label=f"{label} (soft)", linewidth=2.5, color=color, linestyle=next(ls_s))
        ax_v.hlines([lb_x[idx], ub_x[idx]], time_all[0], time_all[-1], colors="red",
                    linestyles=[':'], linewidth=2.5)
    ax_v.set_ylabel("Velocity [m/s]")
    ax_v.set_xlabel("Time [s]")
    ax_v.grid(True, **grid_kw)
    compact_dual_legend(ax_v, ncol=3)

    # Panel 2: Angular velocity (omega_x, omega_y, omega_z)
    ax_w = axs[1]
    omega_labels = [r"$\omega_{x}$", r"$\omega_{y}$", r"$\omega_{z}$"]
    colors_w = viridis(np.linspace(0.3, 0.7, len(omega_labels)))
    ls_r = cycle(["-"] * len(omega_labels))
    ls_s = cycle(["--"] * len(omega_labels))
    for j, (idx, (label, color)) in enumerate(zip(range(10, 13), zip(omega_labels, colors_w))):
        ax_w.plot(time_all, Xr[idx], label=f"{label} (robust)", linewidth=2.5, color=color, linestyle=next(ls_r))
        ax_w.plot(time_all, Xs[idx], label=f"{label} (soft)", linewidth=2.5, color=color, linestyle=next(ls_s))
        ax_w.hlines([lb_x[idx], ub_x[idx]], time_all[0], time_all[-1], colors="red",
                    linestyles=[':'], linewidth=2.5)
    ax_w.set_ylabel(r"Angular vel. [rad/s]")
    ax_w.set_xlabel("Time [s]")
    ax_w.grid(True, **grid_kw)
    compact_dual_legend(ax_w, ncol=3)

    plt.tight_layout(pad=1.2)
    out_pdf = os.path.join(folder_name, "trajectory_plot_compare_vel_omega.pdf")
    plt.savefig(out_pdf, format="pdf", dpi=600, bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    optimize = False  # set True to run both controllers and save results
    if optimize:
        generate()
    # plot()
    plot_vel_omega_inputs()
