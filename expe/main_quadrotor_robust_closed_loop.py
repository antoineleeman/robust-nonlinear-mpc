from dyn.quadrotor import Quadrotor

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from matplotlib.lines import Line2D
from itertools import cycle
from matplotlib.patches import Polygon
import matplotlib.colors as mcolors
import argparse

# Output folder
folder_name = "quadrotor_robust_closed_loop"
os.makedirs(folder_name, exist_ok=True)

# Prefer the JIT solver (with warm-start support)
from solver.SCP_SLS_jit import SCP_SLS as SCP_SLS_impl


def _affine_to_unit(x, lb, ub):
    denom = (ub - lb) if np.all(ub - lb != 0) else 1.0
    return 2.0 * (x - lb) / denom - 1.0


def _tube_halfwidth_to_unit(halfw, lb, ub):
    denom = (ub - lb) if (ub - lb) != 0 else 1.0
    return 2.0 * halfw / denom


# fix the random seed for reproducibility
np.random.seed(0)


def generate(N: int | None = None):
    m = Quadrotor()

    # Costs
    Q = np.diag([
        10.0, 10.0, 10.0,  # pos
        1.0, 1.0, 1.0,     # vel
        1.0, 1.0, 1.0, 1.0,  # quat (qw,qx,qy,qz)
        2.0, 2.0, 2.0      # omega
    ])
    R = np.diag([1.0, 1.0, 1.0, 1.0])
    Qf = 10 * Q

    N = int(N) if N is not None else 15  # horizon length
    m.dt = 0.05

    # Process noise scaling (conservative)
    sigma_theta = np.deg2rad(2.0)
    q_vec_std = 0.5 * sigma_theta
    q_w_std = 0.1 * q_vec_std
    m.E = m.dt * 5*np.diag([
        0.10, 0.10, 0.10,  # pos [m]
        0.15, 0.15, 0.15,  # vel [m/s]
        q_w_std, q_vec_std, q_vec_std, q_vec_std,  # quat [qw,qx,qy,qz]
        0.2, 0.2, 0.2      # omega [rad/s]
    ])

    solver = SCP_SLS_impl(
        N, Q, R, m, Qf,
        Q_reg=1e4 * np.eye(m.nx),
        R_reg=1e4 * np.eye(m.nu),
        Q_reg_f=1e4 * np.eye(m.nx),
        rti=3,
        fast_sls_rti_steps=2,
    )
    solver.fast_SLS_solver.solver_forward.export_standard_qp = False
    solver.verbose = True
    solver.fast_SLS_solver.verbose = True
    solver.fast_SLS_solver.solver_forward.verbose = False
    solver.fast_SLS_solver.CONV_EPS = 1e-4
    solver.epsilon_convergence = 1e-3

    # derive state bounds from m.g
    ub_x = m.g[:m.nx]
    lb_x = -m.g[m.nx + m.nu: m.nx + m.nu + m.nx]

    # initial condition: random within half-range, with unit quaternion
    rng = np.random.default_rng()
    x0 = rng.uniform(lb_x / 2, ub_x / 2)
    q_rand = rng.normal(size=4)
    nrm = np.linalg.norm(q_rand)
    if nrm < 1e-12:
        q_rand = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        q_rand = q_rand / nrm
    x0[6:10] = q_rand  # [qw,qx,qy,qz]

    simulation_time_steps = 30

    # Allocate logs
    state_trajectory = np.zeros((m.nx, simulation_time_steps))
    input_trajectory = np.zeros((m.nu, simulation_time_steps - 1))
    backoff_trajectory_x = np.zeros((m.nx, N + 1, simulation_time_steps))
    backoff_trajectory_u = np.zeros((m.nu, N, simulation_time_steps))
    nominal_trajectory_x = np.zeros((m.nx, N + 1, simulation_time_steps))
    nominal_trajectory_u = np.zeros((m.nu, N, simulation_time_steps))

    t_jac = np.zeros((simulation_time_steps,1))
    t_qp = np.zeros((simulation_time_steps,1))
    t_riccati = np.zeros((simulation_time_steps,1))


    state_trajectory[:, 0] = np.array(x0).reshape(-1)

    # Closed-loop MPC rollout
    for i in range(simulation_time_steps):
        if i > 0 and hasattr(solver, 'reset_warm_start'):
            solver.reset_warm_start()

        solution = solver.solve(x0)

        tj = float(solution.get('t_jac_ms', np.nan))
        tq = float(solution.get('t_qp_ms', np.nan))
        tb = float(solution.get('t_backward_ms', np.nan))
        t_jac[i] = tj
        t_qp[i] = tq
        t_riccati[i] = tb

        backoff_trajectory_x[:, :, i] = solution['backoff_x'].T
        backoff_trajectory_u[:, :, i] = solution['backoff_u'].T
        nominal_trajectory_x[:, :, i] = solution['primal_x']
        nominal_trajectory_u[:, :, i] = solution['primal_u']

        state_trajectory[:, i] = solution['primal_x'][:, 0]
        if i < simulation_time_steps - 1:
            input_trajectory[:, i] = solution['primal_u'][:, 0]

        # propagate without noise
        x0 = m.ddyn(x0, solution['primal_u'][:, 0])

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(folder_name, f"quadrotor_robust_closed_loop_{timestamp}.npz")

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


def plot():
    files = [f for f in os.listdir(folder_name) if f.endswith('.npz')]
    if not files:
        print("No data files found in the directory.")
        return

    tube_frequency = 5
    latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(folder_name, x)))
    solution = np.load(os.path.join(folder_name, latest_file))

    g = solution['g']
    nx = int(solution['nx'])
    nu = int(solution['nu'])
    dt = float(solution['dt'])
    simulation_time_steps = int(solution['simulation_time_steps'])
    nominal_trajectory_x = solution['nominal_trajectory_x']
    nominal_trajectory_u = solution['nominal_trajectory_u']
    backoff_trajectory_x = solution['backoff_trajectory_x']
    backoff_trajectory_u = solution['backoff_trajectory_u']
    state_trajectory = solution['state_trajectory']
    input_trajectory = solution['input_trajectory']
    N = int(solution['N'])

    ub_x = g[:nx]
    ub_u = g[nx:nx + nu]
    lb_x = -g[nx + nu: nx + nu + nx]
    lb_u = -g[nx + nu + nx: nx + nu + nx + nu]

    plt.rcParams.update({
        "font.size": 18,
        "axes.labelsize": 18,
        "legend.fontsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    })

    # 3x2 grid: 4 state panels + 1 inputs panel, hide the last
    fig, axs = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    axs = axs.flatten()

    labels = [
        [r"$x$", r"$y$", r"$z$"],
        [r"$v_{x}$", r"$v_{y}$", r"$v_{z}$"],
        [r"$q_{w}$", r"$q_{x}$", r"$q_{y}$", r"$q_{z}$"],
        [r"$\omega_{x}$", r"$\omega_{y}$", r"$\omega_{z}$"],
    ]
    input_labels = [r"$f_{1}$", r"$f_{2}$", r"$f_{3}$", r"$f_{4}$"]
    indices = [
        range(0, 3),   # pos
        range(3, 6),   # vel
        range(6, 10),  # quat
        range(10, 13), # omega
    ]

    viridis = plt.cm.viridis
    grid_kw = dict(alpha=0.3, linestyle="--")
    ylabs = [
        "Position [m]",
        "Velocity [m/s]",
        "Quaternion [-]",
        r"Angular rate [rad/s]"
    ]

    alpha_start = 0.35
    alpha_end = 0.05
    denom_time = max(1, int(simulation_time_steps) - 1)

    def _draw_alpha_gradient_tube(ax, x, y1, y2, base_color, a_start, a_end, zorder=1.0):
        """
        Draw a tube between y1 and y2 along x with a gradient alpha from a_start to a_end.
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw on.
        x : array-like, shape (n,)
            The x coordinates.
        y1 : array-like, shape (n,)
            The lower y coordinates.
        y2 : array-like, shape (n,)
            The upper y coordinates.
        base_color : str or tuple
            The base color for the tube.
        a_start : float
            The starting alpha value (at the beginning of x).
        a_end : float
            The ending alpha value (at the end of x).
        """
        x = np.asarray(x)
        y1 = np.asarray(y1)
        y2 = np.asarray(y2)
        if x.ndim != 1 or y1.shape != x.shape or y2.shape != x.shape:
            return
        Ncol = max(200, 4 * len(x))
        alpha = np.linspace(a_start, a_end, Ncol)[None, :]
        rgb = np.array(mcolors.to_rgb(base_color), dtype=float)[:, None]
        rgb_img = np.repeat(rgb, Ncol, axis=1)[None, :, :]
        img = np.concatenate([rgb_img, alpha[None, :, :]], axis=1)
        img = np.moveaxis(img, 1, -1)
        ymin = float(np.minimum(y1, y2).min())
        ymax = float(np.maximum(y1, y2).max())
        if np.isclose(ymax, ymin):
            pad = 1e-9
            ymin -= pad
            ymax += pad
        im = ax.imshow(
            img,
            extent=[float(x.min()), float(x.max()), ymin, ymax],
            origin='lower',
            aspect='auto',
            interpolation='bilinear',
            zorder=zorder,
            clip_on=True,
        )
        poly_verts = np.vstack([
            np.column_stack([x, y1]),
            np.column_stack([x[::-1], y2[::-1]])
        ])
        clip_poly = Polygon(poly_verts, closed=True, facecolor='none', edgecolor='none')
        ax.add_patch(clip_poly)
        im.set_clip_path(clip_poly)

    # main tubes overlay (subset frequency)
    first_pass = True

    for i in range(simulation_time_steps):
        X = nominal_trajectory_x[:, :, i]
        U = nominal_trajectory_u[:, :, i]
        backoff_x_i = backoff_trajectory_x[:, :, i]
        backoff_u_i = backoff_trajectory_u[:, :, i]
        time = np.arange(0, N + 1) * dt + i * dt

        if i % tube_frequency != 0:
            continue

        frac_i = i / denom_time
        scale = 1.0 - 0.4 * float(frac_i)
        a_start_i = alpha_start * scale
        a_end_i = alpha_end * scale
        z_i = 1.0 + (denom_time - i) * 1e-3

        # states panels (raw units)
        for k, (ax, idxs, lbls) in enumerate(zip(axs[:4], indices, labels)):
            colors = viridis(np.linspace(0.3, 0.7, len(idxs)))
            for idx, label, color in zip(idxs, lbls, colors):
                _draw_alpha_gradient_tube(ax, time, X[idx] - backoff_x_i[idx], X[idx] + backoff_x_i[idx],
                                          base_color=color, a_start=a_start_i, a_end=a_end_i, zorder=z_i)
                # draw state constraints if meaningful (skip quaternion panel k==2)
                if k != 2:
                    ax.hlines([lb_x[idx], ub_x[idx]], time[0], time[-1], colors="black", linestyles=['--', '--'])
            ax.set_ylabel(ylabs[k])
            ax.grid(True, **grid_kw)
            if first_pass:
                ax.legend(loc="best")

        # inputs panel (raw units)
        ax_u = axs[4]
        t_u = time[:-1]
        colors_u = viridis(np.linspace(0.3, 0.7, len(input_labels)))
        for j, (label, color) in enumerate(zip(input_labels, colors_u)):
            _draw_alpha_gradient_tube(ax_u, t_u, U[j] - backoff_u_i[j], U[j] + backoff_u_i[j],
                                      base_color=color, a_start=a_start_i, a_end=a_end_i, zorder=z_i)
            ax_u.hlines([lb_u[j], ub_u[j]], t_u[0], t_u[-1], colors='black', linestyles=['--', '--'])
        ax_u.set_ylabel("Rotor thrusts [N]")
        ax_u.grid(True, **grid_kw)
        if first_pass:
            ax_u.legend(loc="best")

        # bottom x labels once
        if first_pass:
            axs[3].set_xlabel("Time [s]")
            axs[4].set_xlabel("Time [s]")

        first_pass = False

    # aggregate trajectories
    X_all = state_trajectory
    U_all = input_trajectory
    time_all = np.arange(X_all.shape[1]) * dt
    time_all_cons = np.arange(X_all.shape[1] + N) * dt

    # states (raw units)
    for k, (ax, idxs, lbls) in enumerate(zip(axs[:4], indices, labels)):
        colors = viridis(np.linspace(0.3, 0.7, len(idxs)))
        ls_cycle = cycle(["-", "--", "-.", ":"])  # reset per panel
        for idx, label, color in zip(idxs, lbls, colors):
            ax.plot(time_all, X_all[idx], label=label, linewidth=2.5, color=color, linestyle=next(ls_cycle))
            if k != 2:
                ax.hlines([lb_x[idx], ub_x[idx]], time_all_cons[0], time_all_cons[-1], colors="black",
                          linestyles=['--', '--'])
        ax.set_ylabel(ylabs[k])
        ax.grid(True, **grid_kw)
        ax.legend(loc="best")

    # inputs (raw units)
    ax_u = axs[4]
    t_u_all = time_all[:-1]
    time_u_all_cons = time_all_cons[:-1]
    colors_u = viridis(np.linspace(0.3, 0.7, len(input_labels)))
    ls_cycle = cycle(["-", "--", "-.", ":"])
    for j, (label, color) in enumerate(zip(input_labels, colors_u)):
        ax_u.plot(t_u_all, U_all[j], label=label, linewidth=2.5, color=color, linestyle=next(ls_cycle))
        ax_u.hlines([lb_u[j], ub_u[j]], time_u_all_cons[0], time_u_all_cons[-1], colors='black', linestyles=['--', '--'])
    ax_u.set_ylabel("Rotor thrusts [N]")
    ax_u.grid(True, **grid_kw)
    ax_u.legend(loc="best")

    # x labels bottom row
    axs[3].set_xlabel("Time [s]")
    axs[4].set_xlabel("Time [s]")

    # hide unused last axis
    axs[5].axis('off')

    plt.tight_layout(pad=1.5)
    out_pdf = os.path.join(folder_name, "trajectory_plot_closed_loop.pdf")
    plt.savefig(out_pdf, format="pdf", dpi=600, bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Quadrotor robust closed-loop experiment")
    parser.add_argument('--run', action='store_true', help='Run the closed-loop simulation (generate data)')
    parser.add_argument('--N', type=int, default=15, help='Horizon length (overrides default if provided)')
    args = parser.parse_args()

    if args.run:
        generate(args.N)
    else:
        plot()