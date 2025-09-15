from dyn.rockETH.rockETH import RockETH

import numpy as np
import matplotlib.pyplot as plt
import os
from util.footnote import add_footnote_time
folder_name = "rockETH_robust_closed_loop"

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

from datetime import datetime
from matplotlib.lines import Line2D
from itertools import cycle
import argparse

# Prefer the JIT solver (with warm-start support); fallback to non-JIT if not available

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
    m = RockETH()
    Q = np.diag([
        10.0,  # x
        10.0,  # y
        10.0,  # z
        1.0,  # vx
        1.0,  # vy
        1.0,  # vz
        1.0,  # qx
        1.0,  # qy
        1.0,  # qz
        1.0,  # qw
        1.0,  # omega_x
        5.0,  # omega_y
        5.0,  # omega_z
        1.0,  # dT
        1.0,  # dtorque
        1.0,  # dservo1
        1.0  # dservo2
    ])

    R = np.diag([
        1.0,
        1.0,
        1.0,
        1.0
    ])

    Qf = 10 * Q

    N = int(N) if N is not None else 15  # horizon length

    m.dt = 0.05
    sigma_theta = np.deg2rad(2.0)  # ~0.035 rad
    q_vec_std = 0.5 * sigma_theta  # ~0.0175
    q_w_std = 0.1 * q_vec_std  # ~0.00175

    m.E = m.dt * np.diag([
        0.20, 0.20, 0.20,  # pos [m]
        0.2, 0.20, 0.20,  # vel [m/s]
        q_vec_std, q_vec_std, q_vec_std, q_w_std,  # quat
        0.2, 0.2, 0.2,  # angular vel [rad/s]
        0.8,  # thrust magnitude
        0.2,  # torque x
        0.04, 0.04  # servo angles [rad]
    ])

    solver = SCP_SLS_impl(N, Q, R, m, Qf,
                                    Q_reg = 1e4 * np.eye(m.nx),
                                    R_reg = 1e4 * np.eye(m.nu),
                                    Q_reg_f = 1e4 * np.eye(m.nx),
                                    rti = 1,
                          fast_sls_rti_steps = 1)
    solver.fast_SLS_solver.solver_forward.export_standard_qp = False

    # quiet down inner solver unless debugging
    solver.verbose = True
    solver.fast_SLS_solver.verbose = True
    solver.fast_SLS_solver.solver_forward.verbose = False
    solver.fast_SLS_solver.CONV_EPS = 1e-4
    solver.epsilon_convergence = 1e-3

    # pick a random initial condition within the state constraints
    # derive bounds from m.g (first nx are state upper bounds; next nu are input ub; then state lb negated)
    ub_x = m.g[:m.nx]
    lb_x = -m.g[m.nx + m.nu: m.nx + m.nu + m.nx]

    # uniform sample per state
    rng = np.random.default_rng()  # for reproducibility
    x0 = rng.uniform(lb_x/2, ub_x/2)

    # overwrite quaternion (slots 6..9) with a random unit quaternion [w,x,y,z]
    q_rand = rng.normal(size=4)
    q_rand /= np.linalg.norm(q_rand) if np.linalg.norm(q_rand) != 0 else 1.0
    x0[6:10] = q_rand

    # final safety clamp
    x0 =np.array([1.75729,
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

    simulation_time_steps = 30  # Define the number of time steps for the simulation
    # Initialize the state trajectory with the initial state
    state_trajectory = np.zeros((m.nx, simulation_time_steps))
    state_trajectory[:, 0] = np.array(x0).reshape(-1)
    # Initialize the input trajectory
    input_trajectory = np.zeros((m.nu, simulation_time_steps - 1))

    # Initialize the backoff trajectory
    backoff_trajectory_x = np.zeros((m.nx, N + 1, simulation_time_steps))
    backoff_trajectory_u = np.zeros((m.nu, N, simulation_time_steps))

    # Initialize the nominal trajectory
    nominal_trajectory_x = np.zeros((m.nx, N + 1, simulation_time_steps))
    nominal_trajectory_u = np.zeros((m.nu, N, simulation_time_steps))

    t_jac = np.zeros((simulation_time_steps,1))
    t_qp = np.zeros((simulation_time_steps,1))
    t_riccati = np.zeros((simulation_time_steps,1))


    # Iterate through the simulation time steps
    for i in range(simulation_time_steps):
        if i > 0 and hasattr(solver, 'reset_warm_start'):
            solver.reset_warm_start()
        print(f"[RockETH] Step {i+1}/{simulation_time_steps}: solving ...")
        # Solve the SCP-SLS problem at each time step
        solution = solver.solve(x0)

        tj = float(solution.get('t_jac_ms', np.nan))
        tq = float(solution.get('t_qp_ms', np.nan))
        tb = float(solution.get('t_backward_ms', np.nan))
        t_jac[i] = tj
        t_qp[i] = tq
        t_riccati[i] = tb
        print(f"[RockETH]   -> done (Δt: jac={solution.get('t_jac_ms', 0.0):.2f} ms, qp={solution.get('t_qp_ms', 0.0):.2f} ms, riccati={solution.get('t_backward_ms', 0.0):.2f} ms)")
        # Store the backoff trajectories
        backoff_trajectory_x[:, :, i] = solution['backoff_x'].T
        backoff_trajectory_u[:, :, i] = solution['backoff_u'].T

        # Store the nominal trajectories
        nominal_trajectory_x[:, :, i] = solution['primal_x']
        nominal_trajectory_u[:, :, i] = solution['primal_u']

        # Store the state and input trajectories
        state_trajectory[:, i] = solution['primal_x'][:, 0]
        if i < simulation_time_steps - 1:
            input_trajectory[:, i] = solution['primal_u'][:, 0]

        # Update the initial state for the next iteration
        # create perturbation for the vertical speed
        w = np.zeros(m.nx)
        #w[5] = 2 * np.random.rand(1) -1  # add noise to vz
        w = 2 * np.random.rand(m.nx) - 1  # add noise to vz

        x0 = m.ddyn(x0, solution['primal_u'][:, 0]) + m.E@ w


    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(folder_name, f"rockETH_robust_closed_loop_{timestamp}.npz")

    # Save the results to a file
    np.savez(filename,
             state_trajectory = state_trajectory,
             input_trajectory = input_trajectory,
             nominal_trajectory_x = nominal_trajectory_x,
             nominal_trajectory_u = nominal_trajectory_u,
             backoff_trajectory_x = backoff_trajectory_x,
             backoff_trajectory_u = backoff_trajectory_u,
             dt=m.dt,
             g = m.g,
             nx = m.nx,
             nu = m.nu,
             simulation_time_steps = simulation_time_steps,
             N = N,
             # timings (ms)
             t_jac=t_jac,
             t_qp=t_qp,
             t_riccati=t_riccati,
    )

    print(f"Results saved to {filename}")


def plot():
    import numpy as np
    import matplotlib.pyplot as plt
    from itertools import cycle
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon
    import matplotlib.colors as mcolors
    files = [f for f in os.listdir(folder_name) if f.endswith('.npz')]

    if not files:
        print("No data files found in the directory.")
        return

    tube_frequency = 5  # how often to plot the tubes (every k time steps)

    latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(folder_name, x)))
    solution = np.load(os.path.join(folder_name, latest_file))

    g = solution['g']
    nx = solution['nx']
    nu = solution['nu']
    dt = solution['dt']
    simulation_time_steps = solution['simulation_time_steps']
    nominal_trajectory_x = solution['nominal_trajectory_x']
    nominal_trajectory_u = solution['nominal_trajectory_u']
    backoff_trajectory_x = solution['backoff_trajectory_x']
    backoff_trajectory_u = solution['backoff_trajectory_u']

    state_trajectory = solution['state_trajectory']
    input_trajectory = solution['input_trajectory']
    N = solution['N']

    ub_x = g[:nx]
    ub_u = g[nx:nx + nu]
    lb_x = -g[nx + nu: nx + nu + nx]
    lb_u = -g[nx + nu + nx: nx + nu + nx + nu]

    plt.rcParams.update({
        "font.size": 18,  # base font size
        "axes.labelsize": 18,  # axis labels
        "legend.fontsize": 16,  # legend
        "xtick.labelsize": 16,  # x ticks
        "ytick.labelsize": 16,  # y ticks
    })

    fig, axs = plt.subplots(2, 3, figsize=(20, 10), sharex=True)
    axs = axs.flatten()  # flatten to list of 6 axes
    labels = [
        [r"$x$", r"$y$", r"$z$"],
        [r"$v_{x}$", r"$v_{y}$", r"$v_{z}$"],
        [r"$q_{x}$", r"$q_{y}$", r"$q_{z}$", r"$q_{w}$"],
        [r"$\omega_{x}$", r"$\omega_{y}$", r"$\omega_{z}$"],
        [r"$T$", r"$\tau$", r"$\theta_{1}$", r"$\theta_{2}$"]
    ]
    input_labels = [r"$T_{in}$", r"$\tau_{in}$", r"$\theta_{1,in}$", r"$\theta_{2,in}$"]
    indices = [
        range(0, 3),
        range(3, 6),
        range(6, 10),
        range(10, 13),
        range(13, 17)
    ]

    viridis = plt.cm.viridis

    grid_kw = dict(alpha=0.3, linestyle="--")
    styles = cycle(["-", "--", "-.", ":"])
    # y-axis labels per panel
    ylabs = ["Position [m]", "Velocity [m/s]", "Quaternion [-]",
             r"Angular vel. [rad/s]", "Thrust/torques (norm.) [-]"]
    import numpy as np
    import matplotlib.pyplot as plt
    from itertools import cycle
    from matplotlib.lines import Line2D

    # alpha schedule for tubes: earlier times within a tube more opaque, later times more transparent
    alpha_start = 0.35  # opacity at tube start time (left)
    alpha_end = 0.05    # opacity at tube end time (right)
    denom_time = max(1, int(simulation_time_steps) - 1)

    # Helper: draw smooth alpha-gradient tube using an RGBA image clipped to the polygon
    def _draw_alpha_gradient_tube(ax, x, y1, y2, base_color, a_start, a_end, zorder=1.0):
        x = np.asarray(x)
        y1 = np.asarray(y1)
        y2 = np.asarray(y2)
        if x.ndim != 1 or y1.shape != x.shape or y2.shape != x.shape:
            return
        # Build a horizontal RGBA strip with alpha varying along x
        N = max(200, 4 * len(x))  # more columns for smoother gradient
        alpha = np.linspace(a_start, a_end, N)[None, :]           # (1, N)
        rgb = np.array(mcolors.to_rgb(base_color), dtype=float)[:, None]  # (3,1)
        rgb_img = np.repeat(rgb, N, axis=1)[None, :, :]           # (1, 3, N)
        img = np.concatenate([rgb_img, alpha[None, :, :]], axis=1)  # (1, 4, N)
        img = np.moveaxis(img, 1, -1)  # -> (1, N, 4)
        ymin = float(np.minimum(y1, y2).min())
        ymax = float(np.maximum(y1, y2).max())
        # Avoid zero-height extent (degenerate) by a tiny padding
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
        # Build clipping polygon between y1 and y2
        poly_verts = np.vstack([
            np.column_stack([x, y1]),
            np.column_stack([x[::-1], y2[::-1]])
        ])
        clip_poly = Polygon(poly_verts, closed=True, facecolor='none', edgecolor='none')
        ax.add_patch(clip_poly)
        im.set_clip_path(clip_poly)

    # gate legends/labels to only add on the first plotted tube
    first_pass = True

    for i in range(simulation_time_steps):
        X = nominal_trajectory_x[:, :, i]
        U = nominal_trajectory_u[:, :, i]
        backoff_x_i = backoff_trajectory_x[:, :, i]
        backoff_u_i = backoff_trajectory_u[:, :, i]
        time = np.arange(0, N + 1) * dt + i * dt

        if i % tube_frequency != 0:
            continue

        # per-tube overall zorder and optional fade across tubes (earlier more opaque)
        frac_i = i / denom_time
        scale = 1.0 - 0.4 * float(frac_i)  # later tubes slightly more transparent overall
        a_start_i = alpha_start * scale
        a_end_i = alpha_end * scale
        z_i = 1.0 + (denom_time - i) * 1e-3

        # --- states (5 subplots) ---
        for k, (ax, idxs, lbls) in enumerate(zip(axs[:5], indices, labels)):
            colors = viridis(np.linspace(0.3, 0.7, len(idxs)))
            ls_cycle = cycle(["-", "--", "-.", ":"])  # reset per panel

            if k == 4:  # normalized to [-1,1]
                for idx, label, color in zip(idxs, lbls, colors):
                    x_norm = _affine_to_unit(X[idx], lb_x[idx], ub_x[idx])
                    b_norm = _tube_halfwidth_to_unit(backoff_x_i[idx], lb_x[idx], ub_x[idx])
                    _draw_alpha_gradient_tube(ax, time, x_norm - b_norm, x_norm + b_norm,
                                              base_color=color, a_start=a_start_i, a_end=a_end_i, zorder=z_i)
                ax.set_ylim(-1.1, 1.1)
                ax.set_ylabel(ylabs[k])
            else:
                for idx, label, color in zip(idxs, lbls, colors):
                    _draw_alpha_gradient_tube(ax, time, X[idx] - backoff_x_i[idx], X[idx] + backoff_x_i[idx],
                                              base_color=color, a_start=a_start_i, a_end=a_end_i, zorder=z_i)
                ax.set_ylabel(ylabs[k])

            ax.grid(True, **grid_kw)
            if first_pass:
                ax.legend(loc="best")

        # --- inputs axis (normalized) ---
        ax_u = axs[-1]
        t_u = time[:-1]
        colors_u = viridis(np.linspace(0.3, 0.7, len(input_labels)))
        ls_cycle = cycle(["-", "--", "-.", ":"])
        for j, (label, color) in enumerate(zip(input_labels, colors_u)):
            u_norm = _affine_to_unit(U[j], lb_u[j], ub_u[j])
            b_norm = _tube_halfwidth_to_unit(backoff_u_i[j], lb_u[j], ub_u[j])
            _draw_alpha_gradient_tube(ax_u, t_u, u_norm - b_norm, u_norm + b_norm,
                                      base_color=color, a_start=a_start_i, a_end=a_end_i, zorder=z_i)
        if first_pass:
            ax_u.hlines([-1, 1], t_u[0], t_u[-1], colors='red', linestyles=[ ':'], label='_nolegend_',linewidth=2.5)
        ax_u.set_ylim(-1.1, 1.1)
        ax_u.set_ylabel("Inputs (norm.) [-]")
        ax_u.grid(True, **grid_kw)
        if first_pass:
            ax_u.legend(loc="best")

        # x labels on bottom row (once)
        if first_pass:
            axs[4].set_xlabel("Time [s]")
            axs[5].set_xlabel("Time [s]")

        # one legend entry for constraints (proxy) added once on panel 2
        if first_pass:
            constraint_proxy = [Line2D([0], [0], color='red', linestyle=':', lw=2.5, label='Constraint')]
            axs[1].legend(handles=axs[1].get_legend_handles_labels()[0] + constraint_proxy, loc="best")

        first_pass = False

    X_all = state_trajectory
    U_all = input_trajectory
    time_all = np.arange(X_all.shape[1]) * dt
    time_all_cons = np.arange(X_all.shape[1] + N) * dt


    # --- states (same 5 panels) ---
    for k, (ax, idxs, lbls) in enumerate(zip(axs[:5], indices, labels)):
        colors = viridis(np.linspace(0.3, 0.7, len(idxs)))
        ls_cycle = cycle(["-", "--", "-.", ":"])  # reset per panel

        if k == 4:  # normalized
            for idx, label, color in zip(idxs, lbls, colors):
                x_norm = _affine_to_unit(X_all[idx], lb_x[idx], ub_x[idx])
                ax.plot(time_all, x_norm, label=label, linewidth=2.5, color=color, linestyle=next(ls_cycle))
            ax.hlines([-1, 1], time_all_cons[0], time_all_cons[-1], colors='red', linestyles=[ ':'],linewidth=2.5)
            ax.set_ylim(-1.1, 1.1)
        else:
            for idx, label, color in zip(idxs, lbls, colors):
                ax.plot(time_all, X_all[idx], label=label, linewidth=2.5, color=color, linestyle=next(ls_cycle))
                if k not in (0, 2):
                    ax.hlines([lb_x[idx], ub_x[idx]], time_all_cons[0], time_all_cons[-1], colors="red",
                              linestyles=[':'],linewidth=2.5)
                    ax.set_ylim(1.1 *lb_x[idx], 1.1* ub_x[idx])

        ax.set_ylabel(ylabs[k])
        ax.grid(True, **grid_kw)
        ax.legend(loc="best")

    # --- inputs (normalized) ---
    ax_u = axs[-1]
    t_u_all = time_all[:-1]
    time_u_all_cons = time_all_cons[:-1]
    colors_u = viridis(np.linspace(0.3, 0.7, len(input_labels)))
    ls_cycle = cycle(["-", "--", "-.", ":"])
    for j, (label, color) in enumerate(zip(input_labels, colors_u)):
        u_norm = _affine_to_unit(U_all[j], lb_u[j], ub_u[j])
        ax_u.plot(t_u_all, u_norm, label=label, linewidth=2.5, color=color, linestyle=next(ls_cycle))
    ax_u.hlines([-1, 1], time_u_all_cons[0], time_u_all_cons[-1], colors='red', linestyles=[':'], linewidth=2.5)
    ax_u.set_ylim(-1.1, 1.1)
    ax_u.set_ylabel("Inputs (norm.) [-]")
    ax_u.grid(True, **grid_kw)
    ax_u.legend(loc="best")

    # x labels bottom row
    axs[3].set_xlabel("Time [s]")
    axs[4].set_xlabel("Time [s]")
    axs[5].set_xlabel("Time [s]")

    plt.tight_layout(pad=1.5)
    plt.savefig("rockETH_robust_closed_loop/trajectory_plot_closed_loop.pdf", format="pdf", dpi=600, bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RockETH robust closed-loop experiment")
    parser.add_argument('--run', action='store_true', help='Run the closed-loop simulation (generate data)')
    parser.add_argument('--N', type=int, default=None, help='Horizon length (overrides default if provided)')
    args = parser.parse_args()

    if args.run:
        generate(args.N)
    else:
        plot()