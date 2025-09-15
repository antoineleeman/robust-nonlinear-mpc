
import casadi as ca
import numpy as np
from dyn.model import Model
import matplotlib.pyplot as plt

import os
from datetime import datetime
import time
from util.quaternion_to_euler import quaternion_to_euler


class Rocket(Model):
    """
    A continuous-time model for a 6-DOF rocket dynamics, including thrust and torque control.
    This model uses quaternion representation for orientation and includes gimbal angles for thrust vectoring.
    The model is based on the equations of motion for a rigid body in space, with thrust and torque inputs.
    The state vector includes position, velocity, orientation (quaternion), angular velocity, thrust magnitude,
    torque, and servo angles for gimbal control.
    The model is based on this paper: https://arxiv.org/abs/1802.03827
    The parameters are based on this paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9636430
    """
    def __init__(self):
        # --- Parameters ---
        self.params = {
            'mass': 1.16,
            'gravity_constant': 9.81,
            'inertia_xx': 0.00210,
            'inertia_yy': 0.10000,
            'inertia_zz': 0.10000,
            'thrust_cog_offset': 0.42000,
            'thrust_magnitude_time_constant': 0.06000,
            'servo_angle_time_constant': 0.10,
            'gimbal_a': 5.0,
            'gimbal_b': 35.2,
            'gimbal_c': 33.0,
            'gimbal_d': 28.0,
            'gimbal_e': 35.2,
        }

        # --- State and Control Definitions ---
        self.state_names = [
            'x', 'y', 'z',
            'v_x', 'v_y', 'v_z',
            'quat_x', 'quat_y', 'quat_z', 'quat_w',
            'angular_vx', 'angular_vy', 'angular_vz',
            'thrust_magnitude', 'torque_x', 'servo_angle_1', 'servo_angle_2'
        ]
        self.control_names = ['thrust_magnitude_u', 'torque_u', 'servo_angle_1_u', 'servo_angle_2_u']

        self.state = [ca.SX.sym(name) for name in self.state_names]
        self.control = [ca.SX.sym(name) for name in self.control_names]

        self.state_index = {name: i for i, name in enumerate(self.state_names)}
        self.control_index = {name: i for i, name in enumerate(self.control_names)}
        self.get_state_index = lambda n: self.state_index[n] if isinstance(n, str) else [self.state_index[k] for k
                                                                                         in n]
        self.get_control_index = lambda n: self.control_index[n] if isinstance(n, str) else [self.control_index[k]
                                                                                             for k in n]
        self.state_groups = {
            'pos': slice(0, 3),
            'vel': slice(3, 6),
            'quat': slice(6, 10),
            'omega': slice(10, 13),
            'act': slice(13, 17),
        }

        # --- Neutral State ---
        self.neutral_state = ca.vertcat(
            0, 0, 0,  # position
            0, 0, 0,  # velocity
            1, 0, 0, 0,  # quaternion (w last)
            0, 0, 0,  # angular velocity
            0,  # thrust magnitude
            0,  # torque
            0, 0  # servo angles
        )

        self.nx = len(self.state)
        self.nu = len(self.control)
        self.dt = 0.05

        # --- Bounds (modular) ---
        MAX_QUAT_BOUND = 1.5
        state_bounds = {
            'x': (-10.0, 10.0),
            'y': (-10.0, 10.0),
            'z': (-10.0, 10.0),
            'v_x': (-1.0, 1.0),
            'v_y': (-1.0, 1.0),
            'v_z': (-1.0, 1.0),
            'quat_x': (-MAX_QUAT_BOUND, MAX_QUAT_BOUND),
            'quat_y': (-MAX_QUAT_BOUND, MAX_QUAT_BOUND),
            'quat_z': (-MAX_QUAT_BOUND, MAX_QUAT_BOUND),
            'quat_w': (-MAX_QUAT_BOUND, MAX_QUAT_BOUND),
            'angular_vx': (-2.0, 2.0),
            'angular_vy': (-2.0, 2.0),
            'angular_vz': (-2.0, 2.0),
            'thrust_magnitude': (-50.0, 50.0),
            'torque_x': (-2.0, 2.0),
            'servo_angle_1': (-1.0, 1.0),
            'servo_angle_2': (-1.0, 1.0),
        }

        control_bounds = {
            'thrust_magnitude_u': (-50.0, 50.0),
            'torque_u': (-2.0, 2.0),
            'servo_angle_1_u': (-1.0, 1.0),
            'servo_angle_2_u': (-1.0, 1.0),
        }

        x_lb, x_ub = [], []
        for s in self.state:
            lb, ub = state_bounds[s.name()]
            x_lb.append(lb)
            x_ub.append(ub)

        u_lb, u_ub = [], []
        for u in self.control:
            lb, ub = control_bounds[u.name()]
            u_lb.append(lb)
            u_ub.append(ub)

        x_lb = np.array(x_lb)
        x_ub = np.array(x_ub)
        u_lb = np.array(u_lb)
        u_ub = np.array(u_ub)

        self.G = ca.vertcat(np.eye(self.nx + self.nu), -np.eye(self.nx + self.nu))
        self.g = np.concatenate((np.concatenate((x_ub, u_ub)), -np.concatenate((x_lb, u_lb))))
        self.ni = 2 * (self.nx + self.nu)
        state_names = [s.name() for s in self.state]
        control_names = [u.name() for u in self.control]
        constraint_vars = state_names + control_names  # Order matches x_ub + u_ub
        self.constraint_names = [f"{name}_ub" for name in constraint_vars] + [f"{name}_lb" for name in constraint_vars]

        self.Gf = ca.vertcat(np.eye(self.nx), -np.eye(self.nx))
        self.gf = np.concatenate((x_ub, -x_lb))
        self.ni_f = 2 * self.nx
        self.constraint_names_f = [f"{name}_ub" for name in state_names] + [f"{name}_lb" for name in state_names]

        # --- E matrix (uncertainty scaling) ---
        self.E_crs = np.diag([
            4.2, 3.5, 3.5,  # pos
            1.8, 1.6, 1.6,  # vel
            20.0, 20.0, 20.0, 20.0,  # quat
            0.01, 2.7, 2.7,  # angular vel
            0.1, 0.1, 0.1, 0.1  # thrust, torque, servo1, servo2
        ])
        sigma_theta = np.deg2rad(2.0)  # ~0.035 rad
        q_vec_std = 0.5 * sigma_theta  # ~0.0175
        q_w_std = 0.1 * q_vec_std  # ~0.00175

        self.E = np.diag([
            0.03, 0.03, 0.03,  # pos [m]
            0.08, 0.08, 0.08,  # vel [m/s]
            q_vec_std, q_vec_std, q_vec_std, q_w_std,  # quat
            0.10, 0.10, 0.10,  # angular vel [rad/s]
            0.8,  # thrust magnitude
            0.2,  # torque x
            0.04, 0.04  # servo angles [rad]
        ])
        self.nw = self.nx

    def ode(self, X, u):
        p = self.params

        # state unpack
        v = ca.vertcat(X[3], X[4], X[5])
        qw, qx, qy, qz = X[6], X[7], X[8], X[9]  # MuJoCo: [w,x,y,z]
        omega = ca.vertcat(X[10], X[11], X[12])

        thrust_mag = X[13] + 11.3796  # gravity comp
        torque_x = X[14]
        sa1 = X[15]
        sa2 = X[16]

        # inputs
        thrust_input = u[0] + 11.3796
        torque_input = u[1]
        sa1_input = u[2]
        sa2_input = u[3]

        # gimbals
        gimbal1 = self.compute_gimbal_angle(sa1, 0.0)
        gimbal2 = self.compute_gimbal_angle(sa2, gimbal1)

        # body-frame thrust (z-up; zero gimbal => +Z thrust)
        B_thrust = ca.vertcat(
            -thrust_mag * ca.sin(gimbal1) * ca.cos(gimbal2),  # Fx
            thrust_mag * ca.sin(gimbal2),  # Fy
            thrust_mag * ca.cos(gimbal1) * ca.cos(gimbal2)  # Fz
        )

        r00 = 1 - 2*qy**2 - 2*qz**2
        r01 = 2*qx*qy - 2*qz*qw
        r02 = 2*qx*qz + 2*qy*qw
        r10 = 2*qx*qy + 2*qz*qw
        r11 = 1 - 2*qx**2 - 2*qz**2
        r12 = 2*qy*qz - 2*qx*qw
        r20 = 2*qx*qz - 2*qy*qw
        r21 = 2*qy*qz + 2*qx*qw
        r22 = 1 - 2*qx**2 - 2*qy**2
        R = ca.vertcat(
            ca.hcat([r00, r01, r02]),
            ca.hcat([r10, r11, r12]),
            ca.hcat([r20, r21, r22])
        )

        acc = (1 / p['mass']) * ca.mtimes(R, B_thrust)
        acc[2] -= p['gravity_constant']

        # Build omega_mat using SX-friendly concatenation
        wx, wy, wz = omega[0], omega[1], omega[2]
        omega_mat = ca.vertcat(
            ca.hcat([0,   -wx,  -wy,  -wz]),
            ca.hcat([wx,    0,   wz,  -wy]),
            ca.hcat([wy,  -wz,    0,   wx]),
            ca.hcat([wz,   wy,  -wx,    0])
        )

        quat = ca.vertcat(qw, qx, qy, qz)  # [w,x,y,z]
        q_dot = 0.5 * ca.mtimes(omega_mat, quat)  # returns [ẇ,ẋ,ẏ,ż]

        # rotational dynamics
        cog_offset = ca.vertcat(0, 0, -p['thrust_cog_offset'])
        torque_vec = ca.cross(cog_offset, B_thrust)
        J = ca.diag(ca.vertcat(p['inertia_xx'], p['inertia_yy'], p['inertia_zz']))
        omega_dot = ca.mtimes(ca.inv(J), (torque_vec - ca.cross(omega, ca.mtimes(J, omega))))

        # actuator first-order lags
        thrust_dot = (thrust_input - thrust_mag) / p['thrust_magnitude_time_constant']
        torque_dot = (torque_input - torque_x) / p['thrust_magnitude_time_constant']
        sa1_dot = (sa1_input - sa1) / p['servo_angle_time_constant']
        sa2_dot = (sa2_input - sa2) / p['servo_angle_time_constant']

        # pack state derivative (quaternion back as [w,x,y,z] in slots 6..9)
        return ca.vertcat(
            v,
            acc,
            q_dot[0], q_dot[1], q_dot[2], q_dot[3],
            omega_dot,
            thrust_dot, torque_dot, sa1_dot, sa2_dot
        )

    def compute_gimbal_angle(self, servo_angle, tilt_axis_angle):
        p = self.params
        iv1 = p['gimbal_d'] + p['gimbal_a'] * ca.cos(servo_angle)
        iv2 = p['gimbal_e'] - p['gimbal_a'] * ca.sin(servo_angle)
        u = p['gimbal_b']**2 - p['gimbal_c']**2 - iv1**2 - iv2**2
        v = 2 * p['gimbal_c'] * ca.cos(tilt_axis_angle) * iv2
        w = -2 * p['gimbal_c'] * iv1
        iv3 = w**2 + v**2 - u**2
        return 2 * ca.atan((v - ca.sqrt(iv3)) / (u + w))

    def plot_state_trajectory(self, X, U, time=None, ax=None):
        """
        Plot grouped state and input trajectories.
        X: state matrix (shape: nx x N+1)
        U: input matrix (shape: nu x N)
        """
        if time is None:
            time = np.arange(X.shape[1]) * self.dt

        if ax is None:
            fig, axs = plt.subplots(6, 1, figsize=(12, 18), sharex=True)
        else:
            axs = ax

        labels = [
            ['x', 'y', 'z'],
            ['v_x', 'v_y', 'v_z'],
            ['q_x', 'q_y', 'q_z', 'q_w'],
            ['w_x', 'w_y', 'w_z'],
            ['T', 'tau', 'theta_1', 'theta_2']
        ]
        indices = [
            range(0, 3),
            range(3, 6),
            range(6, 10),
            range(10, 13),
            range(13, 17)
        ]
        viridis = plt.cm.viridis

        for ax, idxs, lbls in zip(axs[:5], indices, labels):
            colors = viridis(np.linspace(0.3, 0.7, len(idxs)))
            for i, (label, color) in enumerate(zip(lbls, colors)):
                ax.plot(time, X[idxs[i]], label=label, linewidth=3.0, color=color)
            ax.set_ylabel('State', fontsize=14)
            ax.tick_params(axis='both', labelsize=12)
            ax.legend(fontsize=12)
            ax.grid(True)

        input_labels = ['T_in', 'τ_in', 'θ₁_in', 'θ₂_in']
        colors = viridis(np.linspace(0.3, 0.7, len(input_labels)))
        ax = axs[-1]
        for i, (label, color) in enumerate(zip(input_labels, colors)):
            ax.plot(time[:-1], U[i], label=label, linewidth=3.0, color=color)
        ax.set_ylabel('Input', fontsize=14)
        ax.set_xlabel('Time [s]', fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        ax.legend(fontsize=12)
        ax.grid(True)

    def plot_normalized_state_tube_with_constraints(self, center, backoff, labels=None):
        """
        Plot normalized state tubes between constraint bounds (0 = lower_bound, 1 = upper_bound).

        :param center: (nx, N) array of state trajectories
        :param backoff: (nx, N) array of backoff widths
        :param g: constraint vector (length: 2*(nx + nu))
        :param nx: number of states
        :param nu: number of inputs
        :param dt: timestep
        :param labels: list of label groups (optional)
        """
        nx, N = center.shape
        time = np.arange(N) * self.dt
        g = self.g
        if labels is None:
            labels = [
                ['x', 'y', 'z'],
                ['v_x', 'v_y', 'v_z'],
                ['q_x', 'q_y', 'q_z', 'q_w'],
                ['w_x', 'w_y', 'w_z'],
                ['T', 'τ', 'θ₁', 'θ₂'],
            ]

        indices = [
            range(0, 3),
            range(3, 6),
            range(6, 10),
            range(10, 13),
            range(13, 17),
        ]

        fig, axs = plt.subplots(5, 1, figsize=(12, 18), sharex=True)
        viridis = plt.cm.viridis

        for ax, idxs, lbls in zip(axs, indices, labels):
            colors = viridis(np.linspace(0.3, 0.7, len(idxs)))
            for idx, label, color in zip(idxs, lbls, colors):
                # lb = g[idx]
                # ub = -g[idx + self.nx + self.nu]
                ub = g[idx]
                lb = -g[idx + self.nx + self.nu]
                denom = ub - lb if ub - lb != 0 else 1.0

                lower = (center[idx] - backoff[idx] - lb) / denom
                upper = (center[idx] + backoff[idx] - lb) / denom

                ax.fill_between(time, lower, upper, alpha=0.4, color=color, label=label)
                ax.hlines([0, 1], time[0], time[-1], colors=color, linestyles=['--', ':'])

            ax.set_ylabel('Normalized State', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True)

        axs[-1].set_xlabel('Time [s]', fontsize=12)
        plt.tight_layout()
        plt.show()

    def plot_states_constraints(self, N, ax = None):
        """
        Similar as plot_state_tube, but for plotting state constraints stored in m.g
        :param X:
        :param time:
        :param ax:
        :return:
        """
        if ax is None:
            fig, axs = plt.subplots(5, 1, figsize=(12, 18), sharex=True)
        else:
            axs = ax

        time = np.arange(N) * self.dt

        labels = [
            ['x', 'y', 'z'],
            ['v_x', 'v_y', 'v_z'],
            ['q_x', 'q_y', 'q_z', 'q_w'],
            ['w_x', 'w_y', 'w_z'],
            ['T', 'τ', 'θ₁', 'θ₂'],
        ]
        indices = [
            range(0, 3),
            range(3, 6),
            range(6, 10),
            range(10, 13),
            range(13, 17),
        ]

        viridis = plt.cm.viridis

        for ax, idxs, lbls in zip(axs, indices, labels):
            colors = viridis(np.linspace(0.3, 0.7, len(idxs)))
            for i, (idx, label, color) in enumerate(zip(idxs, lbls, colors)):
                lower_bound = self.g[idx]
                upper_bound = -self.g[idx + self.nx + self.nu]
                # plot the constraints as horizontal lines
                ax.hlines(lower_bound, time[0], time[-1],  color=color, linestyle='--', label=f'{label} lower')
                ax.hlines(upper_bound, time[0], time[-1], color=color, linestyle=':', label=f'{label} upper')
            ax.tick_params(axis='both', labelsize=12)
            ax.legend(fontsize=12)

        plt.show()

    def plot_state_tube(self, backoff, center, time=None, ax = None):
        """
        Plot grouped state tube (center ± backoff) over time.
        """
        if backoff.shape[0] != self.nx:
            backoff = backoff.T
        if center.shape[0] != self.nx:
            center = center.T
        if time is None:
            time = np.arange(center.shape[1]) * self.dt

        if ax is None:
            fig, axs = plt.subplots(5, 1, figsize=(12, 18), sharex=True)
        else:
            axs = ax

        labels = [
            ['x', 'y', 'z'],
            ['v_x', 'v_y', 'v_z'],
            ['q_x', 'q_y', 'q_z', 'q_w'],
            ['w_x', 'w_y', 'w_z'],
            ['T', 'τ', 'θ₁', 'θ₂'],
        ]
        indices = [
            range(0, 3),
            range(3, 6),
            range(6, 10),
            range(10, 13),
            range(13, 17),
        ]

        viridis = plt.cm.viridis

        for ax, idxs, lbls in zip(axs, indices, labels):
            colors = viridis(np.linspace(0.3, 0.7, len(idxs)))
            for i, (idx, label, color) in enumerate(zip(idxs, lbls, colors)):
                lower = center[idx] - backoff[idx] + 1e-6
                upper = center[idx] + backoff[idx] - 1e-6
                ax.fill_between(time, lower, upper, alpha=0.5, color=color, label=label)
            ax.set_ylabel('State', fontsize=14)
            ax.tick_params(axis='both', labelsize=12)
            ax.legend(fontsize=12)
            ax.grid(True)

        axs[-1].set_xlabel('Time [s]', fontsize=14)
        axs[-1].tick_params(axis='both', labelsize=12)

    @staticmethod
    def save_trajectory(solution_to_save, directory="trajectories"):
        """
        Save the state and control trajectories to a file.
        :param directory: Folder to save the trajectory file.
        :param solution_to_save: Dictionary containing 'primal_x' and 'primal_u'.
        """
        if 'primal_x' not in solution_to_save or 'primal_u' not in solution_to_save:
            raise RuntimeError("No solution found to save. Run `.solve()` first.")

        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(directory, f"trajectory_{timestamp}.npz")

        x = solution_to_save['primal_x'].copy()

        # Save the trajectory
        np.savez(
            filename,
            x=x,
            u=solution_to_save['primal_u']
        )

    @staticmethod
    def load_trajectory(directory="trajectories"):
        """
        Load the most recent trajectory from a folder.
        :param directory: Folder containing .npz trajectory files.
        :return: Dictionary with keys 'primal_x' and 'primal_u'
        """
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory '{directory}' does not exist.")

        files = [f for f in os.listdir(directory) if f.endswith(".npz")]
        if not files:
            raise FileNotFoundError(f"No .npz files found in directory '{directory}'.")

        latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
        filepath = os.path.join(directory, latest_file)

        data = np.load(filepath)
        print(f"Loaded trajectory from {filepath}")
        print(data['x'])
        return {
            'primal_x': data['x'],
            'primal_u': data['u']
        }