import casadi as ca
import numpy as np
from dyn.model import Model


class Quadrotor(Model):
    """
    6-DOF quadrotor (drone) rigid-body model with quaternion attitude and 4 actuators (rotors).

    State x (nx=13):
      [0:3]   position (x, y, z) in world [m]
      [3:6]   linear velocity (vx, vy, vz) in world [m/s]
      [6:10]  unit quaternion (qw, qx, qy, qz) body->world
      [10:13] body angular velocity (wx, wy, wz) [rad/s]

    Input u (nu=4):
      rotor thrusts [f1, f2, f3, f4] in Newtons (non-negative). By default X configuration:
        - Arms of length l along +/-x and +/-y axes.
        - f1: +x arm (front)
        - f2: +y arm (right)
        - f3: -x arm (rear)
        - f4: -y arm (left)
      Roll torque ~ l*(f2 - f4), pitch torque ~ l*(f3 - f1), yaw torque ~ kM*(f1 - f2 + f3 - f4)

    Parameters (defaults tuned for a small quad):
      mass m [kg], gravity g [m/s^2], arm length l [m],
      inertia diag Jx,Jy,Jz [kg m^2], yaw moment coeff kM [m].
    """

    def __init__(self):
        # Physical parameters
        self.params = {
            'm': 1.0,              # mass [kg]
            'g': 9.81,             # gravity [m/s^2]
            'l': 0.15,             # arm length [m]
            'Jx': 0.02,            # inertia [kg m^2]
            'Jy': 0.02,
            'Jz': 0.04,
            'kM': 0.01,            # yaw moment coefficient [m] (N->Nm)
        }

        # State and input symbols
        self.state_names = [
            'x', 'y', 'z',
            'vx', 'vy', 'vz',
            'qw', 'qx', 'qy', 'qz',
            'wx', 'wy', 'wz',
        ]
        self.control_names = ['f1', 'f2', 'f3', 'f4']

        self.state = [ca.SX.sym(n) for n in self.state_names]
        self.control = [ca.SX.sym(n) for n in self.control_names]

        # Dimensions and step
        self.nx = len(self.state)
        self.nu = len(self.control)
        self.dt = 0.05

        # Neutral hover (z-up world): at origin, level attitude, zero rates, hover thrust per rotor
        m, g = self.params['m'], self.params['g']
        f_hover = m * g / 4.0
        self.neutral_state = ca.vertcat(
            0, 0, 0,     # pos
            0, 0, 0,     # vel
            1, 0, 0, 0,  # quaternion [qw,qx,qy,qz]
            0, 0, 0      # omega
        )
        self.neutral_input = ca.vertcat(f_hover, f_hover, f_hover, f_hover)

        # Bounds
        state_bounds = {
            'x': (-20.0, 20.0), 'y': (-20.0, 20.0), 'z': (-20.0, 20.0),
            'vx': (-10.0, 10.0), 'vy': (-10.0, 10.0), 'vz': (-10.0, 10.0),
            'qw': (-1.5, 1.5), 'qx': (-1.5, 1.5), 'qy': (-1.5, 1.5), 'qz': (-1.5, 1.5),
            'wx': (-20.0, 20.0), 'wy': (-20.0, 20.0), 'wz': (-20.0, 20.0),
        }
        control_bounds = {n: (0.0, 20.0) for n in self.control_names}

        x_lb, x_ub = [], []
        for s in self.state:
            lb, ub = state_bounds[s.name()]
            x_lb.append(lb)
            x_ub.append(ub)
        u_lb, u_ub = [], []
        for ui in self.control:
            lb, ub = control_bounds[ui.name()]
            u_lb.append(lb)
            u_ub.append(ub)
        x_lb = np.array(x_lb); x_ub = np.array(x_ub)
        u_lb = np.array(u_lb); u_ub = np.array(u_ub)

        self.G = ca.vertcat(np.eye(self.nx + self.nu), -np.eye(self.nx + self.nu))
        self.g = np.concatenate((np.concatenate((x_ub, u_ub)), -np.concatenate((x_lb, u_lb))))
        self.ni = 2 * (self.nx + self.nu)
        self.Gf = ca.vertcat(np.eye(self.nx), -np.eye(self.nx))
        self.gf = np.concatenate((x_ub, -x_lb))
        self.ni_f = 2 * self.nx

        # Disturbance scaling (diagonal)
        self.E = np.diag([
            0.05, 0.05, 0.05,    # pos [m]
            0.1, 0.1, 0.1,       # vel [m/s]
            0.02, 0.02, 0.02, 0.01,  # quat comps
            0.2, 0.2, 0.2        # omega [rad/s]
        ])
        self.nw = self.nx

    def ode(self, X, u):
        p = self.params
        m, g = p['m'], p['g']
        l, kM = p['l'], p['kM']
        Jx, Jy, Jz = p['Jx'], p['Jy'], p['Jz']

        # Unpack state
        pos = X[0:3]  # unused in dynamics directly
        v = X[3:6]
        qw, qx, qy, qz = X[6], X[7], X[8], X[9]  # [qw,qx,qy,qz]
        wx, wy, wz = X[10], X[11], X[12]

        # Unpack inputs (thrusts >= 0)
        f1, f2, f3, f4 = u[0], u[1], u[2], u[3]
        Fz = f1 + f2 + f3 + f4  # total thrust along body +Z

        # Rotation matrix R(q): body->world (same formulation used in rockETH)
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

        # Body-frame thrust vector (z-up in body)
        Fb = ca.vertcat(0, 0, Fz)
        a = (1.0 / m) * ca.mtimes(R, Fb)
        a = ca.vertcat(a[0], a[1], a[2] - g)  # subtract gravity in world z

        # Quaternion kinematics: qdot = 0.5 * Omega(omega) * q, with q=[qw,qx,qy,qz]
        omega_mat = ca.vertcat(
            ca.hcat([0,   -wx,  -wy,  -wz]),
            ca.hcat([wx,    0,   wz,  -wy]),
            ca.hcat([wy,  -wz,    0,   wx]),
            ca.hcat([wz,   wy,  -wx,    0])
        )
        q = ca.vertcat(qw, qx, qy, qz)
        q_dot = 0.5 * ca.mtimes(omega_mat, q)

        # Moments from rotor thrusts (X config mapping)
        # tau_x: roll, tau_y: pitch, tau_z: yaw
        tau_x = l * (f2 - f4)
        tau_y = l * (f3 - f1)
        tau_z = kM * (f1 - f2 + f3 - f4)
        tau = ca.vertcat(tau_x, tau_y, tau_z)

        # Rigid-body rotational dynamics: J*wdot = tau - omega x (J*omega)
        J = ca.diag(ca.vertcat(Jx, Jy, Jz))
        omega = ca.vertcat(wx, wy, wz)
        omega_dot = ca.mtimes(ca.inv(J), (tau - ca.cross(omega, ca.mtimes(J, omega))))

        # State derivative
        xdot = ca.vertcat(
            v,
            a,
            q_dot,
            omega_dot
        )
        return xdot

