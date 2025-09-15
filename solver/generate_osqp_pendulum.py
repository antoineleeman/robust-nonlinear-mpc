# generate_osqp_pendulum.py
from dyn.pendulum import Pendulum
from dyn.LTV import LTV
from solver.qp_jit import QP
import numpy as np

import shutil
import os
import argparse
build_dir = "build/osqp_fast"
if os.path.exists(build_dir):
    shutil.rmtree(build_dir)


m = Pendulum()
Q = np.eye(m.nx)
R = np.eye(m.nu)
Qf = np.eye(m.nx)
parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int, default=15, help='Horizon length')
args = parser.parse_args()
N = int(args.N)
m.E = 0.003 * np.eye(m.nx)

m.dt = 0.05

x_max = 10 * np.ones(m.nx)
x_min = -10 * np.ones(m.nx)
u_max = 5 * np.ones(m.nu)
u_min = -5 * np.ones(m.nu)
x_max_f = 10 * np.ones(m.nx)
x_min_f = -10 * np.ones(m.nx)

m.replace_constraints(x_max, x_min, u_max, u_min, x_max_f, x_min_f)

m_LTV = LTV(m, N)
qp = QP(N, Q, R, m_LTV, Qf, backend="osqp")  # uses Python OSQP just to export
qp.export_osqp_solver("build/osqp_fast", extension_name="osqp_generated")
print("Wrote code to build/osqp_fast")
