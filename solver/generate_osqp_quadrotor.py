# generate_osqp_integrator.py
from dyn.quadrotor import Quadrotor
from dyn.LTV import LTV
from dyn.quadrotor import Quadrotor
from solver.qp_jit import QP
from dyn.integrator import Integrator
import numpy as np
# delete the build directory if it exists
import shutil
import os
import argparse
build_dir = "build/osqp_fast"
if os.path.exists(build_dir):
    shutil.rmtree(build_dir)


m = Quadrotor()
Q = np.eye(m.nx)
R = np.eye(m.nu)

Qf = 10 * Q
parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int, default=15, help='Horizon length')
args = parser.parse_args()
N = int(args.N)

m_LTV = LTV(m, N)
qp = QP(N, Q, R, m_LTV, Qf, backend="osqp")  # uses Python OSQP just to export
qp.export_osqp_solver("build/osqp_fast", extension_name="osqp_generated")
print("Wrote code to build/osqp_fast")
