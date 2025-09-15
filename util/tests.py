import matplotlib.pyplot as plt
import numpy as np
from util.SLS import SLS
from scipy.linalg import block_diag

def check_output_feedback(solution_sls, x0, m, N, delay=False, axes=None):
    if not np.all(axes):
        raise NotImplementedError
    # np.random.seed(0)
    primal_x = solution_sls["primal_x"]
    primal_u = solution_sls["primal_u"]
    tube = solution_sls["backoff"]
    tube_f = solution_sls["backoff_f"]
    
    # tube and nominal trajectory
    for idx in range(m.nx):
        for t in range(N):
            axes[idx].plot([t-0.1, t+0.1], [primal_x[idx,t]+tube[t,idx], primal_x[idx,t]+tube[t,idx]], 'k-', linewidth=1)
            axes[idx].plot([t-0.1, t+0.1], [primal_x[idx,t]-tube[t,idx], primal_x[idx,t]-tube[t,idx]], 'k-', linewidth=1)
        axes[idx].plot([N-0.1, N+0.1], [primal_x[idx,N]+tube_f[idx], primal_x[idx,N]+tube_f[idx]], 'k-', linewidth=1)
        axes[idx].plot([N-0.1, N+0.1], [primal_x[idx,N]-tube_f[idx], primal_x[idx,N]-tube_f[idx]], 'k-', linewidth=1)

        axes[idx].plot(np.arange(N+1), primal_x[idx,:], label="nominal")
        axes[idx].legend()
    
    # rollouts
    K_mat = solution_sls["Phi_uy_mat"] - solution_sls["Phi_ux_mat"] @ np.linalg.inv(solution_sls["Phi_xx_mat"]) @ solution_sls["Phi_xy_mat"]
    for rollout in range(10):
        states = np.zeros((m.nx, N+1))
        states[:,0] = x0
        observations = np.zeros((m.ny, N+1))
        disturbance_state = np.zeros((m.nx, N))
        disturbance_obs = np.zeros((m.nx, N))
        
        if rollout < 5:
            for i in range(N):
                disturbance_state[:,i] = (np.random.random(m.nx)-0.5) 
                disturbance_state[:,i] = np.random.random() * disturbance_state[:,i]/np.linalg.norm(disturbance_state[:,i])
            for i in range(N):
                disturbance_obs[:,i] = (np.random.random(m.nx)-0.5) 
                disturbance_obs[:,i] = np.random.random() * disturbance_obs[:,i]/np.linalg.norm(disturbance_obs[:,i])
        else:
            for i in range(N):
                disturbance_state[:,i] = (np.random.random(m.nx)-0.5) 
                disturbance_state[:,i] = disturbance_state[:,i]/np.linalg.norm(disturbance_state[:,i])
            for i in range(N):
                disturbance_obs[:,i] = (np.random.random(m.nx)-0.5) 
                disturbance_obs[:,i] = disturbance_obs[:,i]/np.linalg.norm(disturbance_obs[:,i])
        
        for t in range(1, N+1):
            delta_u = np.zeros((m.nu))
            if delay:
                # TODO: add a x(-1) to compute y(0) in this case
                if t == 1:
                    observations[:,t-1] = m.C @ states[:,t-1] + m.F @ disturbance_obs[:,t-1]
                else:
                    observations[:,t-1] = m.C @ states[:,t-2] + m.F @ disturbance_obs[:,t-1]
                
                delta_u += K_mat[(t-1)*m.nu:t*m.nu, 0:m.ny] @ (observations[:,0] - m.C @ primal_x[:,0])
                for j in range(1, t):
                    delta_u += K_mat[(t-1)*m.nu:t*m.nu, j*m.ny:(j+1)*m.ny] @ (observations[:,j] - m.C @ primal_x[:,j-1])
            else:
                observations[:,t-1] = m.C @ states[:,t-1] + m.F @ disturbance_obs[:,t-1]
                for j in range(t):
                    delta_u += K_mat[(t-1)*m.nu:t*m.nu, j*m.ny:(j+1)*m.ny] @ (observations[:,j] - m.C @ primal_x[:,j])
            
            u = primal_u[:,t-1] + delta_u
            for i in range(m.nu):
                if abs(u[i] - primal_u[i,t-1]) > tube[t-1, m.nx+i]:
                    print("warning: input exceeds tube")

            states[:,t] = np.array(m.ddyn(states[:,t-1], u, m.dt)).squeeze() + m.E @ disturbance_state[:,t-1]
        
        for t in range(1, N):
            for i in range(m.nx):
                if abs(states[i,t] - primal_x[i,t]) > tube[t,i]:
                    print("warning: actual state exceeds tube by {:.3f}".format(abs(states[i,t] - primal_x[i,t]) - tube[t,i]))

        for idx in range(m.nx):
            axes[idx].plot(np.arange(N+1), states[idx,:], c='r', linewidth=1)

def check_parameterization(fast_sls, N, nx, nu, ny, solution_fast_sls, delay=False):
    Z_downshift = SLS.get_block_downshift_matrix(N + 1, nx)
    I_blk = np.eye(nx * (N + 1))

    A_list = fast_sls.A_list
    B_list = fast_sls.B_list
    E_list = fast_sls.E_list
    C_list = fast_sls.C_list
    F_list = fast_sls.F_list

    A_blk = SLS.convert_list_to_blk_matrix(A_list)
    B_blk = SLS.convert_list_to_blk_matrix(B_list)
    C_blk = SLS.convert_list_to_blk_matrix(C_list)

    A_blk = block_diag(A_blk, np.zeros((nx, nx)))
    B_blk = block_diag(B_blk, np.zeros((nx, nu)))
    C_blk = block_diag(C_blk, np.zeros((ny, nx)))  

    Phi_xx_mat = solution_fast_sls['Phi_xx_mat']
    Phi_ux_mat = solution_fast_sls['Phi_ux_mat']
    Phi_xy_mat = solution_fast_sls['Phi_xy_mat']
    Phi_uy_mat = solution_fast_sls['Phi_uy_mat']

    # check if all elements are zeros with a 1e-6 tolerance
    assert np.all(np.abs((I_blk - Z_downshift @ A_blk) @ Phi_xx_mat - Z_downshift @ B_blk @ Phi_ux_mat - I_blk) < 1e-6)
    assert np.all(np.abs((I_blk - Z_downshift @ A_blk) @ Phi_xy_mat - Z_downshift @ B_blk @ Phi_uy_mat < 1e-6))
    if delay:
        assert np.all(np.abs(Phi_xx_mat @ (I_blk - Z_downshift @ A_blk) - Phi_xy_mat @ Z_downshift @ C_blk - I_blk < 1e-6))
        assert np.all(np.abs(Phi_ux_mat @ (I_blk - Z_downshift @ A_blk) - Phi_uy_mat @ Z_downshift @ C_blk < 1e-6))
    else:
        assert np.all(np.abs(Phi_xx_mat @ (I_blk - Z_downshift @ A_blk) - Phi_xy_mat @ C_blk - I_blk < 1e-6))
        assert np.all(np.abs(Phi_ux_mat @ (I_blk - Z_downshift @ A_blk) - Phi_uy_mat @ C_blk < 1e-6))
    print("parameterization correct")

def check_parameterization_separate(A, B, C, N, nx, nu, ny, Phi_x_f, Phi_u_f, Phi_x_o, Phi_y_o, delay):
    A_blk = SLS.convert_list_to_blk_matrix(A)
    A_blk = block_diag(A_blk, np.zeros((nx, nx)))
    B_blk = SLS.convert_list_to_blk_matrix(B)
    B_blk = block_diag(B_blk, np.zeros((nx, nu)))
    Z_downshift = SLS.get_block_downshift_matrix(N + 1, nx)
    I_blk = np.eye(nx * (N + 1))
    I_ZA = I_blk - Z_downshift @ A_blk
    C_blk = SLS.convert_list_to_blk_matrix(C)
    C_blk = block_diag(C_blk, np.zeros((ny, nx)))
    
    assert np.all(np.fabs(I_ZA @ SLS.convert_tensor_to_matrix(Phi_x_f) - Z_downshift @ B_blk @ SLS.convert_tensor_to_matrix(Phi_u_f) - I_blk) < 1e-6), \
            "state-feedback parameterization incorrect"
    if delay:
        assert np.all(np.fabs(SLS.convert_tensor_to_matrix(Phi_x_o) @ I_ZA - SLS.convert_tensor_to_matrix(Phi_y_o) @ Z_downshift @ C_blk - I_blk) < 1e-6), \
                "observer parameterization incorrect"
    else:
        assert(np.all(np.fabs(SLS.convert_tensor_to_matrix(Phi_x_o) @ I_ZA - SLS.convert_tensor_to_matrix(Phi_y_o) @ C_blk - I_blk) < 1e-6)), \
                "observer parameterization incorrect"


def optimize_observer(fast_of_sls):
    import cvxpy as cp

    nx = fast_of_sls.m.nx
    nu = fast_of_sls.m.nu
    nw = fast_of_sls.m.nw
    ny = fast_of_sls.m.ny
    N = fast_of_sls.N

    A_list = fast_of_sls.A_list
    B_list = fast_of_sls.B_list
    C_list = fast_of_sls.C_list

    # System Response Matrix
    Phi_x = [[cp.Variable((nx, nx), name='Phi_x_{},{}'.format(i, j)) for j in range(N + 1)] for i in range(N + 1)]
    Phi_y = [[cp.Variable((nx, ny), name='Phi_y_{},{}'.format(i, j)) for j in range(N + 1)] for i in range(N + 1)]

    Phi_x_mat = cp.bmat([[Phi_x[i][j] for j in range(N + 1)] for i in range(N + 1)])
    Phi_y_mat = cp.bmat([[Phi_y[i][j] for j in range(N + 1)] for i in range(N + 1)])

    # Define the cost function
    cost = 0

    # Add the system response matrix constraints in cost function
    Q_reg = np.sqrt(fast_of_sls.Q_reg)
    R_reg = np.sqrt(fast_of_sls.R_reg)
    Q_reg_f = np.sqrt(fast_of_sls.Q_reg_f)
    
    # TODO: take care of the regularizations
    Q_reg_blk = block_diag(np.kron(np.eye(N), Q_reg), Q_reg_f)
    R_reg_blk = np.kron(np.eye(N + 1), R_reg)
    # concatenate the Phi matrices
    # cost += cp.norm(block_diag(Q_reg_blk, R_reg_blk) @ Phi_mat, p='fro')
    # cost += cp.norm(Phi_x_mat, p='fro') + cp.norm(Phi_y_mat, p='fro')
    cost += cp.norm(Q_reg_blk @ Phi_x_mat, p='fro')**2
    for i in range(N+1):
        for j in range(N+1):
            # TODO: we assume I as cost for Phi_y now
            cost += cp.sum_squares(Phi_y[i][j])

    constraints = []
    # Define the constraints
    A_blk = SLS.convert_list_to_blk_matrix(A_list)
    B_blk = SLS.convert_list_to_blk_matrix(B_list)
    C_blk = SLS.convert_list_to_blk_matrix(C_list)

    # add a block at the end of A_blk
    A_blk = block_diag(A_blk, np.zeros((nx, nx)))
    B_blk = block_diag(B_blk, np.zeros((nx, nu)))
    C_blk = block_diag(C_blk, np.zeros((ny, nx)))

    Z_downshift = SLS.get_block_downshift_matrix(N + 1, nx)
    I_blk = np.eye(nx * (N + 1))

    # system response matrix constraints
    if fast_of_sls.delay:
        constraints += [Phi_x_mat @ (I_blk - Z_downshift @ A_blk) - Phi_y_mat @ Z_downshift @ C_blk == I_blk]
    else:
        constraints += [Phi_x_mat @ (I_blk - Z_downshift @ A_blk) - Phi_y_mat @ C_blk == I_blk]

    # add causal structure constraints
    for k in range(N):
        for j in range(N + 1):
            if j > k:
                constraints += [Phi_x[k][j] == np.zeros((nx, nx))]
                constraints += [Phi_y[k][j] == np.zeros((nx, ny))]

    # Define the problem
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.MOSEK)

    Phi_x_tensor = np.array([[
        Phi_x[i][j].value for j in range(len(Phi_x[0]))
    ] for i in range(len(Phi_x))])

    Phi_y_tensor = np.array([[
        Phi_y[i][j].value for j in range(len(Phi_y[0]))
    ] for i in range(len(Phi_y))])

    fast_of_sls.current_iteration['Phi_x_o_cvxpy'] = Phi_x_tensor
    fast_of_sls.current_iteration['Phi_y_o_cvxpy'] = Phi_y_tensor
    fast_of_sls.current_iteration["cost_cvxpy"] = cost.value

    return Phi_x_tensor, Phi_y_tensor

def compare_observer(fast_of_sls, Phi_x_o, Phi_y_o):
    Phi_x_o_cvxpy, Phi_y_o_cvxpy = optimize_observer(fast_of_sls)
    diff = np.max(np.fabs(Phi_x_o_cvxpy - Phi_x_o))
    diff = max(diff, np.max(np.fabs(Phi_y_o_cvxpy - Phi_y_o)))
    if diff > 1e-3:
        print(diff)
        for i in range(Phi_x_o.shape[0]):
            for j in range(Phi_x_o.shape[1]):
                if np.any(np.fabs(Phi_x_o_cvxpy[i,j,:,:] - Phi_x_o[i,j,:,:]) > 1e-3):
                    print("Phi_x_o does not match at ", i, " ", j)
                if np.any(np.fabs(Phi_y_o_cvxpy[i,j,:,:] - Phi_y_o[i,j,:,:]) > 1e-3):
                    print("Phi_y_o does not match at ", i, " ", j)
        print("warning: Riccati recursion does not match cvxpy for observer")
    else:
        print("observer matches cvxpy result")
