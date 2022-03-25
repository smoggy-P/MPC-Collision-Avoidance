import cvxpy as cp
import numpy as np
from numpy import newaxis
from convexification import obstacle_list, convexify, get_intermediate_goal

def mpc_control(quadrotor, N, x_init, x_target):
    weight_input = 0.2*np.eye(4)    # Weight on the input
    
    cost = 0.
    constraints = []
    
    # Create the optimization variables
    x = cp.Variable((10, N + 1)) # cp.Variable((dim_1, dim_2))
    u = cp.Variable((4, N))

    # Get constraints from obstacle list
    A_obs, b_obs= convexify(x_init[:2].flatten(), 0.5, obstacle_list)
    x_inter = get_intermediate_goal(x_target[:2], A_obs, b_obs)
    # For each stage in the MPC horizon
    Q = np.identity(2)
    for n in range(N):
        cost += (cp.quad_form((x[:2,n+1]-x_inter.flatten()),Q)  + cp.quad_form(u[:,n], weight_input))
        constraints += [x[:,n+1] == quadrotor.A @ x[:,n] + quadrotor.B @ u[:,n]]
        constraints += [A_obs @ x[:2,n] <= b_obs.flatten()]
    # Implement the cost components and/or constraints that need to be added once, here
    constraints += [x[:,0] == x_init.flatten()]
    
    # Solves the problem
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(solver=cp.OSQP)

    # We return the MPC input
    return u[:, 0].value

