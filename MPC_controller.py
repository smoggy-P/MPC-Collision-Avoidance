import cvxpy as cp
import numpy as np
from numpy import newaxis
from convexification import obstacle_list, convexify

def mpc_control(quadrotor_linear, N, x_init, x_target,A_obs,b_obs):
    weight_input = 0.2*np.eye(4)    # Weight on the input
    
    cost = 0.
    constraints = []
    
    # Create the optimization variables
    x = cp.Variable((10, N + 1)) # cp.Variable((dim_1, dim_2))
    u = cp.Variable((4, N))

    # Get constraints from obstacle list
    

    # For each stage in the MPC horizon
    Q = np.identity(10)
    for n in range(N):
        cost += (cp.quad_form((x[:,n+1]-x_target),Q)  + cp.quad_form(u[:,n], weight_input))
        constraints += [x[:,n+1] == quadrotor_linear.A @ x[:,n] + quadrotor_linear.B @ u[:,n]]
        constraints += [A_obs @ x[:2,n] <= b_obs.flatten()]
    # Implement the cost components and/or constraints that need to be added once, here
    constraints += [x[:,0] == x_init.flatten()]
    
    # Solves the problem
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(solver=cp.OSQP)

    # We return the MPC input
    return u[:, 0].value

