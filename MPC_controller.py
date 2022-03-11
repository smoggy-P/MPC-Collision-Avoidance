import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from Quadrotor import Quadrotor_linear

def mpc_control(quadrotor, N, x_init, x_target):
    weight_input = 0.2*np.eye(1)    # Weight on the input
    weight_tracking = 1.0           # Weight on the tracking state
    
    cost = 0.
    constraints = []
    
    # Create the optimization variables
    x = cp.Variable((2, N + 1)) # cp.Variable((dim_1, dim_2))
    u = cp.Variable((1, N))

    # For each stage in the MPC horizon
    Q = np.array([[weight_tracking, 0], [0,0]])
    for n in range(N):
        # EXERCISE: Implement the cost components and/or constraints that need to be satisfied for each step, here   
        cost += (cp.quad_form((x[:,n+1]-x_target),Q)  + cp.quad_form(u[:,n], weight_input))
        constraints += [x[0,n+1] >= -1]
        constraints += [x[1,n+1] >= -10]
        constraints += [x[0,n+1] <= 100]
        constraints += [x[1,n+1] <= 25]
        constraints += [u[:,n] <= 4]
        constraints += [u[:,n] >= -4]
        constraints += [x[:,n+1] == quadrotor.A @ x[:,n] + quadrotor.B @ u[:,n]]

    # Implement the cost components and/or constraints that need to be added once, here
    constraints += [x[:,0] == x_init]
    
    # Solves the problem
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(solver=cp.OSQP)

    # We return the MPC input and the next state (and also the plan for visualization)
    return u[:, 0].value, x[:, 1].value, x[:, :].value, None
