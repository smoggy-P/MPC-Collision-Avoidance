import cvxpy as cp
import numpy as np
from numpy import newaxis
import control

def OTS(quadrotor_linear,y_ref,d_hat, A_obs,b_obs):

    cost = 0.
    constraints = []
    
    # Create the optimization variables
    x_ref = cp.Variable((10, 1)) # cp.Variable((dim_1, dim_2))
    u_ref = cp.Variable((4, 1))
    
    #Cost : we minimize the input u_ref
    cost+=cp.sum_squares(u_ref)
    # Add constraints
    
    constraints+=[A_obs[:2,:] @ x_ref <= b_obs[:,np.newaxis]] 


    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve()
    return x_ref.value


def mpc_control(quadrotor_linear, N, x_init, x_target,A_obs,b_obs):
    weight_input = 0*np.eye(4)    # Weight on the input
    
    cost = 0.
    constraints = []
    
    # Create the optimization variables
    x = cp.Variable((10, N + 1)) # cp.Variable((dim_1, dim_2))
    u = cp.Variable((4, N))

    # Get constraints from obstacle list
    

    # For each stage in the MPC horizon
    Q = np.diag([1,1,1,1,1,1,1,1,1,1])
    for n in range(N):
        cost += (cp.quad_form((x[:,n+1]-x_target),Q)  + cp.quad_form(u[:,n], weight_input))
        constraints += [x[:,n+1] == quadrotor_linear.A @ x[:,n] + quadrotor_linear.B @ u[:,n]]
        constraints += [x[6,n+1] <= 0.5]
        constraints += [x[7,n+1] <= 0.5]
        constraints += [x[6,n+1] >= -0.5]
        constraints += [x[7,n+1] >= -0.5]
        constraints += [u[:,n] >= -20]
        constraints += [u[:,n] <= 20]
        constraints += [A_obs @ x[:2,n] <= b_obs.flatten()]
    # Implement the cost components and/or constraints that need to be added once, here
    constraints += [x[:,0] == x_init.flatten()]
    
    
    # Solves the problem
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(solver=cp.OSQP)
    # We return the MPC input
    return u[:, 0].value

def mpc_control_stable(quadrotor, N, x_init, x_target,A_obs,b_obs,c=1):
    weight_input = 0.02*np.eye(4)    # Weight on the input
    
    cost = 0.
    constraints = []
    
    # Create the optimization variables
    x = cp.Variable((10, N + 1)) # cp.Variable((dim_1, dim_2))
    u = cp.Variable((4, N))

    # Get constraints from obstacle list
    

    # For each stage in the MPC horizon
    Q = np.identity(10)
    R = weight_input
    
    P, L, K = control.dare(quadrotor.A, quadrotor.B, Q, R)
    eig_val,eig_vec=np.linalg.eig(P)
    #print(eig_val,eig_vec)
    for n in range(N):
        cost += (cp.quad_form((x[:,n+1]-x_target),Q)  + cp.quad_form(u[:,n], weight_input))
        constraints += [x[:,n+1] == quadrotor.A @ x[:,n] + quadrotor.B @ u[:,n]]
        constraints += [A_obs @ x[:2,n] <= b_obs.flatten()]
    # Implement the cost components and/or constraints that need to be added once, here
    cost+=cp.quad_form((x[:,N]-x_target),P)
    constraints += [x[:,0] == x_init.flatten()]
    
    for j in range(len(eig_val)):
        #print("*************")
        #print(np.shape(eig_vec[j]))
        #print((x[:,N]-x_target)  @ eig_vec[j])
        constraints+= [(x[:,N]-x_target)  @ eig_vec[j]<=eig_val[j]*c]
    #constraints += [cp.quad_form((x[:,N]-x_target),P) <= c]
    
    # Solves the problem
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(solver=cp.OSQP)

    # We return the MPC input
    return u[:, 0].value
