import cvxpy as cp
import numpy as np
from numpy import newaxis
import control

def get_observer_gain(quadrotor_linear, Bd,Cd,eigen_values):
    nb_disturbances=len(Cd[0])
    nb_output=len(Cd)
    
    C_tilde=np.concatenate((quadrotor_linear.C,Cd), axis=1)
    A_tilde = np.block([[quadrotor_linear.A,               Bd                     ],
                        [np.zeros((nb_disturbances,10)),  np.eye(nb_disturbances) ]])
    L = control.place(A_tilde.T, C_tilde.T, eigen_values).T
    
    return L
    
def luenberger_observer(quadrotor_linear, x_hat, d_hat, y, u, Bd, Cd, L):
    nb_disturbances=len(Cd[0])
    nb_output=len(Cd)
    nb_input=len(u)
    
    C_tilde=np.block([quadrotor_linear.C, Cd])

    A_tilde=np.block([[quadrotor_linear.A,              Bd                     ],
                      [np.zeros((nb_disturbances,10)),  np.eye(nb_disturbances)]])

    cur_state = np.block([[x_hat.reshape(-1,1)],
                          [d_hat.reshape(-1,1)]])

    new_state  = A_tilde @ cur_state

    new_state += np.block([[quadrotor_linear.B                  ], 
                           [np.zeros((nb_disturbances,nb_input))]]) @ u

    new_state += L @ (y.reshape(-1,1) - C_tilde @ cur_state)

    # print("real output:", y.flatten())

    # print("estimated output:",(C_tilde @ cur_state).flatten())

    # print("output error:", (y.reshape(-1,1) - C_tilde @ cur_state).flatten())

    # print("observer:", new_state[:10].flatten())

    return new_state[:10], new_state[10:]


def mpc_control(quadrotor_linear, N, x_init, x_target, A_obs, b_obs):
    weight_input = 0 * np.eye(4)    # Weight on the input
    
    cost = 0.
    constraints = []
    
    # Create the optimization variables
    x = cp.Variable((10, N + 1)) 
    u = cp.Variable((4, N))

    # For each stage in the MPC horizon
    Q = np.diag([1,1,1,1,1,1,1,1,1,1])
    for n in range(N):
        cost += (cp.quad_form((x[:,n+1]-x_target),Q)  + cp.quad_form(u[:,n], weight_input))
        constraints += [x[:,n+1] == quadrotor_linear.A @ x[:,n] + quadrotor_linear.B @ u[:,n]]

        constraints += [x[6,n+1] <= 0.01]
        constraints += [x[7,n+1] <= 0.01]
        constraints += [x[6,n+1] >= -0.01]
        constraints += [x[7,n+1] >= -0.01]
        constraints += [u[:,n] >= -0.07]
        constraints += [u[:,n] <= 0.07]
        constraints += [A_obs @ x[:2,n] <= b_obs.flatten()]
    # Implement the cost components and/or constraints that need to be added once, here
    constraints += [x[:,0] == x_init.flatten()]
    
    
    # Solves the problem
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(solver=cp.OSQP)
    # We return the MPC input
    return u[:, 0].value

def mpc_control_stable(quadrotor, N, x_init, x_target, A_obs,b_obs,c=1):
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
