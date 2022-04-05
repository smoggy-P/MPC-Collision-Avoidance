import cvxpy as cp
import numpy as np
from numpy import newaxis
import control

def OTS(quadrotor_linear,y_ref,d_hat, A_obs,b_obs,Bd,Cd):

    cost = 0.
    constraints = []
    
    # Create the optimization variables
    x_ref = cp.Variable((10, 1)) # cp.Variable((dim_1, dim_2))
    u_ref = cp.Variable((4, 1))
    
    #Cost : we minimize the input u_ref
    #
    cost+=cp.sum_squares(u_ref)
    cost+=cp.sum_squares(y_ref[:,np.newaxis]-x_ref)
    # Add constraints
    #M=np.concatenate((np.concatenate((np.eye(10)-quadrotor_linear.A,-quadrotor_linear.B), axis=1),
    #                  np.concatenate((quadrotor_linear.C,np.zeros((len(y_ref),4))), axis=1)),axis=0)
    
    #constraints+= [M @ np.concatenate((x_ref,u_ref),axis=0)== np.concatenate((Bd @ d_hat,y_ref-Cd @ d_hat),axis=0)]
    #print(y_ref)
    #print("lol :",Bd @ d_hat)
    constraints += [(np.eye(10)-quadrotor_linear.A)@ x_ref - quadrotor_linear.B @ u_ref == Bd @ d_hat]
    constraints += [quadrotor_linear.C @ x_ref == y_ref[:,np.newaxis] - Cd @ d_hat]
    # admissible x_ref and u_ref
    constraints += [A_obs[:,:] @ x_ref[:2] <= b_obs[:,np.newaxis]] 
    constraints += [x_ref[6] <= 0.5]
    constraints += [x_ref[7] <= 0.5]
    constraints += [x_ref[6] >= -0.5]
    constraints += [x_ref[7] >= -0.5]
    constraints += [u_ref[:] >= -20]
    constraints += [u_ref[:] <= 20]
    
    #constraints += [A_obs[:2,:] @ (quadrotor_linear.C @ x_ref + Cd @ d_hat)[:2] <= b_obs[:,np.newaxis]]
    
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve()
    return x_ref.value,u_ref.value

def get_observer_gain(quadrotor_linear, Bd,Cd,eigen_values):
    nb_disturbances=len(Cd[0])
    nb_output=len(Cd)
    
    C_tilde=np.concatenate((quadrotor_linear.C,Cd), axis=1)
    A_tilde = np.block([[quadrotor_linear.A,               Bd                     ],
                        [np.zeros((nb_disturbances,10)),  np.eye(nb_disturbances) ]])
    L = control.place(A_tilde.T, C_tilde.T, eigen_values).T
    
    return L
    
def luenberger_observer(quadrotor_linear, x_hat,d_hat,y,u,Bd,Cd,L):
    nb_disturbances=len(Cd[0])
    nb_output=len(Cd)
    nb_input=len(u)
    
    C_tilde=np.concatenate((quadrotor_linear.C,Cd), axis=1)

    A_tilde=np.concatenate((np.concatenate((quadrotor_linear.A,Bd), axis=1),
                            np.concatenate((np.zeros((nb_disturbances,10)),np.eye(nb_disturbances)), axis=1)),axis=0)
    new_state  = A_tilde @ np.concatenate((x_hat.reshape(-1,1),d_hat.reshape(-1,1)),axis=0)
    new_state += np.concatenate((quadrotor_linear.B,np.zeros((nb_disturbances,nb_input))),axis=0) @ u
    #print((new_state).shape)
    #print((L@y[:,np.newaxis]-(L@(C_tilde @ np.concatenate((x_hat.reshape(-1,1),d_hat.reshape(-1,1)),axis=0)))).shape)
    #print((L@(C_tilde @ np.concatenate((x_hat.reshape(-1,1),d_hat.reshape(-1,1)),axis=0))).shape)
    new_state += L @ (y[:,np.newaxis]-C_tilde @ np.concatenate((x_hat.reshape(-1,1),d_hat.reshape(-1,1)),axis=0))
    
    return new_state[:10],new_state[10:]


def mpc_control(quadrotor_linear, N, x_init, x_target,u_target, A_obs,b_obs):
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
        cost += (cp.quad_form((x[:,n+1]-x_target),Q)  + cp.quad_form(u[:,n]-u_target, weight_input))
        constraints += [x[:,n+1] == quadrotor_linear.A @ x[:,n] + quadrotor_linear.B @ u[:,n]]

        constraints += [x[6,n+1] <= 0.5]
        constraints += [x[7,n+1] <= 0.5]
        constraints += [x[6,n+1] >= -0.5]
        constraints += [x[7,n+1] >= -0.5]
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

def mpc_control_stable(quadrotor, N, x_init, x_target,u_target, A_obs,b_obs,c=1):
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
        cost += (cp.quad_form((x[:,n+1]-x_target),Q)  + cp.quad_form(u[:,n]-u_target, weight_input))
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
