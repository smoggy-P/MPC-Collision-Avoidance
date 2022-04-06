import numpy as np
import control
from Quadrotor import Quadrotor_linear

g=9.81
A_c = np.array([     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0        ],    #dx
                     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0        ],    #dy
                     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0        ],    #dz
                     [0, 0, 0, 0, 0, 0, 0,-g, 0, 0        ],    #ddx
                     [0, 0, 0, 0, 0, 0, g, 0, 0, 0        ],    #ddy
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0        ],    #ddz
                     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0        ],    #phi_dot
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1        ],    #theta_dot
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0        ],    #phi_ddot
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0        ]])   #theta_ddot
A= np.eye(10) + A_c * 0.1

C=np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])

O=control.obsv(A,C)

#print(O)
print("obs matrix rank = ",np.linalg.matrix_rank(O))
########
quadrotor_linear = Quadrotor_linear()


Cd= np.array([[0, 0, 0 ],
              [0, 0, 0 ],
              [0, 0, 0 ],
              [0, 0, 0 ],
              [0, 0, 0 ],
              [0, 0, 0 ],
              [0, 0, 0 ],
              [0, 0, 0 ],
              [0, 0, 0 ],
              [0, 0, 0 ]])

Bd= np.array([[0, 0, 0],
              [0, 0, 0],
              [0, 0, 0],
              [1, 0, 0],
              [0, 1, 0],
              [0, 0, 1],
              [0, 0, 0],
              [0, 0, 0],
              [0, 0, 0],
              [0, 0, 0]])
M=np.concatenate((np.concatenate((np.eye(10)-quadrotor_linear.A,-Bd), axis=1),
                  np.concatenate((quadrotor_linear.C,Cd), axis=1)),axis=0)


print("aug sys matrix rank = ",np.linalg.matrix_rank(M))


# array([[ 1.   ,  0.   ,  0.   ,  0.1  ,  0.   ,  0.   ,  0.   ,  0.   ,   0.   ,  0.   ],
#        [ 0.   ,  1.   ,  0.   ,  0.   ,  0.1  ,  0.   ,  0.   ,  0.   ,   0.   ,  0.   ],
#        [ 0.   ,  0.   ,  1.   ,  0.   ,  0.   ,  0.1  ,  0.   ,  0.   ,   0.   ,  0.   ],
#        [ 0.   ,  0.   ,  0.   ,  1.   ,  0.   ,  0.   ,  0.   ,  0.981,   0.   ,  0.   ],
#        [ 0.   ,  0.   ,  0.   ,  0.   ,  1.   ,  0.   , -0.981,  0.   ,   0.   ,  0.   ],
#        [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  1.   ,  0.   ,  0.   ,   0.   ,  0.   ],
#        [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  1.   ,  0.   ,   0.1  ,  0.   ],
#        [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  1.   ,   0.   ,  0.1  ],
#        [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,   1.   ,  0.   ],
#        [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,   0.   ,  1.   ]])
nb_disturbances=len(Cd[0])
nb_output=len(Cd)
L1=np.array([[ 1.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  1.   ,  0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  1.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  1.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ,  1.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  1.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ]])

L2=np.array([[ 1.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  1.   ,  0.   ,  0.   ,  0.   ,  0.   ],
             [ 0.   ,  0.   ,  1.   ,  0.   ,  0.   ,  0.   ]])
L=np.concatenate((L1,L2), axis=0)

C_tilde=np.concatenate((quadrotor_linear.C,Cd), axis=1)
A_tilde = np.block([[quadrotor_linear.A,               Bd                     ],
                    [np.zeros((nb_disturbances,10)),  np.eye(nb_disturbances) ]])

# Define the eigen value we need for observers
eigs = np.array([-0.1, -0.2, -0.2, -0.2, -0.3, -0.3, -0.3, -0.4, -0.4, -0.4, -0.5, -0.5, -0.5])
print(eigs)

# Calculate estimator gain
L = control.place(A_tilde.T, C_tilde.T, eigs).T
print("L=",L)
eig_val,eig_vec=np.linalg.eig(A_tilde-L@C_tilde)
print("eigen values of A-LC : ", eig_val)

######################## Test for terminal set
def get_terminal_set_corners(quadrotor_linear,goal,cst=0.0037):
    input_ub=0.07
    input_lb=-0.07
    Q = np.identity(10)
    R = 0.02*np.eye(4)
    P, L, K = control.dare(quadrotor_linear.A, quadrotor_linear.B, Q, R)
    eig_val,eig_vec=np.linalg.eig(P)
    
    corner_list=[]
    for a in range(2):
        for b in range(2):
            for c in range(2):
                x = (-1)**a *eig_vec[0]/np.linalg.norm(eig_vec[0])*eig_val[0]*cst
                x+= (-1)**b *eig_vec[1]/np.linalg.norm(eig_vec[1])*eig_val[1]*cst
                # eigenvalues 2 and 3 are the same, so I pick the one with a stronger influence on position
                # for better representation
                x+= (-1)**c *eig_vec[3]/np.linalg.norm(eig_vec[3])*eig_val[3]*cst
                corner=np.array(goal)+np.array(x)[:3]
                corner_list.append(corner)
    return corner_list
    
cst=0.0037

input_ub=0.07
input_lb=-0.07
Q = np.identity(10)
R = 0.02*np.eye(4)

P, L, K = control.dare(quadrotor_linear.A, quadrotor_linear.B, Q, R)
eig_val,eig_vec=np.linalg.eig(P)


for a in range(2):
    for b in range(2):
        for c in range(2):
            for d in range(2):
                for e in range(2):
                    for f in range(2):
                        for g in range(2):
                            for h in range(2):
                                for i in range(2):
                                    for j in range(2):
                                        x = (-1)**a *eig_vec[0]/np.linalg.norm(eig_vec[0])*eig_val[0]*cst
                                        x+= (-1)**b *eig_vec[1]/np.linalg.norm(eig_vec[1])*eig_val[1]*cst
                                        x+= (-1)**c *eig_vec[2]/np.linalg.norm(eig_vec[2])*eig_val[2]*cst
                                        x+= (-1)**d *eig_vec[3]/np.linalg.norm(eig_vec[3])*eig_val[3]*cst
                                        x+= (-1)**e *eig_vec[4]/np.linalg.norm(eig_vec[4])*eig_val[4]*cst
                                        x+= (-1)**f *eig_vec[5]/np.linalg.norm(eig_vec[5])*eig_val[5]*cst
                                        x+= (-1)**g *eig_vec[6]/np.linalg.norm(eig_vec[6])*eig_val[6]*cst
                                        x+= (-1)**h *eig_vec[7]/np.linalg.norm(eig_vec[7])*eig_val[7]*cst
                                        x+= (-1)**i *eig_vec[8]/np.linalg.norm(eig_vec[8])*eig_val[8]*cst
                                        x+= (-1)**j *eig_vec[9]/np.linalg.norm(eig_vec[9])*eig_val[9]*cst
                                        #print("***")
                                        print(np.dot(K,x)<=input_ub)
                                        #if np.any(K@x<input_lb):
                                        #    print(a,b,c,d,e,f,g,h,i,j)
                                        print(np.dot(K,x)>=input_lb)
                                        