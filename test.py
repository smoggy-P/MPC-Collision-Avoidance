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
              [0, 0, 0 ]])

Bd= np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1],
              [0, 0, 0],
              [0, 0, 0],
              [0, 0, 0],
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

A_tilde=np.concatenate((np.concatenate((quadrotor_linear.A,Bd), axis=1),
                        np.concatenate((np.zeros((nb_disturbances,10)),np.eye(nb_disturbances)), axis=1)),axis=0)

# Find the eigenvalue from the characteristic polynomial
zo = 0.7        # damping ratio for the observer
wo = 1

# Define the eigen value we need for observers
eigs = np.roots([1, 2*zo*wo, wo**2])
eigs = np.append(eigs, np.roots([1, 2*zo*1.2, 1.2**2]))
eigs = np.append(eigs, np.roots([1, 2*1.2*2, 2**2]))
eigs = np.append(eigs, np.roots([1, 2*zo*3, 3**2]))
eigs = np.append(eigs, np.roots([1, 2*zo*4, 4**2]))
eigs = np.append(eigs, [-20, -60])


#►print("eigen values modulus of A-LC : ", np.abs(eig_val))

# Calculate estimator gain
L = control.acker(A_tilde.T, C_tilde.T, eigs).T
print("L=",L)
eig_val,eig_vec=np.linalg.eig(A_tilde-L@C_tilde)
print("eigen values of A-LC : ", eig_val)