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
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])

O=control.obsv(A,C)

#print(O)
print("obs matrix rank = ",np.linalg.matrix_rank(O))
########
quadrotor_linear = Quadrotor_linear()

Cd= np.array([[1, 0],
              [0, 1],
              [0, 0]])
Bd= np.array([[1, 0],
              [0, 1],
              [0, 0],
              [0, 0],
              [0, 0],
              [0, 1],
              [0, 0],
              [0, 0],
              [1, 0],
              [0, 0]])
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
