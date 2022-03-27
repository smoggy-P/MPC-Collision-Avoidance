import numpy as np
class Quadrotor_linear():
    def __init__(self):
        """
        Define model parameters and state space for linear quadrotor dynamics
        """

        self.mass            =  0.030  # kg
        self.Ixx             = 1.43e-5  # kg*m^2
        self.Iyy             = 1.43e-5  # kg*m^2
        self.Izz             = 2.89e-5  # kg*m^2
        self.arm_length      = 0.046  # meters
        self.k_thrust        = 2.3e-08  # N/(rad/s)**2
        self.k_drag          = 7.8e-11   # Nm/(rad/s)**2
        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz])) # kg*m^2
        g = 9.81 # m/s^2

        k = self.k_drag/self.k_thrust
        L = self.arm_length
        self.to_TM = np.array([[1,  1,  1,  1],
                               [ 0,  L,  0, -L],
                               [-L,  0,  L,  0]])
        self.t_step = 0.5

        """
        Continuous state space without considering yawing(assuming yaw angle is 0 for all the time)
        state = [x y z dx dy dz phi theta phi_dot theta_dot]
        dot_state = [dx dy dz ddx ddy ddz phi_dot theta_dot phi_ddot theta_ddot]
        u = [F1 F2 F3 F4]
        """
        #                     x  y  z  dx dy dz phi theta phi_dot theta_dot
        self.A_c = np.array([[0, 0, 0, 1, 0, 0, 0,  0,    0,      0        ],    #dx
                             [0, 0, 0, 0, 1, 0, 0,  0,    0,      0        ],    #dy
                             [0, 0, 0, 0, 0, 1, 0,  0,    0,      0        ],    #dz
                             [0, 0, 0, 0, 0, 0, 0,  -g,   0,      0        ],    #ddx
                             [0, 0, 0, 0, 0, 0, g,  0,    0,      0        ],    #ddy
                             [0, 0, 0, 0, 0, 0, 0,  0,    0,      0        ],    #ddz
                             [0, 0, 0, 0, 0, 0, 0,  0,    1,      0        ],    #phi_dot
                             [0, 0, 0, 0, 0, 0, 0,  0,    0,      1        ],    #theta_dot
                             [0, 0, 0, 0, 0, 0, 0,  0,    0,      0        ],    #phi_ddot
                             [0, 0, 0, 0, 0, 0, 0,  0,    0,      0        ]])   #theta_ddot
        self.B_c = np.array([[0, 0, 0], #dx
                             [0, 0, 0], #dy
                             [0, 0, 0], #dz
                             [0, 0, 0], #ddx
                             [0, 0, 0], #ddy
                             [1/self.mass, 0, 0], #ddz
                             [0, 0, 0],           #phi_dot
                             [0, 0, 0],           #theta_dot
                             [0, 1/self.Ixx, 0],  #phi_ddot
                             [0, 0, 1/self.Iyy]  #theta_ddot
                             ]) @ self.to_TM
        self.C_c = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        self.D_c = np.zeros((1,4))

        # Discretization state space
        self.A = np.eye(10) + self.A_c * self.t_step
        self.B = self.B_c * self.t_step
        self.C = self.C_c
        self.D = self.D_c


    def next_x(self, x, u):
        return self.A.dot(x).reshape(-1,1) + self.B.dot(u)
