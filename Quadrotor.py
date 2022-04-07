from cmath import cos, sin
import numpy as np
import scipy.integrate
import math
from scipy.spatial.transform import Rotation
from numpy.linalg import inv

class Quadrotor_linear():
    def __init__(self):
        """
        Define model parameters and state space for linear quadrotor dynamics
        """

        self.mass            =  0.90  # kg
        self.Ixx             = 30*1.43e-5  # kg*m^2
        self.Iyy             = 30*1.43e-5  # kg*m^2
        self.Izz             = 30*2.89e-5  # kg*m^2
        self.arm_length      = 0.046  # meters
        g = 9.81 # m/s^2
        L = self.arm_length
        self.to_TM = np.array([[1,  1,  1,  1],
                               [ 0,  L,  0, -L],
                               [-L,  0,  L,  0]])
        self.t_step = 0.04

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
                             [0, 0, 0, 0, 0, 0, 0,  g,    0,      0        ],    #ddx
                             [0, 0, 0, 0, 0, 0, -g, 0,    0,      0        ],    #ddy
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
        self.C_c = np.identity(10)
        self.D_c = np.zeros((1,4))

        # Discretization state space
        self.A = np.eye(10) + self.A_c * self.t_step
        self.B = self.B_c * self.t_step
        self.C = self.C_c
        self.D = self.D_c

    def next_x(self, x, u):
        return self.A.dot(x).reshape(-1,1) + self.B.dot(u)
    
    def disturbed_next_x(self,x,u,real_d,Bd):
        #print("xxx")
        #print(x.shape)
        #print((self.A.dot(x).reshape(-1,1)).shape)
        #print((Bd @ real_d).shape)
        return self.A @ x.reshape(-1,1) + self.B @ u.reshape(-1,1) + Bd @ real_d.reshape(-1,1)
    
    def disturbed_output(self,x,real_d, Cd, sigma_noise):
        return self.C @ x.reshape(-1,1) + Cd @ real_d.reshape(-1,1) + np.random.normal(loc=np.zeros((1,10)),scale=sigma_noise).reshape(10,1)
    
    
    def from_nonlinear(self, quadrotor):
        linearized_state = np.zeros(10).reshape(-1,1)
        linearized_state[:8] = quadrotor.state[:8].reshape(-1,1)
        linearized_state[8:] = quadrotor.state[9:11].reshape(-1,1)
        return linearized_state

class Quadrotor():
    """
    Quadrotor forward dynamics model.
    """
    def __init__(self):
        self.mass            =  0.030  # kg
        self.Ixx             = 1.43e-5  # kg*m^2
        self.Iyy             = 1.43e-5  # kg*m^2
        self.Izz             = 2.89e-5  # kg*m^2
        self.arm_length      = 0.046  # meters

        #TODO: change force limit
        self.cmd_rotor_forces_min = -0.7  # N
        self.cmd_rotor_forces_max = 0.7  # N
        self.k_thrust        = 2.3e-08  # N/(rad/s)**2
        self.k_drag          = 7.8e-11   # Nm/(rad/s)**2
        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz])) # kg*m^2
        self.g = 9.81 # m/s^2
        # Precomputes
        k = self.k_drag/self.k_thrust
        L = self.arm_length
        self.to_TM = np.array([[1,  1,  1,  1],
                               [ 0,  L,  0, -L],
                               [-L,  0,  L,  0],
                               [ k, -k,  k, -k]])
        self.inv_inertia = np.linalg.inv(self.inertia)
        #self.weight = np.array([0, 0, -self.mass*self.g])
        self.weight = np.array([0, 0, 0])
        self.t_step = 0.05
        # self.inv_inertia = inv(self.inertia)
        # self.weight = np.array([0, 0, 0])
        # self.t_step = 0.05
    def reset(self, position=[0, 0, 0], yaw =0, pitch=0, roll=0):
        '''
        state is a 12 dimensional vector
            postion*3 velocity*3 attitude(rpy)*3 angular velocity*3
        state = [x y z dx dy dz psi theta phi p q r]
        dot_state = [dx dy dz ddx ddy ddz dqw dqx dqy dqz dr dp dq]
        '''
        s = np.zeros(12)
        s[0] = position[0]
        s[1] = position[1]
        s[2] = position[2]
        s[6] = pitch
        s[7] = roll
        s[8] = yaw
        self.state = s
        return self.state

    def step(self, cmd_rotor_forces):
        '''
        Considering the max and min of rotor forces
        action is a 4 dimensional vector: conmmand rotor forces
        action = [F1, F2, F3, F4]
        '''
        rotor_thrusts = np.clip(cmd_rotor_forces, self.cmd_rotor_forces_min, self.cmd_rotor_forces_max)
        TM = self.to_TM @ cmd_rotor_forces
        T = TM[0]
        M = TM[1:4].flatten()
        # Form autonomous ODE for constant inputs and integrate one time step.
        def s_dot_fn(t, s):
            return self._s_dot_fn(t, s, T, M)
        '''
        The next state can be obtained through integration （Runge-Kutta）
        '''
        s = self.state
        sol = scipy.integrate.solve_ivp(s_dot_fn, (0, self.t_step), s, first_step=self.t_step)
        self.state = sol['y'][:,-1]
        return self.state

    def _s_dot_fn(self, t, s, u1, u2):
        """
        Compute derivative of state for quadrotor given fixed control inputs as
        an autonomous ODE.
        """
        state = self.state
        x_dot = state[3:6]
        v_dot = np.array([u1[0]/self.mass * (cos(state[8]) * sin(state[7]) + cos(state[7]) * sin(state[6]) * sin(state[8])), 
                          u1[0]/self.mass * (sin(state[8]) * sin(state[7]) - cos(state[8]) * cos(state[7]) * sin(state[6])), 
                          u1[0]/self.mass * (cos(state[6]) * cos(state[7]))])
        q_dot = state[9:]
        q_dot_hat = Quadrotor.hat_map(q_dot)
        w_dot = self.inv_inertia @ (u2 - q_dot_hat @ (self.inertia @ q_dot))

        # Pack into vector of derivatives.
        s_dot = np.zeros((12,))
        s_dot[0:3] = x_dot
        s_dot[3:6] = v_dot
        s_dot[6:9] = q_dot
        s_dot[9:] = w_dot
        return s_dot
    @classmethod
    def hat_map(cls, s):
        """
        Given vector s in R^3, return associate skew symmetric matrix S in R^3x3
        """
        return np.array([[0, -s[2], s[1]],
                         [s[2], 0, -s[0]],
                         [-s[1], s[0], 0]])