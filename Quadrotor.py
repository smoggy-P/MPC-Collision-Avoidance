import numpy as np
import scipy.integrate
from scipy.spatial.transform import Rotation

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
        self.t_step = 0.1

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



def quat_dot(quat, omega):
    """
    Parameters:
        quat, [i,j,k,w]
        omega, angular velocity of body in body axes
    Returns
        quat_dot, [i,j,k,w]
    """
    # Adapted from "Quaternions And Dynamics" by Basile Graf.
    (q0, q1, q2, q3) = (quat[0], quat[1], quat[2], quat[3])
    G = np.array([[ q3,  q2, -q1, -q0],
                  [-q2,  q3,  q0, -q1],
                  [ q1, -q0,  q3, -q2]])
    quat_dot = 0.5 * G.T @ omega
    # Augment to maintain unit quaternion.
    quat_err = np.sum(quat**2) - 1
    quat_err_grad = 2 * quat
    quat_dot = quat_dot - quat_err * quat_err_grad
    return quat_dot

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
        self.cmd_rotor_forces_min = 0  # rad/s
        self.cmd_rotor_forces_max = 2500  # rad/s
        self.k_thrust        = 2.3e-08  # N/(rad/s)**2
        self.k_drag          = 7.8e-11   # Nm/(rad/s)**2
        # Additional constants.
        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz])) # kg*m^2
        self.g = 9.81 # m/s^2
        # Precomputes
        k = self.k_drag/self.k_thrust
        L = self.arm_length
        self.to_TM = np.array([[1,  1,  1,  1],
                               [ 0,  L,  0, -L],
                               [-L,  0,  L,  0],
                               [ k, -k,  k, -k]])
        self.inv_inertia = inv(self.inertia)
        self.weight = np.array([0, 0, -self.mass*self.g])
        self.t_step = 0.1
    def reset(self, position=[0, 0, 0], yaw =0, pitch=0, roll=0):
        '''
        state is a 13 dimensional vector
            postion*3 velocity*3 attitude(quaternion)*4 angular velocity*3
        state = [x y z dx dy dz qw qx qy qz r p q]
        dot_state = [dx dy dz ddx ddy ddz dqw dqx dqy dqz dr dp dq]
        '''
        s = np.zeros(13)
        s[0] = position[0]
        s[1] = position[1]
        s[2] = position[2]
        r = Rotation.from_euler('zxy', [yaw, roll, pitch], degrees=True)
        quat = r.as_quat()
        s[6] = quat[0]
        s[7] = quat[1]
        s[8] = quat[2]
        s[9] = quat[3]
        self.state = self._unpack_state(s)
        return self.state
    def step(self, cmd_rotor_forces):
        '''
        Considering the max and min of rotor forces
        action is a 4 dimensional vector: conmmand rotor forces
        action = [F1, F2, F3, F4]
        '''
        rotor_thrusts = np.clip(cmd_rotor_forces, self.cmd_rotor_forces_min, self.cmd_rotor_forces_max)
        TM = self.to_TM @ rotor_thrusts
        T = TM[0]
        M = TM[1:4]
        # Form autonomous ODE for constant inputs and integrate one time step.
        def s_dot_fn(t, s):
            return self._s_dot_fn(t, s, T, M)
        '''
        The next state can be obtained through integration （Runge-Kutta）
        '''
        s = Quadrotor._pack_state(self.state)
        sol = scipy.integrate.solve_ivp(s_dot_fn, (0, self.t_step), s, first_step=self.t_step)
        s = sol['y'][:,-1]
        # turn state back to dict
        self.state = Quadrotor._unpack_state(s)
        # Re-normalize unit quaternion.
        reward = 0
        done = 0
        info = {}
        return self.state, reward, done, info
    def _s_dot_fn(self, t, s, u1, u2):
        """
        Compute derivative of state for quadrotor given fixed control inputs as
        an autonomous ODE.
        """
        state = Quadrotor._unpack_state(s)
        # page 73
        # Position derivative.
        x_dot = state['v']
        # Velocity derivative.
        F = u1 * Quadrotor.rotate_k(state['q'])
        v_dot = (self.weight + F) / self.mass
        # Orientation derivative.
        q_dot = quat_dot(state['q'], state['w'])
        # Angular velocity derivative. page 26 Equation 4
        omega = state['w']
        omega_hat = Quadrotor.hat_map(omega)
        w_dot = self.inv_inertia @ (u2 - omega_hat @ (self.inertia @ omega))
        # Pack into vector of derivatives.
        s_dot = np.zeros((13,))
        s_dot[0:3] = x_dot
        s_dot[3:6] = v_dot
        s_dot[6:10] = q_dot
        s_dot[10:13] = w_dot
        return s_dot
    @classmethod
    def rotate_k(cls, q):
        """
        Rotate the unit vector k by quaternion q. This is the third column of
        the rotation matrix associated with a rotation by q.
        """
        return np.array([2 * (q[0] * q[2] + q[1] * q[3]),
                         2 * (q[1] * q[2] - q[0] * q[3]),
                         1 - 2 * (q[0] ** 2 + q[1] ** 2)])
    @classmethod
    def hat_map(cls, s):
        """
        Given vector s in R^3, return associate skew symmetric matrix S in R^3x3
        """
        return np.array([[0, -s[2], s[1]],
                         [s[2], 0, -s[0]],
                         [-s[1], s[0], 0]])
    @classmethod
    def _pack_state(cls, state):
        """
        Convert a state dict to Quadrotor's private internal vector representation.
        """
        s = np.zeros((13,))
        s[0:3] = state['x']
        s[3:6] = state['v']
        s[6:10] = state['q']
        s[10:13] = state['w']
        return s
    @classmethod
    def _unpack_state(cls, s):
        """
        Convert Quadrotor's private internal vector representation to a state dict.
        """
        state = {'x': s[0:3], 'v': s[3:6], 'q': s[6:10], 'w': s[10:13]}
        return state