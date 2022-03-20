import numpy as np
from Quadrotor import Quadrotor_linear
from MPC_controller import mpc_control
from visualization import visualization


if __name__ == "__main__":
    quadrotor = Quadrotor_linear()

    x_init = np.zeros(10)
    x_target = np.zeros(10)
    x_target[0] = 2
    x_target[1] = 2
    x_target[2] = 2
    x_next = x_init
    real_trajectory = {'x': [], 'y': [], 'z': []}


    for i in range(10):
        u = mpc_control(quadrotor, 10, x_next, x_target).reshape(-1,1)

        real_trajectory['x'].append(x_next[0])
        real_trajectory['y'].append(x_next[1])
        real_trajectory['z'].append(x_next[2])
        x_next = quadrotor.next_x(x_next, u)

    visualization(real_trajectory)
    