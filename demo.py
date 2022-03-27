import numpy as np
from Quadrotor import Quadrotor_linear
from MPC_controller import mpc_control
from visualization import visualization
from convexification import obstacle_list
from convexification import get_intermediate_goal
from convexification import convexify
if __name__ == "__main__":
    quadrotor = Quadrotor_linear()

    x_init = np.zeros(10)
    x_target = np.zeros(10)
    x_target[0] = 6
    x_target[1] = -5
    x_target[2] = 2
    x_next = x_init
    A,b=convexify(x_init[:2].flatten(),0.5,obstacle_list)
    inter_goal=get_intermediate_goal(x_init[:2].flatten(), 0.5, x_target[:2].flatten(), A,b).flatten()
    x_intergoal=np.zeros(10)
    x_intergoal[:2]=inter_goal
    x_intergoal[2]=2
    real_trajectory = {'x': [], 'y': [], 'z': []}


    while np.linalg.norm(x_next.flatten() - x_target) >= 0.1:
        A_obs, b_obs= convexify(x_init[:2].flatten(), 0.5, obstacle_list)
        u = mpc_control(quadrotor, 10, x_next, x_intergoal,A_obs,b_obs).reshape(-1,1)

        real_trajectory['x'].append(x_next[0])
        real_trajectory['y'].append(x_next[1])
        real_trajectory['z'].append(x_next[2])
        x_next = quadrotor.next_x(x_next, u)
        
        A,b=convexify(x_next[:2].flatten(),0.5,obstacle_list)
        inter_goal=get_intermediate_goal(x_next[:2].flatten(), 0.5,x_target[:2].flatten(), A,b).flatten()
        x_intergoal=np.zeros(10)
        x_intergoal[:2]=inter_goal
        x_intergoal[2]=2
        
        print(x_next[:3].flatten())

    visualization(real_trajectory, obstacle_list)
    