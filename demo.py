import numpy as np
from Quadrotor import Quadrotor_linear, Quadrotor
from MPC_controller import mpc_control
from visualization import visualization
from convexification import obstacle_list, get_intermediate_goal, convexify
if __name__ == "__main__":
    quadrotor_linear = Quadrotor_linear()
    quadrotor = Quadrotor()
    quadrotor.reset()

    x_init = np.zeros(10)
    x_target = np.zeros(10)
    x_target[0:3] = np.array([-6, -5, 0])
    x_next = x_init
    A,b = convexify(x_init[:2].flatten(),0.5,obstacle_list)
    inter_goal=get_intermediate_goal(x_init[:2].flatten(), 0.5, x_target[:2].flatten(), A,b).flatten()
    x_intergoal=np.zeros(10)
    x_intergoal[:2]=inter_goal
    x_intergoal[2] = x_target[2]
    real_trajectory = {'x': [], 'y': [], 'z': []}

    i = 0
    while np.linalg.norm(x_next.flatten() - x_target) >= 0.1 and i < 100:
        
        A_obs,b_obs=convexify(x_next[:2].flatten(),0.5,obstacle_list)
        
        u = mpc_control(quadrotor_linear, 1, x_next, x_intergoal,A_obs,b_obs).reshape(-1,1)

        real_trajectory['x'].append(x_next[0])
        real_trajectory['y'].append(x_next[1])
        real_trajectory['z'].append(x_next[2])

        quadrotor.step(u)

        x_next = quadrotor_linear.next_x(quadrotor)
        
        A,b = convexify(x_next[:2].flatten(),0.5,obstacle_list)
        inter_goal=get_intermediate_goal(x_next[:2].flatten(), 0.5,x_target[:2].flatten(), A,b).flatten()
        x_intergoal=np.zeros(10)
        x_intergoal[:2]=inter_goal
        x_intergoal[2] = x_target[2]
        i+=1
        print(x_next[:3].flatten())

    visualization(real_trajectory, obstacle_list)
    