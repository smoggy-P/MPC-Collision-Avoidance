import numpy as np
from Quadrotor import Quadrotor_linear, Quadrotor
from MPC_controller import mpc_control,mpc_control_stable
from visualization import visualization
from convexification import obstacle_list
from convexification import get_intermediate_goal
from convexification import convexify

drone = [0,0,0.5]  #pos_x,pos_y,radius

obs1=[-5,2,2]   #pos_x,pos_y,radius
obs2=[-3,-6,2]  #pos_x,pos_y,radius
obs3=[7,-1.9,2] #pos_x,pos_y,radius

obstacle_list=[obs1,obs2,obs3]

goal = np.array([-6,-5,2]) #pos_x,pos_y,pos_z

if __name__ == "__main__":
    
    N=10

    quadrotor_linear = Quadrotor_linear()
    quadrotor = Quadrotor()
    quadrotor.reset()

    x_init = np.zeros(10)
    x_init[0]=drone[0]
    x_init[1]=drone[1]
    
    x_target = np.zeros(10)
    x_target[0] = goal[0]
    x_target[1] = goal[1]
    x_target[2] = goal[2]
    x_next = x_init
    
    A,b=convexify(x_init[:2].flatten(),0.5,obstacle_list)

    inter_goal=get_intermediate_goal(x_init[:2].flatten(), 0.5, x_target[:2].flatten(), A,b).flatten()
    x_intergoal=np.zeros(10)
    x_intergoal[:2]=inter_goal

    x_intergoal[2] = x_target[2]
    real_trajectory = {'x': [], 'y': [], 'z': []}


    i = 0
    while np.linalg.norm(x_intergoal[:3]-x_target[:3])>0.1 and i < 50:
        i += 1
        A_obs,b_obs=convexify(x_next[:2].flatten(),drone[2],obstacle_list)
        
        u = mpc_control(quadrotor_linear, N, x_next, x_intergoal,A_obs,b_obs).reshape(-1,1)

        real_trajectory['x'].append(x_next[0])
        real_trajectory['y'].append(x_next[1])
        real_trajectory['z'].append(x_next[2])

        x_next = quadrotor_linear.next_x(x_next, u)
        # quadrotor.step(u)
        # x_next = quadrotor_linear.from_nonlinear(quadrotor)
        print(u.flatten())
        A,b=convexify(x_next[:2].flatten(),drone[2],obstacle_list)
        inter_goal=get_intermediate_goal(x_next[:2].flatten(), drone[2],x_target[:2].flatten(), A,b).flatten()
        x_intergoal=np.zeros(10)
        x_intergoal[:2]=inter_goal
        x_intergoal[2]=x_target[2]
        
    A,b = convexify(x_next[:2].flatten(),drone[2],obstacle_list)


    # while np.linalg.norm(x_next.flatten() - x_target) >= 0.1:
        
    #     u = mpc_control_stable(quadrotor_linear, 10, x_next, x_intergoal,A,b).reshape(-1,1)

    #     real_trajectory['x'].append(x_next[0])
    #     real_trajectory['y'].append(x_next[1])
    #     real_trajectory['z'].append(x_next[2])
    #     x_next = quadrotor.next_x(x_next, u)
        
    #     print("o")


    visualization(real_trajectory, obstacle_list)
    