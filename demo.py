import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
from matplotlib import animation
from Quadrotor import Quadrotor_linear, Quadrotor

from MPC_controller import mpc_control,mpc_control_stable
from visualization import data_for_cylinder_along_z
from convexification import get_intermediate_goal, convexify

drone = [0,0,0.2]  #pos_x,pos_y,radius

obs1=np.array([-3,1,1])   #pos_x,pos_y,radius
obs2=np.array([-2,-3,1])  #pos_x,pos_y,radius
obs3=np.array([0,2,1]) #pos_x,pos_y,radius
obs4=np.array([-5,-1.9,1]) #pos_x,pos_y,radius
obs5=np.array([0.5,-2,1]) #pos_x,pos_y,radius

obstacle_list=[obs1,obs2,obs3,obs4,obs5]#,obs1*2,obs2*2,obs3*2,obs4*2,obs5*2]

goal = np.array([-5,-6,0]) #pos_x,pos_y,pos_z

def animate(i):
    line.set_xdata(real_trajectory['x'][:i + 1])
    line.set_ydata(real_trajectory['y'][:i + 1])
    line.set_3d_properties(real_trajectory['z'][:i + 1])
    point.set_xdata(real_trajectory['x'][i])
    point.set_ydata(real_trajectory['y'][i])
    point.set_3d_properties(real_trajectory['z'][i])

if __name__ == "__main__":
    
    N = 10

    quadrotor_linear = Quadrotor_linear()
    #quadrotor = Quadrotor()
    #quadrotor.reset()

    x_init = np.zeros(10)
    x_init[0]=drone[0]
    x_init[1]=drone[1]
    
    x_target = np.zeros(10)
    x_target[0] = goal[0]
    x_target[1] = goal[1]
    x_target[2] = goal[2]
    x_next = x_init

    
    A,b = convexify(x_init[:2].flatten(),0.5,obstacle_list)

    inter_goal=get_intermediate_goal(x_init[:2].flatten(), 0, x_target[:2].flatten(), A,b).flatten()
    x_intergoal=np.zeros(10)
    x_intergoal[:2]=inter_goal
    x_intergoal[2] = x_target[2]

    real_trajectory = {'x': [], 'y': [], 'z': []}

    i = 0
    while np.linalg.norm(x_intergoal[:3].flatten()-x_target[:3]) > 0.1 and i<300:
        i += 1
        A_obs,b_obs=convexify(x_next[:2].flatten(),drone[2],obstacle_list)
        
        u = mpc_control(quadrotor_linear, N, x_next, x_intergoal,A_obs,b_obs)

        if u is None:
            print("no solution")
            break
        else:
            u = u.reshape(-1,1)

        real_trajectory['x'].append(x_next[0])
        real_trajectory['y'].append(x_next[1])
        real_trajectory['z'].append(x_next[2])

        #quadrotor.step(u)
        print(u.flatten())
        x_next=quadrotor_linear.next_x(x_next, u).flatten()
        #x_next = quadrotor_linear.from_nonlinear(quadrotor)
        #print(x_next.flatten())
        print("\n")

        A,b=convexify(x_next[:2].flatten(),drone[2],obstacle_list)
        inter_goal=get_intermediate_goal(x_next[:2].flatten(), 0,x_target[:2].flatten(), A,b).flatten()
        x_intergoal=np.zeros(10)
        x_intergoal[:2]=inter_goal
        x_intergoal[2]=x_target[2]
        
    A,b = convexify(x_next[:2].flatten(),drone[2],obstacle_list)
    print("***")

    while np.linalg.norm(x_next[:3].flatten() - x_target[:3]) >= 0.1 and i<400:
         i+=1
         
         #♦print(i)
         u = mpc_control_stable(quadrotor_linear, 10, x_next, x_intergoal,A,b).reshape(-1,1)

         real_trajectory['x'].append(x_next[0])
         real_trajectory['y'].append(x_next[1])
         real_trajectory['z'].append(x_next[2])
         
         #quadrotor.step(u)
         #x_next = quadrotor_linear.from_nonlinear(quadrotor)
         x_next=quadrotor_linear.next_x(x_next, u).flatten()
        
        #print(x_next[:3].flatten())
    """ Visualisation """
    fig = plt.figure()
    ax1 = p3.Axes3D(fig) # 3D place for drawing
    real_trajectory['x'] = np.array(real_trajectory['x'])
    real_trajectory['y'] = np.array(real_trajectory['y'])
    real_trajectory['z'] = np.array(real_trajectory['z'])
    point, = ax1.plot([real_trajectory['x'][0]], [real_trajectory['y'][0]], [real_trajectory['z'][0]], 'ro', ms=10, label='Quadrotor')
    line, = ax1.plot([real_trajectory['x'][0]], [real_trajectory['y'][0]], [real_trajectory['z'][0]], label='Real_Trajectory')


    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_xlim3d((-5, 5))
    ax1.set_ylim3d((-5, 5))
    ax1.set_zlim3d((0, 3))
    ax1.set_title('3D animate')
    ax1.view_init(30, 35)
    ax1.legend(loc='lower right')

    for obstacle in obstacle_list:
        Xc,Yc,Zc = data_for_cylinder_along_z(obstacle[0],obstacle[1],obstacle[2],5)
        ax1.plot_surface(Xc, Yc, Zc, alpha=0.5)

    

    ani = animation.FuncAnimation(fig=fig,
                                func=animate,
                                frames=len(real_trajectory['x']),
                                interval=5,
                                repeat=True,
                                blit=False)
    plt.show()
    