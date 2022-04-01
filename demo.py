import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
from matplotlib import animation
from Quadrotor import Quadrotor_linear, Quadrotor

from MPC_controller import mpc_control,mpc_control_stable,OTS,get_observer_gain,luenberger_observer
from visualization import data_for_cylinder_along_z
from convexification import get_intermediate_goal, convexify

drone = [0,0,0.05]  #pos_x,pos_y,radius

obs1=np.array([-3,1,1])/2   #pos_x,pos_y,radius
obs2=np.array([-2,-3,1]) /2 #pos_x,pos_y,radius
obs3=np.array([0,2,1])/2 #pos_x,pos_y,radius
obs4=np.array([-5,-1.9,1])/2 #pos_x,pos_y,radius
obs5=np.array([0.5,-2,1])/2 #pos_x,pos_y,radius

obstacle_list=[obs1,obs2,obs3,obs4,obs5]#,obs1*2,obs2*2,obs3*2,obs4*2,obs5*2]

goal = np.array([-3,-3,2]) #pos_x,pos_y,pos_z

sensor_noise_sigma=np.array([0.1,0.1,0.1,0.01,0.01,0.01,0.001,0.001,0.001,0.001])
real_disturbance=np.random.normal(loc=0,scale=0.1,size=3)
print("real _dist", real_disturbance)

Cd= np.array([[0, 0, 0 ],
              [0, 0, 0 ],
              [0, 0, 0 ],
              [0, 0, 0 ],
              [0, 0, 0 ],
              [0, 0, 0 ],
              [0, 0, 0 ],
              [0, 0, 0 ],
              [0, 0, 0 ],
              [0, 0, 0 ]])

Bd= np.array([[0, 0, 0],
              [0, 0, 0],
              [0, 0, 0],
              [1, 0, 0],
              [0, 1, 0],
              [0, 0, 1],
              [0, 0, 0],
              [0, 0, 0],
              [0, 0, 0],
              [0, 0, 0]])

obs_eigen_values= np.array([0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5])

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

    x_init = np.zeros(10)
    x_init[0]=drone[0]
    x_init[1]=drone[1]
    
    x_target = np.zeros(10)
    x_target[0] = goal[0]
    x_target[1] = goal[1]
    x_target[2] = goal[2]
    
    x_hat = x_init
    x_real = x_init
    d_hat=np.zeros((3,1))
    output = x_init
    
    L=get_observer_gain(quadrotor_linear, Bd,Cd,obs_eigen_values)
    A,b = convexify(x_init[:2].flatten(),0.5,obstacle_list)
    
    inter_goal=get_intermediate_goal(x_init[:2].flatten(), 0, x_target[:2].flatten(), A,b).flatten()
    x_intergoal=np.zeros(10)
    x_intergoal[:2]=inter_goal
    x_intergoal[2] = x_target[2]

    real_trajectory = {'x': [], 'y': [], 'z': []}
    est_trajectory = {'x': [], 'y': [], 'z': []}
    
    #print(x_intergoal)
   
    x_ref,u_ref = OTS(quadrotor_linear,x_intergoal,d_hat, A,b,Bd,Cd)
    #print(x_ref,u_ref)
    #print(x_ref.shape,u_ref.shape)
    i = 0
    while np.linalg.norm(x_intergoal[:3].flatten()-x_target[:3]) > 0.1 and i<300:
        i += 1
        A_obs,b_obs=convexify(x_hat[:2].flatten(),drone[2],obstacle_list)
        
        u = mpc_control(quadrotor_linear, N, x_hat, x_ref.flatten(),u_ref.flatten(),A_obs,b_obs)
        
        #print(u)

        if u is None:
            print("no solution")
            break
        else:
            u = u.reshape(-1,1)

        est_trajectory['x'].append(x_hat[0])
        est_trajectory['y'].append(x_hat[1])
        est_trajectory['z'].append(x_hat[2])
        
        real_trajectory['x'].append(x_real[0])
        real_trajectory['y'].append(x_real[1])
        real_trajectory['z'].append(x_real[2])
        
        #print(x_real)
        x_real=quadrotor_linear.disturbed_next_x(x_real,u,real_disturbance,Bd)
        
        
        ouput=quadrotor_linear.disturbed_output(x_real,real_disturbance, Cd, sensor_noise_sigma).flatten()
        
        x_hat,d_hat=luenberger_observer(quadrotor_linear, x_hat,d_hat,output,u,Bd,Cd,L)

        A_obs,b_obs=convexify(x_hat[:2].flatten(),drone[2],obstacle_list)
        #print(x_hat)
        inter_goal=get_intermediate_goal(x_hat[:2].flatten(), 0,x_target[:2].flatten(), A_obs,b_obs).flatten()
        x_intergoal=np.zeros(10)
        x_intergoal[:2]=inter_goal
        x_intergoal[2]=x_target[2]
        
        x_ref,u_ref = OTS(quadrotor_linear,x_intergoal,d_hat, A_obs,b_obs,Bd,Cd)
        print("d_hat:",d_hat)
        #print("ref:",x_ref,u_ref)
        
    A,b = convexify(x_hat[:2].flatten(),drone[2],obstacle_list)
    print("***")

    while np.linalg.norm(x_hat[:3].flatten() - x_target[:3]) >= 0.1 and i<400:
         i+=1
         
         #♦print(i)
         u = mpc_control_stable(quadrotor_linear, 10, x_hat, x_ref,u_ref,A,b).reshape(-1,1)

         if u is None:
             print("no solution")
             break
         else:
             u = u.reshape(-1,1)

         est_trajectory['x'].append(x_hat[0])
         est_trajectory['y'].append(x_hat[1])
         est_trajectory['z'].append(x_hat[2])
         
         real_trajectory['x'].append(x_real[0])
         real_trajectory['y'].append(x_real[1])
         real_trajectory['z'].append(x_real[2])
         
         x_real=quadrotor_linear.disturbed_next_x(x_real,u,real_disturbance,Bd)
         
         ouput=quadrotor_linear.disturbed_output(x_real,real_disturbance, Cd, sensor_noise_sigma).flatten()
         
         x_hat,d_hat=luenberger_observer(quadrotor_linear, x_hat,d_hat,output,u,Bd,Cd,L)
         
         x_ref,u_ref = OTS(quadrotor_linear,x_intergoal,d_hat, A,b,Bd,Cd)
        
        #print(x_hat[:3].flatten())
    """ Visualisation """
    fig = plt.figure()
    ax1 = p3.Axes3D(fig) # 3D place for drawing
    real_trajectory['x'] = np.array(real_trajectory['x'])
    real_trajectory['y'] = np.array(real_trajectory['y'])
    real_trajectory['z'] = np.array(real_trajectory['z'])
    point, = ax1.plot([real_trajectory['x'][0]], [real_trajectory['y'][0]], [real_trajectory['z'][0]], 'ro', ms=2.5, label='Quadrotor')
    line, = ax1.plot([real_trajectory['x'][0]], [real_trajectory['y'][0]], [real_trajectory['z'][0]], label='Real_Trajectory')


    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_xlim3d((-3, 3))
    ax1.set_ylim3d((-3, 3))
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
    