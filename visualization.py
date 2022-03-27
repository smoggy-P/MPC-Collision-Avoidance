import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from numpy import size
from convexification import obstacle_list

def data_for_cylinder_along_z(center_x,center_y,radius,height_z):
    z = np.linspace(0, height_z, 2)
    theta = np.linspace(0, 2*np.pi, 20)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = radius*np.cos(theta_grid) + center_x
    y_grid = radius*np.sin(theta_grid) + center_y
    return x_grid,y_grid,z_grid

def visualization(real_trajectory, obstacle_list):
    fig = plt.figure()
    ax1 = p3.Axes3D(fig) # 3D place for drawing
    real_trajectory['x'] = np.array(real_trajectory['x'],dtype=object)
    real_trajectory['y'] = np.array(real_trajectory['y'],dtype=object)
    real_trajectory['z'] = np.array(real_trajectory['z'],dtype=object)
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

    def animate(i):
        line.set_xdata(real_trajectory['x'][:i + 1])
        line.set_ydata(real_trajectory['y'][:i + 1])
        line.set_3d_properties(real_trajectory['z'][:i + 1])
        point.set_xdata(real_trajectory['x'][i])
        point.set_ydata(real_trajectory['y'][i])
        point.set_3d_properties(real_trajectory['z'][i])

    ani = animation.FuncAnimation(fig=fig,
                                func=animate,
                                frames=len(real_trajectory['x']),
                                interval=5,
                                repeat=True,
                                blit=False)
    plt.show()