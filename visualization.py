import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


def visualization(real_trajectory):
    fig = plt.figure()
    ax1 = p3.Axes3D(fig) # 3D place for drawing
    real_trajectory['x'] = np.array(real_trajectory['x'])
    real_trajectory['y'] = np.array(real_trajectory['y'])
    real_trajectory['z'] = np.array(real_trajectory['z'])
    point, = ax1.plot([real_trajectory['x'][0]], [real_trajectory['y'][0]], [real_trajectory['z'][0]], 'ro', label='Quadrotor')
    line, = ax1.plot([real_trajectory['x'][0]], [real_trajectory['y'][0]], [real_trajectory['z'][0]], label='Real_Trajectory')

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    ax1.set_zlim(0, 3)
    ax1.set_title('3D animate')
    ax1.view_init(30, 35)
    ax1.legend(loc='lower right')

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
                                interval=100,
                                repeat=False,
                                blit=False)
    plt.show()