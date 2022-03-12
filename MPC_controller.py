import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from Quadrotor import Quadrotor_linear
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation

def mpc_control(quadrotor, N, x_init, x_target):
    weight_input = 0.2*np.eye(4)    # Weight on the input
    
    cost = 0.
    constraints = []
    
    # Create the optimization variables
    x = cp.Variable((10, N + 1)) # cp.Variable((dim_1, dim_2))
    u = cp.Variable((4, N))

    # For each stage in the MPC horizon
    Q = np.identity(10)
    for n in range(N):
        cost += (cp.quad_form((x[:,n+1]-x_target),Q)  + cp.quad_form(u[:,n], weight_input))
        constraints += [x[:,n+1] == quadrotor.A @ x[:,n] + quadrotor.B @ u[:,n]]

    # Implement the cost components and/or constraints that need to be added once, here
    constraints += [x[:,0] == x_init.flatten()]
    
    # Solves the problem
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(solver=cp.OSQP)

    # We return the MPC input
    return u[:, 0].value

quadrotor = Quadrotor_linear()

x_init = np.zeros(10)
x_target = np.zeros(10)
x_target[0] = 5
x_target[1] = 5
x_target[2] = 5
x_next = x_init
real_trajectory = {'x': [], 'y': [], 'z': []}


for i in range(50):
    u = mpc_control(quadrotor, 10, x_next, x_target).reshape(-1,1)

    real_trajectory['x'].append(x_next[0])
    real_trajectory['y'].append(x_next[1])
    real_trajectory['z'].append(x_next[2])
    x_next = quadrotor.next_x(x_next, u)

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
                            interval=1,
                            repeat=False,
                            blit=False)
plt.show()