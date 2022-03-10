import gym
import env
import numpy as np
from controller import PDcontrolller
from trajectory import circle, diamond, hovel, tud
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation

def tud_sim():
    env = gym.make('Quadrotor-v0')
    current_state = env.reset(position=[0, 0, 0])
    dt = 0.01
    t = 0
    controller = PDcontrolller()
    plan_trajectory = {'x': [], 'y': [], 'z': []}
    real_trajectory = {'x': [], 'y': [], 'z': []}
    actions = []

    while(t < 23):
        desired_state = tud(t)
        control_var = controller.control(desired_state, current_state)
        action = control_var['cmd_motor_speeds']
        actions.append(action)
        obs, reward, done, info = env.step(action)
        real_trajectory['x'].append(obs['x'][0])
        real_trajectory['y'].append(obs['x'][1])
        real_trajectory['z'].append(obs['x'][2])
        plan_trajectory['x'].append(desired_state['x'][0])
        plan_trajectory['y'].append(desired_state['x'][1])
        plan_trajectory['z'].append(desired_state['x'][2])
        current_state = obs
        t += dt


    fig = plt.figure()
    ax1 = p3.Axes3D(fig) # 3D place for drawing
    real_trajectory['x'] = np.array(real_trajectory['x'])
    real_trajectory['y'] = np.array(real_trajectory['y'])
    real_trajectory['z'] = np.array(real_trajectory['z'])

    track_preformance = np.sum((plan_trajectory['x']-real_trajectory['x'])**2+(plan_trajectory['y']-real_trajectory['y'])**2+(plan_trajectory['z']-real_trajectory['z'])**2)*dt
    print("tracking performance:", track_preformance)
    print("total time:", 23)
    battery_storage = np.sum(np.array(actions)**2)*dt
    print("total battery storage:", battery_storage)



    real_trajectory['x'] = real_trajectory['x'][::10]
    real_trajectory['y'] = real_trajectory['y'][::10]
    real_trajectory['z'] = real_trajectory['z'][::10]
    point, = ax1.plot([real_trajectory['x'][0]], [real_trajectory['y'][0]], [real_trajectory['z'][0]], 'ro', label='Quadrotor')
    line, = ax1.plot([real_trajectory['x'][0]], [real_trajectory['y'][0]], [real_trajectory['z'][0]], label='Real_Trajectory')

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    plt.xlim((-1, 1))
    plt.ylim((-1, 10))
    ax1.set_zlim(0, 10)
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

def hovel_sim():
    env = gym.make('Quadrotor-v0')
    current_state = env.reset(position=[0, 0, 0])
    dt = 0.01
    t = 0
    controller = PDcontrolller()
    plan_trajectory = {'x': [], 'y': [], 'z': []}
    real_trajectory = {'x': [], 'y': [], 'z': []}
    actions = []

    while(t < 30):
        desired_state = hovel(t)
        control_var = controller.control(desired_state, current_state)
        action = control_var['cmd_motor_speeds']
        actions.append(action)
        obs, reward, done, info = env.step(action)
        real_trajectory['x'].append(obs['x'][0])
        real_trajectory['y'].append(obs['x'][1])
        real_trajectory['z'].append(obs['x'][2])
        plan_trajectory['x'].append(desired_state['x'][0])
        plan_trajectory['y'].append(desired_state['x'][1])
        plan_trajectory['z'].append(desired_state['x'][2])
        current_state = obs
        t += dt


    fig = plt.figure()
    ax1 = p3.Axes3D(fig) # 3D place for drawing
    real_trajectory['x'] = np.array(real_trajectory['x'])
    real_trajectory['y'] = np.array(real_trajectory['y'])
    real_trajectory['z'] = np.array(real_trajectory['z'])

    track_preformance = np.sum((plan_trajectory['x']-real_trajectory['x'])**2+(plan_trajectory['y']-real_trajectory['y'])**2+(plan_trajectory['z']-real_trajectory['z'])**2)*dt
    print("tracking performance:", track_preformance)
    print("total time:", 30)
    battery_storage = np.sum(np.array(actions)**2)*dt
    print("total battery storage:", battery_storage)



    real_trajectory['x'] = real_trajectory['x'][::10]
    real_trajectory['y'] = real_trajectory['y'][::10]
    real_trajectory['z'] = real_trajectory['z'][::10]
    point, = ax1.plot([real_trajectory['x'][0]], [real_trajectory['y'][0]], [real_trajectory['z'][0]], 'ro', label='Quadrotor')
    line, = ax1.plot([real_trajectory['x'][0]], [real_trajectory['y'][0]], [real_trajectory['z'][0]], label='Real_Trajectory')

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    plt.xlim((0, 200))
    plt.ylim((0, 200))
    ax1.set_zlim(0, 200)
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

def circle_sim():
    env = gym.make('Quadrotor-v0')
    current_state = env.reset(position=[5, 0, 0])
    dt = 0.01
    t = 0
    controller = PDcontrolller()
    plan_trajectory = {'x': [], 'y': [], 'z': []}
    real_trajectory = {'x': [], 'y': [], 'z': []}
    actions = []

    while(t < 10):
        desired_state = circle(t)
        control_var = controller.control(desired_state, current_state)
        action = control_var['cmd_motor_speeds']
        actions.append(action)
        obs, reward, done, info = env.step(action)
        real_trajectory['x'].append(obs['x'][0])
        real_trajectory['y'].append(obs['x'][1])
        real_trajectory['z'].append(obs['x'][2])
        plan_trajectory['x'].append(desired_state['x'][0])
        plan_trajectory['y'].append(desired_state['x'][1])
        plan_trajectory['z'].append(desired_state['x'][2])
        current_state = obs
        t += dt


    fig = plt.figure()
    ax1 = p3.Axes3D(fig) # 3D place for drawing
    real_trajectory['x'] = np.array(real_trajectory['x'])
    real_trajectory['y'] = np.array(real_trajectory['y'])
    real_trajectory['z'] = np.array(real_trajectory['z'])

    track_preformance = np.sum((plan_trajectory['x']-real_trajectory['x'])**2+(plan_trajectory['y']-real_trajectory['y'])**2+(plan_trajectory['z']-real_trajectory['z'])**2)*dt
    print("tracking performance:", track_preformance)
    print("total time:", 10)
    battery_storage = np.sum(np.array(actions)**2)*dt
    print("total battery storage:", battery_storage)



    real_trajectory['x'] = real_trajectory['x'][::10]
    real_trajectory['y'] = real_trajectory['y'][::10]
    real_trajectory['z'] = real_trajectory['z'][::10]
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

def diamond_sim():
    env = gym.make('Quadrotor-v0')
    current_state = env.reset(position=[0, 0, 0])
    dt = 0.01
    t = 0
    controller = PDcontrolller()
    plan_trajectory = {'x': [], 'y': [], 'z': []}
    real_trajectory = {'x': [], 'y': [], 'z': []}
    actions = []

    while(t < 8):
        desired_state = diamond(t)
        control_var = controller.control(desired_state, current_state)
        action = control_var['cmd_motor_speeds']
        actions.append(action)
        obs, reward, done, info = env.step(action)
        real_trajectory['x'].append(obs['x'][0])
        real_trajectory['y'].append(obs['x'][1])
        real_trajectory['z'].append(obs['x'][2])
        plan_trajectory['x'].append(desired_state['x'][0])
        plan_trajectory['y'].append(desired_state['x'][1])
        plan_trajectory['z'].append(desired_state['x'][2])
        current_state = obs
        t += dt


    fig = plt.figure()
    ax1 = p3.Axes3D(fig) # 3D place for drawing
    real_trajectory['x'] = np.array(real_trajectory['x'])
    real_trajectory['y'] = np.array(real_trajectory['y'])
    real_trajectory['z'] = np.array(real_trajectory['z'])

    track_preformance = np.sum((plan_trajectory['x']-real_trajectory['x'])**2+(plan_trajectory['y']-real_trajectory['y'])**2+(plan_trajectory['z']-real_trajectory['z'])**2)*dt
    print("tracking performance:", track_preformance)
    print("total time:", 8)
    battery_storage = np.sum(np.array(actions)**2)*dt
    print("total battery storage:", battery_storage)



    real_trajectory['x'] = real_trajectory['x'][::10]
    real_trajectory['y'] = real_trajectory['y'][::10]
    real_trajectory['z'] = real_trajectory['z'][::10]
    point, = ax1.plot([real_trajectory['x'][0]], [real_trajectory['y'][0]], [real_trajectory['z'][0]], 'ro', label='Quadrotor')
    line, = ax1.plot([real_trajectory['x'][0]], [real_trajectory['y'][0]], [real_trajectory['z'][0]], label='Real_Trajectory')

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    plt.xlim((0, 1))
    plt.ylim((-2, 2))
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
