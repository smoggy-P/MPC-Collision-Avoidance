from math import cos, pi, sin
import numpy as np
from math import sqrt
def tj_from_line(start_pos, end_pos, time_ttl, t_c):
    v_max = (end_pos-start_pos)*2/time_ttl; 
    if t_c >= 0 and t_c < time_ttl/2:
        vel = v_max*t_c/(time_ttl/2)
        pos = start_pos + t_c*vel/2
        acc = np.array([0,0,0])
        return np.array(pos), np.array(vel),acc
    else:
        vel = v_max*(time_ttl-t_c)/(time_ttl/2)
        pos = end_pos - (time_ttl-t_c)*vel/2
        acc = np.array([0,0,0])
        return np.array(pos), np.array(vel),acc

def hovel(t):
    pos, vel, acc = tj_from_line(np.array([0,0,0]), np.array([200,200,200]), 30, t)
    yaw = 0
    yawdot = 0
    desired_state = {'x':[],'x_dot':[],'x_ddot':[],'yaw':[],'yaw_dot':[]}
    desired_state['x'] = pos
    desired_state['x_dot'] = vel
    desired_state['x_ddot'] = acc
    desired_state['yaw'] = yaw
    desired_state['yaw_dot'] = yawdot
    return desired_state

def circle(t):

    T = 10
    radius = 5
    dt = 0.0001
    def pos_from_angle(a):
        pos = np.array([radius*cos(a), radius*sin(a), 2.5*a/(2*pi)])
        return pos

    def get_vel(t):
        angle1,_,_ = tj_from_line(0, 2*pi, T, t)
        pos1 = pos_from_angle(angle1)
        angle2,_,_ = tj_from_line(0, 2*pi, T, t+dt)
        vel = (pos_from_angle(angle2) - pos1)/dt
        return vel

    if t > T:
        pos = np.array([radius, 0, 2.5])
        vel = np.array([0,0,0])
        acc = np.array([0,0,0])
    else:
        angle,_,_ = tj_from_line(0, 2*pi, T, t)
        pos = pos_from_angle(angle)
        vel = get_vel(t)
        acc = (get_vel(t+dt) - get_vel(t))/dt
    yaw = 0
    yawdot = 0
    desired_state = {'x':[],'x_dot':[],'x_ddot':[],'yaw':[],'yaw_dot':[]}
    desired_state['x'] = pos
    desired_state['x_dot'] = vel
    desired_state['x_ddot'] = acc
    desired_state['yaw'] = yaw
    desired_state['yaw_dot'] = yawdot
    return desired_state

def diamond(t):
    if t<=2:
        [pos, vel, acc] = tj_from_line(np.array([0,0,0]), np.array([0,sqrt(2),sqrt(2)]), 2, t)
    elif t<=4:
        [pos, vel, acc] = tj_from_line(np.array([0, sqrt(2), sqrt(2)]), np.array([0, 0, 2*sqrt(2)]), 2, t-2)
    elif t<=6:
        [pos, vel, acc] = tj_from_line(np.array([0 ,0 ,2*sqrt(2)]), np.array([0 ,-sqrt(2) ,sqrt(2)]), 2, t-4)
    elif t<=8:
        [pos, vel, acc] = tj_from_line(np.array([0 ,-sqrt(2) ,sqrt(2)]), np.array([1 ,0 ,0]), 2, t-6)
    else:
        pos = np.array([1 ,0 ,0])
        vel = np.array([0, 0 ,0])
        acc = np.array([0, 0 ,0])
    yaw = 0
    yawdot = 0
    desired_state = {'x':[],'x_dot':[],'x_ddot':[],'yaw':[],'yaw_dot':[]}
    desired_state['x'] = pos
    desired_state['x_dot'] = vel
    desired_state['x_ddot'] = acc
    desired_state['yaw'] = yaw
    desired_state['yaw_dot'] = yawdot
    return desired_state

def tud(t):
    if t<=2:
        pos, vel, acc = tj_from_line(np.array([0,0,0]), np.array([0,0,7]), 2, t)
    elif t<=4:
        pos, vel, acc = tj_from_line(np.array([0,0,7]), np.array([0,-2,7]), 2, t-2)
    elif t<=7:
        pos, vel, acc = tj_from_line(np.array([0,-2,7]), np.array([0,3,7]), 3, t-4)
    elif t<=9:
        pos, vel, acc = tj_from_line(np.array([0,3,7]), np.array([0,3,2]), 2, t-7)
    elif t<=13:
        ang = tj_from_line(0, pi, 4, t-9)
        dt = 0.0001
        pos = np.array([0,5-2*cos(ang[0]),2-2*sin(ang[0])])
        angle1 = tj_from_line(0, pi, 4, t-9)          
        pos1 = np.array([0,5-2*cos(angle1[0]),2-2*sin(angle1[0])])
        angle2 = tj_from_line(0, pi, 4, t-9+dt)
        vel = (np.array([0,5-2*cos(angle2[0]),2-2*sin(angle2[0])]) - pos1)/dt
        acc = np.array([0,0,0])
    elif t<=15:
        pos, vel, acc = tj_from_line(np.array([0,7,2]), np.array([0,7,7]), 2, t-13)
    elif t<=16:
        pos, vel, acc = tj_from_line(np.array([0,7,7]), np.array([0,8,7]), 1, t-15)
    elif t<=21:
        ang = tj_from_line(0, pi, 5, t-16)
        dt = 0.0001
        pos = np.array([0, 8+3.5*sin(ang[0]), 3.5+3.5*cos(ang[0])])

        angle1 = tj_from_line(0, pi, 5, t-16)           
        pos1 = np.array([0, 8+3.5*sin(angle1[0]),3.5+3.5*cos(angle1[0])])
        angle2 = tj_from_line(0, pi, 5, t-16+dt)
        vel = (np.array([0, 8+3.5*sin(angle2[0]),3.5+3.5*cos(angle2[0])]) - pos1)/dt

        acc = np.array([0,0,0])
    elif t<=23:
        pos, vel, acc = tj_from_line(np.array([0,8,0]), np.array([0,8,7]), 2, t-21)
    yaw = 0
    yawdot = 0
    desired_state = {'x':[],'x_dot':[],'x_ddot':[],'yaw':[],'yaw_dot':[]}
    desired_state['x'] = pos
    desired_state['x_dot'] = vel
    desired_state['x_ddot'] = acc
    desired_state['yaw'] = yaw
    desired_state['yaw_dot'] = yawdot
    return desired_state


