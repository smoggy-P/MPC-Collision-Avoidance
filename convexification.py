import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

p=np.array([0,-2])
r_drone=0.5

goal=np.array([-10,-10])
obs1=[-5,2,2]
p_obs1=np.array([-5,2])
r_obs1=2

obs2=[-3,-6,2]
p_obs2=np.array([-3,-6])
r_obs2=3

obs3=[4,-3,2]
obstacle_list=[obs1,obs2,obs3]
#plt.plot(p)
def get_coeff(p,r_drone,p_obs,r_obs):
    
    point=p_obs+(r_drone+r_obs)*(p-p_obs)/np.linalg.norm(p-p_obs)
    a=-(p-p_obs)[0]/(p-p_obs)[1]
    b=point[1]-a*point[0]
    A=np.array([-a,1])
    if np.dot(A,p)>b:
        A=-A
        b=-b
    return A,b

def convexify(p,r_drone,obstacle_list):
    A=[]
    b=[]
    for obstacle in obstacle_list:
        p_obs=obstacle[:2]
        r_obs=obstacle[2]
        A_obs,b_obs=get_coeff(p,r_drone,p_obs,r_obs)
        A.append(A_obs)
        b.append(b_obs)
    return np.array(A),np.array(b)

def plot_convex_zone(p,r_drone,pos_goal,obstacle_list):
    A,b=convexify(p,r_drone,obstacle_list)
    #print(b)
    intermediate_goal=get_intermediate_goal(p, pos_goal, A,b)
    print(intermediate_goal)
    fig, ax  = plt.subplots()
    
    drone    = plt.Circle(p, r_drone, color='r')
    goal    = plt.Circle(pos_goal, r_drone/2, color='g')
    inter_goal    = plt.Circle(intermediate_goal, r_drone, color='y')
    ax.add_patch(drone)
    ax.add_patch(goal)
    ax.add_patch(inter_goal)
    x=np.array([-20,20])
    
    for i in range(len(obstacle_list)):
        Ai=A[i,:]
        bi=b[i]
        yi=(-Ai[0]*x+bi)/Ai[1]
        obstacle_i=plt.Circle(obstacle_list[i][:2], obstacle_list[i][2], color='blue')
        ax.add_patch(obstacle_i)
        ax.plot(x,yi)
    ax.set_xlim(-20,20)
    ax.set_ylim(-20,20)
    plt.show()
    return A,b
    
def get_intermediate_goal(pos, goal_pos, A,b):
    cost = 0.
    constraints = []
    
    # Create the optimization variables
    x = cp.Variable((2, 1)) # cp.Variable((dim_1, dim_2))
    # Add constraints
    constraints+=[A[:,:]@x<=b[:,np.newaxis]]

    #Cost is distance to goal
    cost+=cp.sum_squares(x-goal_pos[:,np.newaxis])

    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve()
    return x.value


A,b=plot_convex_zone(p,r_drone,goal,obstacle_list)