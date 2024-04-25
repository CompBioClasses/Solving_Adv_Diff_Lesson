
'''
This numerically estimates the 1D advection-diffusion problem with 
homogeneous dirichlet boundary conditions, constant diffution, and 
time-dependent velocity.

Author: Kimberlyn Eversman
Date: 04-25-24'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Seed the random number generator
np.random.seed(10)

###############################################################################
# Initialize parameters and numerical approximation information
###############################################################################

# Space bound
L = 30 
# Time bound 
T = 2 

# Number of space steps (must be a natural number)
I = 200
# Number of time steps (must be a natural number)
K = 300

# Space step
h = L / I
# Time step
tau = T / K

# Diffusion coefficient 
D = 0.5

# Velocity function 
v = 3*np.ones(K) # wind is constant
# v = 15*np.sin(20*np.arange(0,K)*tau) # wind is a sin function 
# v = np.random.normal(0.0, 7.0, K) # wind is drawn from a normal dist

# Initial condition function 
initial_concentration = 10 # Total host population
def u_0(x):
    return initial_concentration*np.exp(-(x-(L/2))**2)

###############################################################################
# Check stability conditions:
###############################################################################

if h >= (2*D)/np.max(np.absolute(v)):
    print("Stability conditions are not met: h = {:.5f} is not less than {:.5f}."\
          .format(h,(2*D)/np.max(np.absolute(v))))
elif tau >= (h**2)/(2*D):
    print("Stability conditions are not met: tau = {:.5f} is not less than {:.5f}."\
          .format(tau,(h**2)/(2*D)))
else: 
    print('Stability conditions are met!')

###############################################################################
# Implement the FDM
###############################################################################

# Create W where we will store each w^k_i:
    # each row is i (space) for i = 0, ..., I
    # each column is k (time) for k = 0, ..., K
    # So W[i,k] is w_i^k and W[:,k] is vector w^k
W = np.zeros((I+1,K+1))

# Calulate w^0_i for i = 0, ..., I using our inital value function 
W[:,0] = u_0(np.arange(0,I+1)*h)

# Create A
A = np.zeros((I+1, I+1))

# Generate the non-zero, time-independent entries of A
index_list = np.arange(1,I)
b = 1 - (2*D*tau)/(h**2)
A[index_list,index_list] = b

# Calculate the vector w^k for k = 1, ..., K
for k in range(K):
    # Generate the non-zero, time-dependent entries of A
    # Note we must generate A for every k because our v is changing at each 
    #   time step
    a = (D*tau)/(h**2) + (tau*v[k])/(2*h)
    c = (D*tau)/(h**2) - (tau*v[k])/(2*h)
    A[index_list,index_list-1] = a
    A[index_list-1,index_list] = c

    # Calculate w^{k+1}_i
    W[:,k+1] = np.matmul(A,W[:,k])


###############################################################################
# Plot w^0_i for i = 0, 1, ..., I.  The approximation of u(x,t) at the initial 
#   time.
###############################################################################

# k = 0

# # Generate x-axis values
# x_vals = np.arange(0,I+1)*h

# # Plot lines 
# plt.plot(x_vals, w[:,k], 'r')
# plt.xlabel('x') #x ~ i*h
# plt.ylabel('u(x,T)') #u*(x,T) ~ w^0_i
# plt.title('Num. Approx. of u(x,t) at time t={0:.3f}'.format(np.round(k*tau,3)))
# plt.show()


###############################################################################
# Plot w^K_i for i = 0, 1, ..., I. The approximation of u(x,t) at the final 
#   time, T.
###############################################################################

# k = K

# # Generate x-axis values
# x_vals = np.arange(0,I+1)*h

# # Plot lines 
# plt.plot(x_vals, w[:,k], 'r')
# plt.xlabel('x') #x ~ i*h
# plt.ylabel('u(x,T)') #u*(x,T) ~ w^0_i
# plt.title('Num. Approx. of u(x,t) at time t={0:.3f}'.format(np.round(k*tau,3)))
# plt.show()


###############################################################################
# Plot a video of w^k_i for k = 0, ..., K. The approximation of u(x,t) over 
#   time.
###############################################################################

# Generate x-axis values
x_vals = np.arange(0,I+1)*h

fig, ax = plt.subplots()
line, = plt.plot([], [], "r-") # start with an empty plot

plt.axis([0, L, 0, 1.1*initial_concentration]) 
plt.xlabel('$x$') #x ~ i*h
plt.ylabel('$u(x,t)$') #u*(x,t) ~ w^k_i

# this function will be called at every iteration of the animation
def update_graph(k, W, line):
    # plot w^k for i = 0, ..., I
    line.set_data(x_vals, W[:,k]) 

    # Update the title with the current time step 
    ax.set_title('Num. Approx. of $u(x,t)$ at time $t$={0:.3f}'.format(np.round(k*tau,3)),
                 fontsize=20) 

    return line,

# fargs is the arguments that are passed to each call of the function update_graph
line_ani = animation.FuncAnimation(fig, update_graph, frames=K,
                                   fargs=(W, line), interval=100)

plt.show()

# # Save the animation as an animated GIF
# from matplotlib.animation import PillowWriter
# line_ani.save("one_dim_adv_diff_animation.gif",
#          dpi=200, writer=PillowWriter(fps=10))

