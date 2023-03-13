import numpy as np
import scipy as sp


def generate_streamline(initial_x, initial_y, velocity_field, n_points=100, dt=0.1):
    """
    Arguments:
    - initial_x : Initial x position of the streamline 
    - initial_y : Initial y position of the streamline
    - velocity_field : complete velocityfield in both x and y directions (u,v) (shape=[u][v])
    - n_points : number of points to generate per streamline, default 100
    - dt : Time step size, default 0.1

    Returns:
    - Array of X and Y positions of the streamline

    """

    #n_streamlines = len(initial_x)
    
    # Position for each streamline point
    X = np.zeros((n_points, 1))
    Y = np.zeros((n_points, 1))

    # Velocity for each streamline point
    U = np.zeros((n_points, 1))
    V = np.zeros((n_points, 1))

    # Set initial position and velocity
    X[0] = initial_x
    Y[0] = initial_y

    U[0] = velocity_field[X[0], Y[0]][0]
    V[0] = velocity_field[X[0], Y[0]][1]

    for i in range(1, n_points):
        X[i] = X[i-1] + U[i-1]*dt
        Y[i] = Y[i-1] + V[i-1]*dt

        U[i], V[i] = velocity_field[X[0], Y[0]]
    

    return X, Y

    



    


    


    