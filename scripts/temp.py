import numpy as np
from numpy import linalg as la
from math import atan2, pi
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from environmentsCreation.FG_env_creator import readMarkrsLocationsFile


def main():
    dir = '/home/majd/catkin_ws/src/basic_rl_agent/data/FG_linux/FG_gatesPlacementFile'
    FG_gatesCenterLocationsDict, FG_markersLocationDict = readMarkrsLocationsFile(dir)

    markersLocation = FG_markersLocationDict['gate0B']['location']
    markersLocation = np.array(markersLocation) 

    v1 = markersLocation[1, :] - markersLocation[0, :] 
    v2 = markersLocation[2, :] - markersLocation[0, :] 
    # v1 and v2 have the same origin.
    v = np.cross(v1, v2)
    v = v/la.norm(v)
    # print(markersLocation.shape)
    print(v1.shape, v2.shape, v.shape)
    print(np.dot(v1, v))
    print(np.dot(v2, v))

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # ax.set_aspect('equal')
    ax.plot3D(markersLocation[:, 0], markersLocation[:, 1], markersLocation[:, 2], 'ro')

    ax.quiver(markersLocation[0, 0],markersLocation[0, 1],markersLocation[0, 2], v[0], v[1], v[2])
    ax.quiver(markersLocation[0, 0],markersLocation[0, 1],markersLocation[0, 2], v1[0], v1[1], v1[2])
    ax.quiver(markersLocation[0, 0],markersLocation[0, 1],markersLocation[0, 2], v2[0], v2[1], v2[2])

    ax.set_xlim([-8, 8])
    ax.set_ylim([-8, 8])
    ax.set_zlim([-8, 8])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_aspect('equal')

    plt.show()



if __name__ == "__main__":
    main()
