import numpy as np
from numpy import linalg as la
from math import atan2, pi
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from .FG_env_creator import readMarkrsLocationsFile

def computeGateNormalVector(markersLocation):
    '''
        computes the normal vector of a gate given its marker locations.
        The normal vector is the normalized cross product from v1 to v2 where vi is the vector that has marker #0 as its
        origin and marker #i as its tip (head).
    '''
    # v1 and v2 have the same origin.
    v1 = markersLocation[1, :] - markersLocation[0, :] 
    v2 = markersLocation[2, :] - markersLocation[0, :] 
    v = np.cross(v1, v2)
    gateNormalVector = v/la.norm(v)

    gateCOM = np.mean(markersLocation, axis=0)

    return gateNormalVector, gateCOM

def main():
    dir = '/home/majd/catkin_ws/src/basic_rl_agent/data/FG_linux/FG_gatesPlacementFile'
    FG_markersLocationDict = readMarkrsLocationsFile(dir)

    markersLocation = FG_markersLocationDict['gate0B']

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

    gateCOM = np.mean(markersLocation, axis=0)

    ax.quiver(markersLocation[0, 0],markersLocation[0, 1],markersLocation[0, 2], v[0], v[1], v[2])
    ax.quiver(markersLocation[0, 0],markersLocation[0, 1],markersLocation[0, 2], v1[0], v1[1], v1[2])
    ax.quiver(markersLocation[0, 0],markersLocation[0, 1],markersLocation[0, 2], v2[0], v2[1], v2[2])

    ax.quiver(gateCOM[0], gateCOM[1], gateCOM[2], v[0], v[1], v[2])

    ax.set_xlim([-8, 8])
    ax.set_ylim([-8, 8])
    ax.set_zlim([0, 8])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_aspect('equal')

    plt.show()



if __name__ == "__main__":
    main()
