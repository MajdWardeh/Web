from operator import pos
import numpy as np
from math import pow, pi
from scipy.special import binom 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def bezier4thOrder(cp, t):
    # t must be in the range [0, 1]
    coeff_arr = np.array([1, 4, 6, 4, 1], dtype=np.float64)
    assert cp.shape[1] == 5, 'assertion failed, the provided control points (cp) are not for a 4th order Bezier.'
    P = np.zeros_like(cp, dtype=np.float64)
    for k in range(5):
        P[:, k] = cp[:, k] * coeff_arr[k] * pow(1-t, 4-k) * pow(t, k)
    return np.sum(P, axis=1)

def bezier3edOrder(cp, t):
    # t must be in the range [0, 1]
    coeff_arr = np.array([1, 3, 3, 1], dtype=np.float64)
    assert cp.shape[1] == 4, 'assertion failed, the provided control points (cp) are not for a 4th order Bezier.'
    P = np.zeros_like(cp, dtype=np.float64)
    for k in range(4):
        P[:, k] = cp[:, k] * coeff_arr[k] * pow(1-t, 3-k) * pow(t, k)
    return np.sum(P, axis=1)

def bezier2ndOrder(cp, t):
    # t must be in the range [0, 1]
    coeff_arr = np.array([1, 2, 1], dtype=np.float64)
    assert cp.shape[1] == 3, 'assertion failed, the provided control points (cp) are not for a 2nd order Bezier.'
    P = np.zeros_like(cp, dtype=np.float64)
    for k in range(3):
        P[:, k] = cp[:, k] * coeff_arr[k] * pow(1-t, 2-k) * pow(t, k)
    return np.sum(P, axis=1)
 
def bezier1stOrder(cp, t):
    # t must be in the range [0, 1]
    coeff_arr = np.array([1, 1], dtype=np.float64)
    assert cp.shape[1] == 2, 'assertion failed, the provided control points (cp) are not for a 2nd order Bezier.'
    P = np.zeros_like(cp, dtype=np.float64)
    for k in range(2):
        P[:, k] = cp[:, k] * coeff_arr[k] * pow(1-t, 1-k) * pow(t, k)
    return np.sum(P, axis=1)



class BezierVisulizer():
    def __init__(self, plot_delay=0.5) -> None:
        self.plot_delay = plot_delay
        numOfSubplots = 4
        fig = plt.figure(figsize=plt.figaspect(1/float(numOfSubplots)))
        self.axes = [0] * 2
        self.axes[0] = fig.add_subplot(1, numOfSubplots, 1, projection='3d')
        self.axes[1] = fig.add_subplot(1, numOfSubplots, 2, projection='3d')
        self.yawAx = fig.add_subplot(1, numOfSubplots, 3)
        self.imageAx = fig.add_subplot(1, numOfSubplots, 4)
        plt.ion()
        plt.show()

        acc = 100
        self.t_space = np.linspace(0, 1, acc)

    def __processPositionContorlPoints(self, cp):
        assert cp.shape==(3, 5), 'cp is not well shaped'
        Ps = []
        for ti in self.t_space:
            P = bezier4thOrder(cp, ti) 
            Ps.append(P)
        Ps = np.array(Ps)
        return Ps

    def __processYawControlPoints(self, cp):
        assert cp.shape==(1, 3), 'cp is not well shaped'
        Ps = []
        for ti in self.t_space:
            P = bezier2ndOrder(cp, ti) 
            Ps.append(P)
        Ps = np.array(Ps)
        return Ps

    def plotBezier(self, image, positionCP, yawCP, positionCP_hat=None, yawCP_hat=None):
        P_position = self.__processPositionContorlPoints(positionCP)
        P_yaw = self.__processYawControlPoints(yawCP)

        P_position_hat = None
        if not positionCP_hat is None:
            P_position_hat = self.__processPositionContorlPoints(positionCP_hat)

        P_yaw_hat = None
        if not yawCP_hat is None:
            P_yaw_hat = self.__processYawControlPoints(yawCP_hat)

        self.__plotPosition3D(P_position, P_position_hat)
        self.__plotYaw(P_yaw, P_yaw_hat)

        self.imageAx.imshow(image)
        plt.draw()
        plt.pause(self.plot_delay)


    def __plotPosition3D(self, P, P_hat):
        for ax in self.axes:
            ax.clear()
            axLim = 2
            ax.set_xlim(-axLim, axLim)
            ax.set_ylim(-axLim, axLim)
            ax.set_zlim(-axLim, axLim)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.azim = 180
            ax.dist = 8
            #ax.elev = 45
            ax.plot3D(P[:, 0], P[:, 1], P[:, 2], 'b')
            if not P_hat is None:
                ax.plot3D(P_hat[:, 0], P_hat[:, 1], P_hat[:, 2], 'r')

        self.axes[0].elev = 0
        self.axes[1].elev = 90

    def __plotYaw(self, Pyaw, Pyaw_hat):
        self.yawAx.clear()
        axLim = pi/20
        self.yawAx.set_xlim(0, np.max(self.t_space))
        self.yawAx.set_ylim(-axLim, axLim)
        self.yawAx.plot(self.t_space, Pyaw, 'b')
        if not Pyaw_hat is None:
            self.yawAx.plot(self.t_space, Pyaw_hat, 'r')


