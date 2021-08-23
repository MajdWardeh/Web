# from mpl_toolkits import mplot3d
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import numpy as np
from numpy import linalg as la, pi
from scipy.special import binom
from sympy import Symbol, Pow, diff, simplify, integrate, lambdify, expand
import cvxpy as cp
# import cv2
from store_read_data import Data_Reader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
import pandas as pd
import cv2
from Bezier_untils import bezier4thOrder

# workingDirectory = "~/drone_racing_ws/catkin_ddr/src/basic_rl_agent/data/dataset"
workingDirectory = '/home/majd/catkin_ws/src/basic_rl_agent/data/debugging_data2/dataset_202108052105_53/data' # provide the data subfolder in the dataset root directory.

def Bk_n(k, n, t):
    return binom(n, k)*Pow(1-t, n-k)*Pow(t, k)

def Plot(p, t, acc=100): 
# plot a Bezier curve and its derivative from 0 to 1 with 'acc' accuracy
    dp = diff(p, t)
    t_space = np.linspace(0, 1, acc)
    p_eval = [p.subs(t, ti) for ti in t_space]
    dp_eval = [dp.subs(t, ti) for ti in t_space]
    if Plotting:
        plt.figure()
        plt.plot(t_space, p_eval, 'r')
        # plt.plot(t_space, dp_eval, 'b')
        plt.show(block=False)

def solve_opt_problem(A, b, t, n):
    x = cp.Variable(n)
    objective = cp.Minimize(cp.sum_squares(A@x - b))
    prob = cp.Problem(objective)
    result = prob.solve(warm_start=True)
    return x.value

def processVelocityData(file_name):
    vel_df = pd.read_pickle('{}.pkl'.format(file_name))
    # vel_list = df['vel'].tolist()
    return vel_df

def processDatasetTxtHeader(txt_file):
    print('processing {}'.format(txt_file))
    txt_file = txt_file.split('.txt')[0]
    try:
        dataReader = Data_Reader(txt_file)
    except:
        print('{} is not a valid dataset file. skipped'.format(txt_file))
        return 
    try:
        vel_df = processVelocityData(txt_file)
    except:
        print('{} does not have twist data. skipped'.format(txt_file))
        return

    indices, images, Px, Py, Pz, Yaw = dataReader.getSamples()

    numOfSamples = dataReader.getNumOfSamples()
    dt = dataReader.getDt()
    sample_length = dataReader.sample_length
    print(dt, sample_length, numOfSamples)

    imageList = [np.array2string(image_np[0, 0])[1:-1] for image_np in images]

    t = np.linspace(0, sample_length*dt, sample_length)
    t = t[:, np.newaxis].astype(np.float64)
    A = np.concatenate((t, np.power(t, 2), np.power(t, 3), np.power(t, 4)), axis=1)
    BernsteinToMonomial = np.array([[1, -4, 6, -4, 1], [0, 4, -12, 12, -4], [0, 0, 6, -12, 6], [0, 0, 0, 4, -4], [0, 0, 0, 0, 1]], dtype=np.float64)
    monomialToBernstein = la.inv(BernsteinToMonomial)
    A_yaw = A[:, 0:2] 
    monomialToBernstein_yaw = np.array([[1, 1, 1], [0, 0.5, 1], [0, 0, 1]], dtype=np.float64)
    n = 5

    numOfSubplots = 4
    fig = plt.figure(figsize=plt.figaspect(1/float(numOfSubplots)))
    axes = [0] * 2
    axes[0] = fig.add_subplot(1, numOfSubplots, 1, projection='3d')
    axes[1] = fig.add_subplot(1, numOfSubplots, 2, projection='3d')
    yawAx = fig.add_subplot(1, numOfSubplots, 3)
    ax = fig.add_subplot(1, numOfSubplots, 4)
    plt.ion()
    plt.show()

    for index in indices:
        print(index)

        P = np.zeros((sample_length, 3))
        P[:, 0] = np.array(Px[index]) - Px[index][0]
        P[:, 1] = np.array(Py[index]) - Py[index][0]
        P[:, 2] = np.array(Pz[index]) - Pz[index][0]
        currYaw = Yaw[index][0] * -1
        rotationMatrix = Rotation.from_euler('z', currYaw).as_dcm()
        P_rotated =  np.matmul(rotationMatrix, P.T).T

        Pyaw = np.array(Yaw[index]) - Yaw[index][0]

        # compute the control points of the corresponding Bezier Curves:
        C_list = []
        for pi in P_rotated.T:
            b = pi.astype(np.float64)
            monomial_coeff = solve_opt_problem(A, b, t, n-1) # n-1 we reduced the opt variables since a0=0.
            monomial_coeff = np.insert(monomial_coeff, 0, 0)
            C = np.matmul(monomial_coeff, monomialToBernstein)
            C = C.astype(np.float32)
            C_list.append(C)
            # print("the Bernstein coefficients: ", C)
        cp = np.array(list(zip(C_list[0], C_list[1], C_list[2]))).T

        # reconstructing the trajectories from the calculated contorl points        
        P_reconstructed = []
        for ti in t:
            P = bezier4thOrder(cp, ti) 
            P_reconstructed.append(P)
        P_reconstructed = np.array(P_reconstructed)

        # plotting the resutls
        plot3D(axes, P_rotated, P_reconstructed)
        plotYaw(yawAx, Pyaw, t)

        imageName = imageList[index]
        image = cv2.imread(imageName)
        ax.imshow(image)

        plt.draw()
        plt.pause(0.01)

def plot3D(axes, P_org, P_recon):
    for ax in axes:
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
        ax.plot3D(P_org[:, 0], P_org[:, 1], P_org[:, 2], 'b')
        ax.plot3D(P_recon[:, 0], P_recon[:, 1], P_recon[:, 2], 'r')
    axes[0].elev = 0
    axes[1].elev = 90

def plotYaw(ax, Pyaw, t):
    ax.clear()
    axLim = pi/20
    ax.set_xlim(0, np.max(t))
    ax.set_ylim(-axLim, axLim)
    y = np.zeros_like(Pyaw)
    ax.plot(t, Pyaw)



def __lookForFiles2():
    overwrite = True
    txtFilesList = [file for file in os.listdir(workingDirectory) if file.endswith('.txt')]
    pklFilesList = [file.split('_preprocessed.pkl')[0] for file in os.listdir(workingDirectory) if file.endswith('_preprocessed.pkl')]
    for txtFile in txtFilesList[1:2]:
        if overwrite or not txtFile.split('.txt')[0] in pklFilesList:
            processDatasetTxtHeader(os.path.join(workingDirectory, txtFile) )

def main():
    __lookForFiles2()




if __name__ == '__main__':
    main()