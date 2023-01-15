# from mpl_toolkits import mplot3d
from genericpath import isdir
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import time
import signal
import numpy as np
from numpy import linalg as la
from scipy.spatial.transform import Rotation
from scipy.special import binom
from sympy import Symbol, Pow, diff, simplify, integrate, lambdify, expand
import cvxpy as cp
# import cv2
from store_read_data import Data_Reader
import matplotlib.pyplot as plt
import pandas as pd

# workingDirectory = "~/drone_racing_ws/catkin_ddr/src/basic_rl_agent/data/dataset"
workingDirectory = '/home/majd/catkin_ws/src/basic_rl_agent/data2/flightgoggles/datasets/imageBezier_updated_datasets/imageBezierData_1000_DImg3_DTwst10'

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
    except Exception as e:
        print('{} is not a valid dataset file. skipped'.format(txt_file))
        print(e)
        return 
    try:
        vel_df = processVelocityData(txt_file)
    except:
        print('{} does not have twist data. skipped'.format(txt_file))
        return

    indices, images, Px, Py, Pz, Yaw = dataReader.getSamples()

    # # debugging:
    # print('processing file {}'.format(txt_file))
    # for idx, imageName in zip(indices, images):
    #     print(idx, imageName)

    numOfSamples = dataReader.getNumOfSamples()
    dt = dataReader.getDt()
    sample_length = dataReader.sample_length

    t = np.linspace(0, sample_length*dt, sample_length)
    t = t[:, np.newaxis].astype(np.float64)
    A = np.concatenate((t, np.power(t, 2), np.power(t, 3), np.power(t, 4)), axis=1)
    BernsteinToMonomial = np.array([[1, -4, 6, -4, 1], [0, 4, -12, 12, -4], [0, 0, 6, -12, 6], [0, 0, 0, 4, -4], [0, 0, 0, 0, 1]], dtype=np.float64)
    monomialToBernstein = la.inv(BernsteinToMonomial)
    A_yaw = A[:, 0:2] 
    monomialToBernstein_yaw = np.array([[1, 1, 1], [0, 0.5, 1], [0, 0, 1]], dtype=np.float64)
    n = 5
    positionControlPointsList = []
    yawControlPointsList = []
    imagesList = []

    Pxc = np.array(Px)
    Pyc = np.array(Py)
    Pzc = np.array(Pz)
    # print(indices, len(Px), len(Py), len(Pz))
    for index in range(len(Px)):
        px = Pxc[index, :] - Pxc[index, 0]
        py = Pyc[index, :] - Pyc[index, 0]
        pz = Pzc[index, :] - Pzc[index, 0]
        Pxyz = np.vstack([px, py, pz])

        # rotate the points according to the current yaw
        yaw = Yaw[index][0] * -1 
        rotationMatrix = Rotation.from_euler('z', yaw).as_dcm()

        Pxyz_rotated = np.matmul(rotationMatrix, Pxyz)

        C_list = []
        for pi in Pxyz_rotated:
            b = pi.astype(np.float64)
            monomial_coeff = solve_opt_problem(A, b, t, n-1) # n-1 we reduced the opt variables since a0=0.
            monomial_coeff = np.insert(monomial_coeff, 0, 0)
            C = np.matmul(monomial_coeff, monomialToBernstein)
            C = C.astype(np.float32)
            C_list.append(C)
            # print("the Bernstein coefficients: ", C)
        positionControlPoints_i = list(zip(C_list[0], C_list[1], C_list[2]))
        positionControlPointsList.append(positionControlPoints_i)

        imagesList.append(images[index])

        # computing Yaw control points
        p_yaw = np.array(Yaw[index])
        p_yaw = p_yaw - p_yaw[0]
        b_yaw = p_yaw.astype(np.float64)
        monomial_coeff_yaw = solve_opt_problem(A_yaw, b_yaw, t, 2)
        monomial_coeff_yaw = np.insert(monomial_coeff_yaw, 0, 0)
        C_yaw = np.matmul(monomial_coeff_yaw, monomialToBernstein_yaw)
        C_yaw = C_yaw.astype(np.float32)
        yawControlPointsList.append(C_yaw)

    dataPointsDict = {
        'images': imagesList,
        'positionControlPoints': positionControlPointsList,
        'yawControlPoints': yawControlPointsList
    }
    images_controlpoints_df = pd.DataFrame(dataPointsDict, columns = ['images', 'positionControlPoints', 'yawControlPoints'])
    df = pd.concat([images_controlpoints_df, vel_df], axis=1)

    # saving files:
    fileToSave = '{}_preprocessed.pkl'.format(txt_file)
    df.to_pickle(fileToSave)
    print('{} was saved.'.format(fileToSave))

def __lookForFiles1():
    for folder in [folder for folder in os.listdir(workingDirectory) if os.path.isdir(os.path.join(workingDirectory, folder))]:
        list1 = [folder1 for folder1 in os.listdir(os.path.join(workingDirectory, folder)) if folder1=='data']
        for dataFolder in list1:
            path = os.path.join(workingDirectory, folder, dataFolder)
            txtFilesList = [file for file in os.listdir(path) if file.endswith('.txt')]
            pklFilesList = [file.split('_preprocessed.pkl')[0] for file in os.listdir(path) if file.endswith('_preprocessed.pkl')]
            for txtFile in txtFilesList:
                if not txtFile.split('.txt')[0] in pklFilesList:
                    processDatasetTxtHeader(os.path.join(path, txtFile) )

def __lookForFiles2():
    overwrite = False
    txtFilesList = [file for file in os.listdir(workingDirectory) if file.endswith('.txt')]
    pklFilesList = [file.split('_preprocessed.pkl')[0] for file in os.listdir(workingDirectory) if file.endswith('_preprocessed.pkl')]

    for txtFile in txtFilesList[:]:
        if overwrite or not txtFile.split('.txt')[0] in pklFilesList:
            processDatasetTxtHeader(os.path.join(workingDirectory, txtFile) )

def main():
    __lookForFiles1()
    # __lookForFiles2()

# def main_debug():
#     txtFilesList = [file for file in os.listdir(workingDirectory) if file.endswith('.txt')]
#     processDatasetTxtHeader(txtFilesList[0])

def run():
    while True:
        __lookForFiles1()
        print('sleeping...')
        time.sleep(5*60)


def signal_handler(sig, frame):
    sys.exit(0)   

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)

    run()
    # main()