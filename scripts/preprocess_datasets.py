# from mpl_toolkits import mplot3d
import os
import numpy as np
from numpy import linalg as la
from scipy.special import binom
from sympy import Symbol, Pow, diff, simplify, integrate, lambdify, expand
import cvxpy as cp
# import cv2
from store_read_data import Data_Reader
import matplotlib.pyplot as plt
import pandas as pd


# TODO:
# 1. visualize the trajectory samples [done]
# 2. 3d visualize the trajectory of x, y, z data. [done]
# 2.1 find out why doesn't a0 equal to 0? [done]
# 2.2 compute the 3d control points for the trajectory. [done]
# 2.3 store the results in panda file. [done]
# 2.4 read the file and make sure it is saved correctly. [done]
# 2.5 read all '.txt' dataset headerfiles and change them to pandas [done]
# 3. install the software on other machines.
# Regress control points of 4th order Beizer curves using CNNs:
#   1. collect a big dataset.
#   2.  


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

def processDatasetTxtHeader(txt_file):
    txt_file = txt_file.split('.txt')[0]
    dataReader = Data_Reader(txt_file)
    indices, images, Px, Py, Pz, Yaw = dataReader.getSamples()
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
    for index in indices:
        C_list = []
        for pi in [Px, Py, Pz]:
            pi = np.array(pi[index])
            pi = pi - pi[0]
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

    dataPointsDect = {
        'images': imagesList,
        'positionControlPoints': positionControlPointsList,
        'yawControlPoints': yawControlPointsList
    }
    df = pd.DataFrame(dataPointsDect, columns = ['images', 'positionControlPoints', 'yawControlPoints'])
    # print(df)
    # df.to_hdf('store.h5', 'table', append=True) 
    fileToSave = '{}.pkl'.format(txt_file)
    df.to_pickle(fileToSave)
    print('{} was saved.'.format(fileToSave))

def main():
    workingDirectory = '.'
    txtFilesList = [file for file in os.listdir(workingDirectory) if file.endswith('.txt')]
    for txtFile in txtFilesList:
        processDatasetTxtHeader(txtFile)

if __name__ == '__main__':
    main()