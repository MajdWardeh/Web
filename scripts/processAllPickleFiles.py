# from mpl_toolkits import mplot3d
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
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

workingDirectory = '/home/majd/catkin_ws/src/basic_rl_agent/data/testing_data'

def processPickleFiles(filesList):
    dataFrameList = []
    for pickle_File in filesList:
        dataFrameList.append(pd.read_pickle(pickle_File))
    allFilesDataFrame = pd.concat(dataFrameList, axis=0)
    allFilesDataFrame.reset_index(drop=True, inplace=True)
    fileToSave = os.path.join(workingDirectory, 'allData.pkl')
    allFilesDataFrame.to_pickle(fileToSave)
    print(allFilesDataFrame)
    print('{} was saved.'.format(fileToSave))

def main():
    pickleFilesList = []
    for folder in os.listdir(workingDirectory):
        for file in os.listdir(os.path.join(workingDirectory, folder)):
            if file.endswith('_preprocessed.pkl'):
                pickleFilesList.append(os.path.join(workingDirectory, folder, file))
    processPickleFiles(pickleFilesList)

    # pickleFilesList = [file for file in os.listdir(workingDirectory) if file.endswith('_preprocessed.pkl')]
    # processPickleFiles(pickleFilesList)



if __name__ == '__main__':
    main()