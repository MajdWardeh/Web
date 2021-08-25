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

workingDirectory = '/home/majd/catkin_ws/src/basic_rl_agent/data/debugging_data3'
saveDirectory = '/home/majd/catkin_ws/src/basic_rl_agent/data/debugging_data3'

def processPickleFiles(filesList):
    dataFrameList = []
    for pickle_File in filesList:
        dataFrameList.append(pd.read_pickle(pickle_File))
    allFilesDataFrame = pd.concat(dataFrameList, axis=0)
    allFilesDataFrame.reset_index(drop=True, inplace=True)
    # if saveDirectory does not exist, create it.
    if not os.path.exists(saveDirectory):
        os.mkdir(saveDirectory)
    # save the file
    fileToSave = os.path.join(saveDirectory, 'allDataWithMarkers.pkl')
    allFilesDataFrame.to_pickle(fileToSave)
    print(allFilesDataFrame)
    print('{} was saved.'.format(fileToSave))

def main():
    pickleFilesList = []
    for folder in [folder for folder in os.listdir(workingDirectory) if os.path.isdir(os.path.join(workingDirectory, folder))]:
        for folder1 in [folder1 for folder1 in os.listdir(os.path.join(workingDirectory, folder)) if folder1=='data']:
            path = os.path.join(workingDirectory, folder, folder1)
            for file in os.listdir(path):
                if file.endswith('_preprocessedWithMarkersData.pkl'):
                    pickleFilesList.append(os.path.join(path, file))
    processPickleFiles(pickleFilesList)

    # pickleFilesList = [os.path.join(workingDirectory, file) for file in os.listdir(workingDirectory) if file.endswith('_preprocessedWithMarkersData.pkl')]
    # processPickleFiles(pickleFilesList)


if __name__ == '__main__':
    main()