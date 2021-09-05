# from mpl_toolkits import mplot3d
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import datetime
import numpy as np
from numpy import linalg as la
from scipy.special import binom
from sympy import Symbol, Pow, diff, simplify, integrate, lambdify, expand
import cvxpy as cp
# import cv2
from store_read_data import Data_Reader
import matplotlib.pyplot as plt
import pandas as pd

workingDirectory = '/home/majd/catkin_ws/src/basic_rl_agent/data/stateAggregationDataFromTrackedTrajectories'
saveDirectory = '/home/majd/catkin_ws/src/basic_rl_agent/data/stateAggregationDataFromTrackedTrajectories'

def processPickleFiles(filesList, save_dir):
    dataFrameList = []
    for pickle_File in filesList:
        dataFrameList.append(pd.read_pickle(pickle_File))
    allFilesDataFrame = pd.concat(dataFrameList, axis=0)
    allFilesDataFrame.reset_index(drop=True, inplace=True)
    # if saveDirectory does not exist, create it.
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # save the file
    file_name = 'allData_{}_{}.pkl'.format(saveDirectory.split('/')[-1], datetime.datetime.now().strftime("%Y%m%d-%H%M"))
    fileToSave = os.path.join(save_dir, file_name)
    allFilesDataFrame.to_pickle(fileToSave)
    print(allFilesDataFrame)
    print('{} was saved.'.format(fileToSave))


def mergeDatasetPickles():
    pickleFilesList = []
    for folder in [folder for folder in os.listdir(workingDirectory) if os.path.isdir(os.path.join(workingDirectory, folder))]:
        for folder1 in [folder1 for folder1 in os.listdir(os.path.join(workingDirectory, folder)) if folder1=='data']:
            path = os.path.join(workingDirectory, folder, folder1)
            for file in os.listdir(path):
                if file.endswith('_preprocessedWithMarkersData.pkl'):
                    pickleFilesList.append(os.path.join(path, file))
    processPickleFiles(pickleFilesList, saveDirectory)

def mergeTwosPickles():
    pickle1 = '/home/majd/catkin_ws/src/basic_rl_agent/data/imageBezierData1/allDataWithMarkers.pkl' 
    pickle2 = '/home/majd/catkin_ws/src/basic_rl_agent/data/stateAggregationDataFromTrackedTrajectories/allDataWithMarkers.pkl' 
    filesList = [pickle1, pickle2]
    save_dir = '/home/majd/catkin_ws/src/basic_rl_agent/data/'
    processPickleFiles(filesList, save_dir)

def main():
    mergeDatasetPickles()

    # mergeTwosPickles()



    # pickleFilesList = [os.path.join(workingDirectory, file) for file in os.listdir(workingDirectory) if file.endswith('_preprocessedWithMarkersData.pkl')]
    # processPickleFiles(pickleFilesList)


if __name__ == '__main__':
    main()