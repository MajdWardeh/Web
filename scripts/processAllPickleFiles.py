# from mpl_toolkits import mplot3d
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
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

workingDirectory = '/home/majd/catkin_ws/src/basic_rl_agent/data2/flightgoggles/datasets/imageBezier_updated_datasets/imageBezierData_1000_20FPS'
saveDirectory = '/home/majd/catkin_ws/src/basic_rl_agent/data2/flightgoggles/datasets/imageBezier_updated_datasets/imageBezierData_1000_20FPS'
# workingDirectory = '/home/majd/catkin_ws/src/basic_rl_agent/data/imageBezierData1'
# saveDirectory = '/home/majd/catkin_ws/src/basic_rl_agent/data/imageBezierData1'

def processPickleFiles(filesList, save_dir, file_name=None):
    dataFrameList = []
    for pickle_File in filesList:
        dataFrameList.append(pd.read_pickle(pickle_File))
    allFilesDataFrame = pd.concat(dataFrameList, axis=0)
    allFilesDataFrame.reset_index(drop=True, inplace=True)
    # if saveDirectory does not exist, create it.
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # save the file
    if file_name is None:
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
    ratio = 0.28
    end = int(round(len(pickleFilesList)*ratio))
    print(len(pickleFilesList), end)
    pickleFilesList = pickleFilesList[:end]
    processPickleFiles(pickleFilesList, saveDirectory)

def processPickleFileRatioTupleList(fileRatioTupleList, save_dir):
    dataFrameList = []
    for pickle_File, ratio in fileRatioTupleList:
        assert ratio > 0 and ratio <= 1.0 
        df = pd.read_pickle(pickle_File) 
        print('file: {}, rows: {}, ratio: {}, sampledRows: {}'.format(pickle_File.split('/')[-1], df.shape[0], ratio, df.shape[0]*ratio))
        df = df.sample(frac=ratio)
        dataFrameList.append(df)
    allFilesDataFrame = pd.concat(dataFrameList, axis=0)
    allFilesDataFrame.reset_index(drop=True, inplace=True)
    # if saveDirectory does not exist, create it.
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # save the file
    file_name = 'DataWithRatio_{}_{}.pkl'.format(saveDirectory.split('/')[-1], datetime.datetime.now().strftime("%Y%m%d-%H%M"))
    fileToSave = os.path.join(save_dir, file_name)
    allFilesDataFrame.to_pickle(fileToSave)
    print(allFilesDataFrame)
    print('{} was saved.'.format(fileToSave))

def mergeListOfPicklesFileWithRatio():
    filesRatioTupleList = [
            # '/home/majd/catkin_ws/src/basic_rl_agent/data/markersBezierData_highSpeed/allData_markersBezierData_highSpeed_20210906-2302.pkl', \
            ('/home/majd/catkin_ws/src/basic_rl_agent/data/imageBezierData1/imageToBezierData1.pkl', 1.0), \
            # '/home/majd/catkin_ws/src/basic_rl_agent/data/stateAggregationDataFromTrackedTrajectories/allData_trackedTrajectories_20210906-0000.pkl', \
            ('/home/majd/catkin_ws/src/basic_rl_agent/data2/flightgoggles/datasets/midPointData2/allData_midPointData2_20210909-1233.pkl', 1.0) \
                ]
    save_dir = '/home/majd/catkin_ws/src/basic_rl_agent/data2/flightgoggles'
    processPickleFileRatioTupleList(filesRatioTupleList, save_dir)

def main():
    mergeDatasetPickles()

    # mergeListOfPicklesFileWithRatio()



    # pickleFilesList = [os.path.join(workingDirectory, file) for file in os.listdir(workingDirectory) if file.endswith('_preprocessedWithMarkersData.pkl')]
    # processPickleFiles(pickleFilesList, saveDirectory)


if __name__ == '__main__':
    main()