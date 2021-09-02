# from mpl_toolkits import mplot3d
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import numpy as np
from numpy import linalg as la
from scipy.special import binom
from sympy import Symbol, Pow, diff, simplify, integrate, lambdify, expand
import cvxpy as cp
import cv2
import datetime
from store_read_data import Data_Reader
import matplotlib.pyplot as plt
import pandas as pd

# workingDirectory = "~/drone_racing_ws/catkin_ddr/src/basic_rl_agent/data/dataset"
workingDirectory = '/home/majd/catkin_ws/src/basic_rl_agent/data/stateAggregationDataFromTrackedTrajectories' # provide the data subfolder in the dataset root directory.

def processVelocityData(file_name):
    vel_df = pd.read_pickle('{}.pkl'.format(file_name))
    # vel_list = df['vel'].tolist()
    return vel_df

def mergeMarkersDataWithPreprocessedFile(pklFile):
    print('processing {}'.format(pklFile))
    pklFile_withoutExtention = pklFile.split('_preprocessed.pkl')[0]
    markersFile = '{}_markersData.pkl'.format(pklFile_withoutExtention)
    try:
        preprocessed_df = pd.read_pickle(pklFile)
    except Exception as e:
        print(e)
        print('{} was not found. skipped'.format(pklFile))
        return 
    try:
        markers_df = pd.read_pickle(markersFile)
    except Exception as e:
        print(e)
        print('{} was not found. skipped'.format(markersFile))
        return

    # processing markers_df
    imageMarkersDataDict = {}
    markersImageList = markers_df['images'].tolist()
    markersDataList = markers_df['markersData'].tolist()
    for i, imageName in enumerate(markersImageList):
        imageMarkersDataDict[imageName] = markersDataList[i]

    markersDataProcessedList = []
    for image_np in preprocessed_df['images'].tolist():
        numOfImages_Sequence = image_np.shape[0]
        numOfImages_Channels = image_np.shape[1]

        if numOfImages_Channels != 1:
            raise NotImplementedError

        # get the markersData for each image in the sequece
        markersForImageSequence = []
        for i in range(numOfImages_Sequence):
            image = image_np[i, 0]
            if image in imageMarkersDataDict:
                markersForImageSequence.append(imageMarkersDataDict[image])
            else:
                print('{} did not have a markersData, added zeros'.format(image))
                markersForImageSequence.append(np.zeros((4, 3)))
        markersForImageSequence = np.array(markersForImageSequence)

        # add the markersForImageSequence to the markersList
        markersDataProcessedList.append(np.array(markersForImageSequence))

    preprocessed_df['markersData'] = markersDataProcessedList
    fileToSave = '{}_preprocessedWithMarkersData.pkl'.format(pklFile_withoutExtention)
    preprocessed_df.to_pickle(fileToSave)
    print('{} was saved.'.format(fileToSave))

    # # debugging
    # df = preprocessed_df
    # imageList = df['images'].tolist()
    # markers = df['markersData'].tolist()
    # print(len(markers))
    # for i, imName_np in enumerate(imageList):
    #     imName = np.array2string(imName_np[0, 0])[1:-1]
    #     image = cv2.imread(imName)
    #     for marker in markers[i]:
    #         marker = marker.astype(np.int)
    #         image = cv2.circle(image, (marker[0], marker[1]), radius=3, color=(255, 0, 0), thickness=-1)
    #     cv2.imwrite('/home/majd/catkin_ws/src/basic_rl_agent/data/debuggingImages/debugImage{}.jpg'.format(datetime.datetime.today().strftime('%Y%m%d%H%M_%S%f')), image)


def __lookForFiles1():
    overwrite = False
    
    for folder in [folder for folder in os.listdir(workingDirectory) if os.path.isdir(os.path.join(workingDirectory, folder))]:
        list1 = [folder1 for folder1 in os.listdir(os.path.join(workingDirectory, folder)) if folder1=='data']
        for dataFolder in list1:
            path = os.path.join(workingDirectory, folder, dataFolder)
            preprocessed_pklFilesList = [file for file in os.listdir(path) if file.endswith('_preprocessed.pkl')]
            preprocessedWithMarkersDataFilesList = [file.split('_preprocessedWithMarkersData.pkl')[0] for file in os.listdir(path) if file.endswith('_preprocessedWithMarkersData.pkl')]

            for pklFile in preprocessed_pklFilesList:
                if overwrite or not pklFile.split('_preprocessed.pkl')[0] in preprocessedWithMarkersDataFilesList:
                    mergeMarkersDataWithPreprocessedFile(os.path.join(path, pklFile) )
                
def __lookForFiles2():
    overwrite = True
    preprocessed_pklFilesList = [file for file in os.listdir(workingDirectory) if file.endswith('_preprocessed.pkl')]
    preprocessedWithMarkersDataFilesList = [file.split('_preprocessedWithMarkersData.pkl')[0] for file in os.listdir(workingDirectory) if file.endswith('_preprocessedWithMarkersData.pkl')]

    for pklFile in preprocessed_pklFilesList:
        if overwrite or not pklFile.split('_preprocessed.pkl')[0] in preprocessedWithMarkersDataFilesList:
            mergeMarkersDataWithPreprocessedFile(os.path.join(workingDirectory, pklFile) )

def main():
    __lookForFiles1()


if __name__ == '__main__':
    main()