# from mpl_toolkits import mplot3d
from genericpath import isdir
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import time
import datetime
import signal
import numpy as np
from numpy import linalg as la
from scipy.spatial.transform import Rotation
from scipy.special import binom
import cvxpy as cp
import cv2
from store_read_data import Data_Reader
import matplotlib.pyplot as plt
import pandas as pd

from mergeMarkersDataWithProcessedDatasets import merge_preprocessed_df_with_markers_df


def debug_merged_df(df):
    imageSequenceList = df['images'].tolist()
    markersSequenceList = np.array(df['markersData'].tolist())

    print(markersSequenceList.shape)
    print(np.array(imageSequenceList).shape)
    ## change the image markers value from (640, 480) to (320, 240)
    markers_factor = np.array([240./480., 320./640., 1])
    markersSequenceList_reshaped = markersSequenceList.reshape(-1, 3)
    markersSequenceList_reshaped = np.multiply(markersSequenceList_reshaped, markers_factor)
    print(markersSequenceList_reshaped.max(axis=0))
    markersSequenceList = markersSequenceList_reshaped.reshape(markersSequenceList.shape)

    for s, sequence in enumerate(imageSequenceList):
        for i, imageName in enumerate(sequence):
            img = cv2.imread(imageName[0])
            print(img.shape)
            if img is None:
                print('image is None')
                print(img)
                continue
            markers = markersSequenceList[s, i, :, :-1].astype(np.int)
            for m in markers:
                cv2.circle(img, (m[0], m[1]), 5, (0,255,0), -1)
            cv2.imshow('image{}'.format(i), img)
        cv2.waitKey(300)


def processDatasetTxtHeader(txt_file):
    print('processing {}'.format(txt_file))

    txt_file = txt_file.split('.txt')[0]
    try:
        dataReader = Data_Reader(txt_file)
    except:
        print('{} is not a valid dataset file. skipped'.format(txt_file))
        return None 
    try:
        vel_df_name = txt_file + '.pkl'
        vel_df = pd.read_pickle(vel_df_name)
    except:
        print('{} does not have twist data. skipped'.format(txt_file))
        return None
    try:
        markers_df_name = txt_file + '_markersData.pkl'
        markers_df = pd.read_pickle(markers_df_name)
    except:
        print('{} does not have marekrs data. skipped'.format(txt_file))
        return None

    indices, images, Thrust, Px, Py, Pz = dataReader.getSamples()

    numOfSamples = dataReader.getNumOfSamples()
    sample_length = dataReader.sample_length

    imageSequenceList = []
    commandSequenceList = []

    Thrust = np.array(Thrust)
    Px = np.array(Px)
    Py = np.array(Py)
    Pz = np.array(Pz)
    for index in range(numOfSamples):
        thrust = Thrust[index, :]
        px = Px[index, :]
        py = Py[index, :]
        pz = Pz[index, :]
        commandSequence = np.vstack([thrust, px, py, pz])
        commandSequenceList.append(commandSequence)
        imageSequenceList.append(images[index])
    
    dataPointsDict = {
        'images': imageSequenceList,
        'commandSequence': commandSequenceList,
    }
    images_ControlCommand_df = pd.DataFrame(dataPointsDict, columns=['images', 'commandSequence'])

    # add the vel data from the vel pkl file
    images_ControlCommand_vel_df = pd.concat([images_ControlCommand_df, vel_df], axis=1)

    # add markersData from the marekrs pkl file
    images_ControlCommand_vel_markers_df = merge_preprocessed_df_with_markers_df(images_ControlCommand_vel_df, markers_df)

    assert images_ControlCommand_vel_markers_df is not None
    return images_ControlCommand_vel_markers_df

def processDatasetTxtHeaderList(txtHeaderList, save_dir):
    df_list = []
    for txt_file in txtHeaderList:
        df = processDatasetTxtHeader(txt_file)
        if df is not None:
            # debug_merged_df(df)
            df_list.append(df)
    print('merging all the data frames...')
    allFilesDataFrame = pd.concat(df_list, axis=0)
    allFilesDataFrame.reset_index(drop=True, inplace=True)
    # if saveDirectory does not exist, create it.
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # save the file
    file_name = 'allData_{}_rowsCount{}_{}.pkl'.format(save_dir.split('/')[-1], allFilesDataFrame.shape[0], datetime.datetime.now().strftime("%Y%m%d-%H%M"))
    fileToSave = os.path.join(save_dir, file_name)
    allFilesDataFrame.to_pickle(fileToSave)
    print(allFilesDataFrame)
    print('{} was saved.'.format(fileToSave))

def __lookForFiles1(workingDirectory):
    textHeadersList = []
    for folder in [folder for folder in os.listdir(workingDirectory) if os.path.isdir(os.path.join(workingDirectory, folder))]:
        list1 = [folder1 for folder1 in os.listdir(os.path.join(workingDirectory, folder)) if folder1=='data']
        for dataFolder in list1:
            path = os.path.join(workingDirectory, folder, dataFolder)
            txtFilesList = [file for file in os.listdir(path) if file.endswith('.txt')]
            for txtFile in txtFilesList:
                textHeadersList.append(os.path.join(path, txtFile))

    processDatasetTxtHeaderList(textHeadersList, workingDirectory)


def main():

    workingDirectory = '/home/majd/catkin_ws/src/basic_rl_agent/data2/flightgoggles/datasets/imageLowLevelControl' # provide the data subfolder in the dataset root directory.
    __lookForFiles1(workingDirectory)

if __name__ == '__main__':
    main()