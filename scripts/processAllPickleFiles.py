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

# workingDirectory = "~/drone_racing_ws/catkin_ddr/src/basic_rl_agent/data/dataset"
workingDirectory = '.'


def processPickleFiles(filesList):
    dataFrameList = []
    for pickle_File in filesList:
        dataFrameList.append(pd.read_pickle(pickle_File))
    allFilesDataFrame = pd.concat(dataFrameList, axis=0)
    allFilesDataFrame.reset_index(drop=True, inplace=True)
    fileToSave = 'allData.pkl'
    allFilesDataFrame.to_pickle(fileToSave)
    print('{} was saved.'.format(fileToSave))

def main():
    pickleFilesList = [file for file in os.listdir(workingDirectory) if file.endswith('_preprocessed.pkl')]
    processPickleFiles(pickleFilesList)



if __name__ == '__main__':
    main()