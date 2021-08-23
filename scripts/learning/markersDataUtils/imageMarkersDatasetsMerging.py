import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
# sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import math
import numpy as np
import cv2
import numpy.linalg as lg
import pandas as pd
from imageMarkersDataSaverLoader import ImageMarkersDataLoader


def mergeDatasets(imageMarkerDataRootDir):
    '''
        1. load a dataset.
        2. modify the images data in the df to include the directories of the dataset name and the 'images' dir.
        3. merge the dfs of all the datasets.
    '''
    imageMarkerDatasets = os.listdir(imageMarkerDataRootDir)
    allDataFrameList = []
    for k, dataset in enumerate(imageMarkerDatasets):
        imageMarkersLoader = ImageMarkersDataLoader(os.path.join(imageMarkerDataRootDir, dataset))
        df = imageMarkersLoader.loadDataFrame()
        pathsDect = imageMarkersLoader.getPathsDict()

        # add full path to images
        imagesPath = pathsDect['Images']
        imagesList = df['images'].tolist()
        for i, image in enumerate(imagesList):
            imagesList[i] = os.path.join(imagesPath, image)
        df.drop('images', axis=1)
        df['images'] = imagesList

        ## normailze the markersData
        # markersDataList = df['markersData'].tolist()
        # markersNormalizer = np.array([1/1024.0, 1/768.0, 1.0])  #(768, 1024, 3) for (y, x, channels)
        # for i, markersData in enumerate(markersDataList):
        #     markersDataList[i] = np.multiply(markersData, markersNormalizer) 
        # df.drop('markersData', axis=1)
        # df['markersData'] = markersDataList

        # append df to allDataFrameList: 
        allDataFrameList.append(df)

    allDataFrames = pd.concat(allDataFrameList, ignore_index=True)
    # print(allDataFrames.columns)
    return allDataFrames

def debugAllDataFrames(df):
    df = df.sample(frac=0.2)
    imagesList = df['images'].tolist()
    markersDataList = df['markersArrays'].tolist()
    for i, imageName in enumerate(imagesList):
        im = cv2.imread(imageName)
        markers = markersDataList[i]
        for marker in markers:
            marker = marker.astype(np.int)
            im = cv2.circle(im, (marker[0], marker[1]), 4, (0, 0, 255), thickness=-1)
        cv2.imshow('image', im)
        cv2.waitKey(1000)



def main():
    imageMarkerDataRootDir = '/home/majd/catkin_ws/src/basic_rl_agent/data/imageMarkersDataWithID'
    allDataFrames = mergeDatasets(imageMarkerDataRootDir)
    # debugAllDataFrames(allDataFrames)

if __name__ == '__main__':
    main()