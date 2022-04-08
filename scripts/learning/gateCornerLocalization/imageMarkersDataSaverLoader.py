from __future__ import print_function
import os
import time, datetime 
from math import pi, atan2
import numpy as np
import pandas as pd
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2

class ImageMarkersDataSaver:
    def __init__(self, path):
        assert os.path.exists(path), 'provided path does not exit'
        self.path = path

        self.imageNamesList = []
        self.nameToImageDict = {}
        self.poseList = []
        self.markersArrayList = []
        self.samplesCount = 0
        self.dataSaved = False

    def addSample(self, imageName, image, markersArray, pose): 
        '''
            @param imageName: the name of the image to be saved.
            @param image: the corresponding image to be saved.
            @param markersArray: an np array of shape (4, 3), stores four rows each row contains (x, y, z). Where
                x, y are the location of the markers in the image, and z is the distace between the marker and the camera.
            @param pose: an np array of shape (7,). stores the pose of the drone when taking the image. the formate of the array is
                [x, y, z, qx, qy, qz, qw].
        '''
        assert markersArray.shape == (4, 3), 'markersArray shape does not equal to the expected one.'
        assert pose.shape == (7,), 'pose shape does not equal the expected one'

        self.imageNamesList.append(imageName)
        self.nameToImageDict[imageName] = image
        self.markersArrayList.append(markersArray)
        self.poseList.append(pose)
        self.samplesCount += 1
        return self.samplesCount
    
    def saveData(self):
        if self.dataSaved == False:
            print('saving data ...')
            self.dateID, self.imagesPath, self.dataPath = self.__createNewDirectory(self.path) 
            modifiedImageNameList = ['{}.jpg'.format(name) for name in self.imageNamesList]
            dataset = {
                'images': modifiedImageNameList,
                'markersArrays': self.markersArrayList,
                'poses': self.poseList
            }
            df = pd.DataFrame(dataset)
            df.to_pickle(os.path.join(self.dataPath, 'MarkersData_{}.pkl'.format(self.dateID) ) )
            for imageName in self.imageNamesList:
                image = self.nameToImageDict[imageName]
                cv2.imwrite(os.path.join(self.imagesPath, '{}.jpg'.format(imageName)), image)
                time.sleep(0.01)
            self.dataSaved = True
            return self.dateID

    def __createNewDirectory(self, base_path):
        dateId = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
        dir_name = 'ImageMarkersDataset_{}'.format(dateId)
        path = os.path.join(base_path, dir_name)
        os.makedirs(path)
        imagesPath = os.path.join(path, 'images')
        os.makedirs(imagesPath)
        dataPath = os.path.join(path, 'Markers_data')
        os.makedirs(dataPath)
        return dateId, imagesPath, dataPath

class ImageMarkersDataLoader:
    def __init__(self, basePath):
        self.imagesPath, self.markersDataPath = self.__processBasePath(basePath)

    def loadData(self):
        self.df = self.loadDataFrame()
        imageNamesList = self.df['images'].tolist()
        markersArrayList = self.df['markersArrays'].tolist()
        gatePosesList = self.df['gatePoses'].tolist()
        dronePosesList = self.df['dronePoses'].tolist()
        return imageNamesList, markersArrayList, gatePosesList, dronePosesList

    def loadImage(self, imageName):
        imageNameWithPath =  os.path.join(self.imagesPath, imageName)
        image = cv2.imread(imageNameWithPath)
        return image

    def loadDataFrame(self):
        pickles = os.listdir(self.markersDataPath)
        # find the 'MarkerData' pickle file
        markerDataPickle = None
        for pickle in pickles:
            if pickle.startswith('MarkersData'):
                markerDataPickle = pickle
                break
        # read data   
        assert not markerDataPickle is None, 'could not find MarkersData pickle file'
        df = pd.read_pickle(os.path.join(self.markersDataPath, markerDataPickle))
        return df
    
    def getPathsDict(self):
        '''
            @return dictionary with keys: 'images', 'markersData'
        '''
        pathDict = {
            'Images': self.imagesPath,
            'MarkersData': self.markersDataPath
        }
        return pathDict

    def __processBasePath(self, basePath):
        imagesPath = os.path.join(basePath, 'images')
        markersDataPath = os.path.join(basePath, 'Markers_data')
        for path in [basePath, imagesPath, markersDataPath]:
            assert os.path.exists(path), 'path {} does not exist'.format(path)
        return imagesPath, markersDataPath

def testing_ImageMarkerDataSaver():
    path='/home/majd/catkin_ws/src/basic_rl_agent/data/test_imageMarkersData'
    imageMarkerSaver = ImageMarkersDataSaver(path)
    imageNameList = []
    markersArrayList = []
    posesList = []
    for i in range(10):
        image = np.random.randint(low=0, high=255, size=(224, 224, 3))
        imageName = 'image{}'.format(i)
        markersArray = np.random.rand(4, 3)
        pose = np.random.rand(7)

        # add data to list
        imageNameList.append('{}.jpg'.format(imageName))
        markersArrayList.append(markersArray)
        posesList.append(pose)
    
        imageMarkerSaver.addSample(imageName, image, markersArray, pose)

    dateID = imageMarkerSaver.saveData()

    path = os.path.join(path, 'ImageMarkersDataset_{}'.format(dateID))
    print('loading data from {}'.format(path))
    imageMarkerDataLoader = ImageMarkersDataLoader(path)
    loadedImageNameList, loadedMarkersArrayList, loadedPosesList = imageMarkerDataLoader.loadData()

    def compareTwoLists(l1, l2):
        sameLists = True 
        if len(l1) == len(l2):
            for i, v in enumerate(l1):
                if isinstance(v, np.ndarray):
                    if not (v == l2[i]).all():
                        sameLists = False
                        print(v, l2[i])
                        break
                else:
                    if v != l2[i]:
                        sameLists = False
                        print(v, l2[i])
                        break
        else:
            sameLists = False
            print('not the same length')
        assert sameLists, 'l1 and l2 are not the same'

    # compare:
    print('testing imageNameLists')
    compareTwoLists(imageNameList, loadedImageNameList)
    print('testing markersArrayLists')
    compareTwoLists(markersArrayList, loadedMarkersArrayList)
    print('testing posesLists')
    compareTwoLists(posesList, loadedPosesList)

    print('unitest passed for testing ImageMarkerDataSaver/Loader')