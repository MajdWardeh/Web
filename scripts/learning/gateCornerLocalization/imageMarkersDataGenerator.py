import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import datetime
import numpy as np
import math
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Input, layers, Model, backend as k
from tensorflow.keras.utils import Sequence
from tensorflow.python.keras import models
from imageMarkers_dataPreprocessing import ImageMarkersGroundTruthPreprocessing
from imageMarkersDatasetsMerging import mergeDatasets

class CornerPAFsDataGenerator(Sequence):
    # TODO include PAFs
    # TODO data augmentation
    def __init__(self, x_set, y_set, batch_size, imageSize, segma=7, d=10):
        '''
            @param x_set: a list that contains the paths for images to be loaded.
            @param y_set: a list that contains the dataMarkers that correspond to the images in x_set.
            @param imageSize: the desired size of the images to be loaded. tuple that looks like (h, w, 3).
        
        '''
        self.x_set = x_set
        self.y_set = y_set
        self.batch_size = batch_size
        self.h, self.w  = imageSize[0], imageSize[1]
        self.markersPreprocessing = ImageMarkersGroundTruthPreprocessing(imageSize, cornerSegma=segma, d=d)
    def __len__(self):
        return math.ceil(len(self.x_set) / self.batch_size)

    def __getitem__(self, index):
        '''
            Generates data containing batch_size samples 
            @return gt_corners (for now)
            TODO return gt_corners, gt_pafs
        '''
        images_batch = []
        gt_corners_batch = []

        for row in range(min(self.batch_size, len(self.x_set)-index*self.batch_size)):
            image = cv2.imread(self.x_set[index*self.batch_size + row])
            if image is None:
                continue
            image = cv2.resize(image, (self.w, self.h))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32)

            markersData = self.y_set[index*self.batch_size + row]
            gt_corners = self.markersPreprocessing.computeGroundTruthCorners(markersData)
            images_batch.append(image)
            gt_corners_batch.append(gt_corners)

        images_batch = np.array(images_batch)
        gt_corners_batch = np.array(gt_corners_batch)
        # Normalize inputs
        images_batch = images_batch/255.
        return (images_batch, gt_corners_batch)




def main():
    imageMarkerDataRootDir = '/home/majd/catkin_ws/src/basic_rl_agent/data/imageMarkersDataWithID'
    df = mergeDatasets(imageMarkerDataRootDir)
    df = df.sample(frac=0.2)
    Xset = df['images'].tolist()
    Yset = df['markersArrays'].tolist()
    batchSize = 5
    dataGen = CornerPAFsDataGenerator(Xset, Yset, batchSize, imageSize=(480, 640, 3), segma=7)
    print(dataGen.__len__())
    for i in range(dataGen.__len__()):
        Xbatch, Ybatch = dataGen.__getitem__(i)
        print('working on batch #{}'.format(i))
        for idx, im in enumerate(Xbatch):
            all_gt_labelImages = np.zeros(shape=(im.shape[0], im.shape[1]), dtype=np.uint8)
            for j, gt_labelImage in enumerate(Ybatch[idx]):
                all_gt_labelImages += (gt_labelImage * 255).astype(np.uint8)
            im[all_gt_labelImages!=0, 0:1] = 255
            cv2.imshow('input image', im)
            cv2.imshow('all label image'.format(idx), all_gt_labelImages)
            cv2.waitKey(1000)

    

if __name__ == '__main__':
    main()