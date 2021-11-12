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
        # remove the data with zeros markers
        self.__removeZerosMarkers()

    def __removeZerosMarkers(self):
        remove_indices = []
        for idx, markersData in enumerate(self.y_set):
            if (markersData[:, -1] == 0).any(): # check if the Z component of any marker is zeros.
                remove_indices.append(idx)
        x_set = []
        y_set = []
        for i in range(len(self.x_set)):
            if not i in remove_indices:
                x_set.append(self.x_set[i])
                y_set.append(self.y_set[i])
        self.x_set = x_set            
        self.y_set = y_set

    
    def __len__(self):
        return math.ceil(len(self.x_set) / self.batch_size)

    def __getitem__(self, index):
        '''
            Generates data containing batch_size samples 
            @return a list: [batch of gt_corners, batch of gt_pafs]
        '''
        images_batch = []
        gt_corners_batch = []
        gt_pafs_batch = []

        for row in range(min(self.batch_size, len(self.x_set)-index*self.batch_size)):
            image = cv2.imread(self.x_set[index*self.batch_size + row])
            if image is None:
                continue
            image = cv2.resize(image, (self.w, self.h))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32)

            markersData = self.y_set[index*self.batch_size + row]
            assert (markersData[:, -1] != 0).any(), 'markersData have Z component euqals to zero' # check if the Z component of any marker is zeros.

            gt_corners = self.markersPreprocessing.computeGroundTruthCorners(markersData)
            gt_pafs = self.markersPreprocessing.computeGroundTruthPartialAfinityFields(markersData)

            images_batch.append(image)
            gt_corners_batch.append(gt_corners)
            gt_pafs_batch.append(gt_pafs)

        images_batch = np.array(images_batch)
        gt_corners_batch = np.array(gt_corners_batch)
        gt_pafs_batch = np.array(gt_pafs_batch)
        # Normalize inputs
        images_batch = images_batch/255.
        return (images_batch, [gt_corners_batch, gt_pafs_batch])




def main():
    imageMarkerDataRootDir = '/home/majd/catkin_ws/src/basic_rl_agent/data/imageMarkersDataWithID'
    df = mergeDatasets(imageMarkerDataRootDir)
    df = df.sample(frac=0.2, random_state=0)
    Xset = df['images'].tolist()
    Yset = df['markersArrays'].tolist()
    batchSize = 5
    dataGen = CornerPAFsDataGenerator(Xset, Yset, batchSize, imageSize=(480, 640, 3), segma=7)
    print(dataGen.__len__())
    for i in [1]: #range(dataGen.__len__()):    
        Xbatch, Ybatch = dataGen.__getitem__(i)
        print('working on batch #{}'.format(i))
        idx_number = 2
        for idx, im in  [(idx_number, Xbatch[idx_number])]: #enumerate(Xbatch): # 
            print('idx={}'.format(idx))
            all_gt_labelImages = np.zeros(shape=(im.shape[0], im.shape[1], 3), dtype=np.uint8)
            gt_corners, gt_pafs = Ybatch[0], Ybatch[1]
            for j in range(4):
                gt_labelImage = gt_corners[idx, :, : , j]
                cv2.imshow("gt_labelImage{}".format(j), gt_labelImage)
                if j != 3:
                    all_gt_labelImages[:, :, j] += (gt_labelImage * 255).astype(np.uint8)
                else:
                    for m in range(3):
                        all_gt_labelImages[:, :, m] += (gt_labelImage * 255).astype(np.uint8)
            pafs_image = np.zeros_like(im)
            zeros = np.zeros((480, 640))
            for j in range(4):
                paf = gt_pafs[idx, :, :, 2*j:2*j+2]
                pafs_image[:, :, 0] += (np.maximum(zeros, paf[:, :, 0]) * 128).astype(np.uint8)
                pafs_image[:, :, 1] += (np.maximum(zeros, -paf[:, :, 0]) * 128).astype(np.uint8)
                pafs_image[:, :, 2] += (np.maximum(zeros, paf[:, :, 1]) * 128).astype(np.uint8)
                pafs_image[:, :, 2] += (np.maximum(zeros, -paf[:, :, 1]) * 128).astype(np.uint8)

            im1 = im.copy()
            im1[np.sum(all_gt_labelImages, axis=-1)!=0, 0] = 255
            im2 = pafs_image
            cv2.imshow('input image', cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            cv2.imshow('corners'.format(idx), all_gt_labelImages)
            cv2.imshow('pafs', im2)
            cv2.waitKey(0)

    

if __name__ == '__main__':
    main()