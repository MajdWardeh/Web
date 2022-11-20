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
    def __init__(self, x_set, y_set, batch_size, imageSize, segma=7, d=10, markersDataFactor=None, conrerToCornerMap=None):
        '''
            @param x_set: a list that contains the paths for images to be loaded.
            @param y_set: a list that contains the dataMarkers that correspond to the images in x_set.
            @param imageSize: the desired size of the images to be loaded. tuple that looks like (h, w, 3).
        
        '''
        self.x_set = x_set
        self.y_set = y_set
        self.batch_size = batch_size
        self.h, self.w  = imageSize[0], imageSize[1]
        assert markersDataFactor is not None, 'markersDataFactor must be provided, it [target_image_w/(original_image_w), target_h/(original_image_h), 1]'
        self.markersPreprocessing = ImageMarkersGroundTruthPreprocessing(imageSize, cornerSegma=segma, d=d, markersDataFactor=markersDataFactor, conrerToCornerMap=conrerToCornerMap)
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
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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

def testDataGenerator(dataGen, waitKeyValue=1000):
    print(dataGen.__len__())
    for i in range(dataGen.__len__()):
        Xbatch, Ybatch = dataGen.__getitem__(i)
        print('working on batch #{}'.format(i))
        for idx, im in enumerate(Xbatch):
            all_gt_labelImages = np.zeros(shape=(im.shape[0], im.shape[1]), dtype=np.uint8)
            gt_corners, gt_pafs = Ybatch[0], Ybatch[1]
            for j in range(4):
                gt_labelImage = gt_corners[idx, :, : , j]
                all_gt_labelImages += (gt_labelImage * 255).astype(np.uint8)
            pafs_image = np.zeros_like(im)
            zeros = np.zeros((480, 640))
            for j in range(4):
                paf = gt_pafs[idx, :, :, 2*j:2*j+2]
                pafs_image[:, :, 0] += (np.maximum(zeros, paf[:, :, 0]) * 128).astype(np.uint8)
                pafs_image[:, :, 1] += (np.maximum(zeros, -paf[:, :, 0]) * 128).astype(np.uint8)
                pafs_image[:, :, 2] += (np.maximum(zeros, paf[:, :, 1]) * 128).astype(np.uint8)
                pafs_image[:, :, 2] += (np.maximum(zeros, -paf[:, :, 1]) * 128).astype(np.uint8)

            im1 = im.copy()
            im1[all_gt_labelImages!=0, 0] = 255
            # im2 = cv2.cvtColor(pafs_image, cv2.COLOR_HSV2BGR)
            im2 = pafs_image
            # im2[:, :, 0] = 
            input_image = (cv2.cvtColor(im, cv2.COLOR_RGB2BGR) * 255.0).astype(np.uint8)
            all_gt_labelImages = cv2.cvtColor(all_gt_labelImages, cv2.COLOR_GRAY2BGR)
            all_image = cv2.addWeighted(input_image, 0.5, all_gt_labelImages, 0.5, 0.0)
            cv2.imshow('all_image', all_image)
            cv2.imshow('input image', input_image)
            # cv2.imshow('corners'.format(idx), all_gt_labelImages)
            cv2.imshow('pafs', im2)
            key = cv2.waitKey(waitKeyValue)
            if key == ord('q'):
                cv2.destroyAllWindows()
                return




def main():
    # imageMarkerDataRootDir = '/home/majd/catkin_ws/src/basic_rl_agent/data/imageMarkersDataWithID'
    imageMarkerDataRootDir = '/home/majd/catkin_ws/src/basic_rl_agent/data/imageMarkersDataWithDronePoses'
    df = mergeDatasets(imageMarkerDataRootDir)
    df = df.sample(frac=0.2)
    Xset = df['images'].tolist()
    Yset = df['markersArrays'].tolist()
    batchSize = 5
    image_size = (480, 640, 3)

    # check if the markers are normalized correctly:
    Yset_np = np.array(Yset)
    max_markerPixel_values = [Yset_np[:, :, i].max() for i in range(2)]
    print(max_markerPixel_values)

    markersDataFactor = [image_size[1]/640.0, image_size[0]/480.0, 1.0] # order: x, y, z so w, h, 1
    

    dataGen = CornerPAFsDataGenerator(Xset, Yset, batchSize, imageSize=image_size, segma=7, 
                                    markersDataFactor=markersDataFactor)
    testDataGenerator(dataGen)


if __name__ == '__main__':
    main()