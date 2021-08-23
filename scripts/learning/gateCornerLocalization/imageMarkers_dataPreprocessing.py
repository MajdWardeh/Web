import sys
from types import CoroutineType
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
# sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import math
from math import sqrt, log
import numpy as np
import cv2
import numpy.linalg as lg
import pandas as pd
from imageMarkersDataSaverLoader import ImageMarkersDataLoader

class ImageMarkersGroundTruthPreprocessing():
    def __init__(self, imageShape, cornerSegma=7, d=10):
        self._h, self._w, _ = imageShape
        self._cornerMask, cornerMaskCenter = self._generateCornerMask(cornerSegma)
        self.xc, self.yc = cornerMaskCenter
        self._d = d
        self._cornerToCornerMap = np.array([1, 3, 0, 2], dtype=np.int)
        self.markersDataFactor = np.array([self._w/1024.0, self._h/768.0, 1.0]) # 768, 1024 for H, W of the origianl image, 1.0 for the Z component


    def _generateCornerMask(self, segma=7):
        d = 2*int(round(segma * sqrt(log(10000)) /2 )) + 1 # find d that satisfies e^(-d^2/segma^2) = 0.0001. Also, d must always be an odd number
        mask = np.zeros((d, d), dtype=np.float32)
        maskCenter = np.array([mask.shape[1]//2, mask.shape[0]//2]) # x then y
        for j in range(mask.shape[0]):
            for i in range(mask.shape[1]):
                d = lg.norm((i, j)-maskCenter)
                mask[j, i] = math.exp(-d**2/segma**2)
        return mask, maskCenter


    def computeGroundTruthCorners(self, markersData):
        ''' 
            @param image: the image containing the gate.
            @param markersData: the data of the markers with shape=(numOfMarkers, 3)
            @param debugging: boolean, if True, the returned data will be plotted on the image. 
            @return gt_cornerImages, np array of shape (4, image.height, image.width) of type float32. for each corner class (0, 1, 2, 3), an image
            of all zeros but a Gaussian in the corresponding marker location.
        ''' 
        # 
        markersData = np.multiply(markersData, self.markersDataFactor)
        markersData = markersData[:, :-1].astype(np.int) # remove the z component and cast it to int
        gt_cornersImages = np.zeros((self._h, self._w, 4), dtype=np.float32) 
        for idx, marker in enumerate(markersData):
            xi, yi = marker[0:2]
            # check for x and y limits:
            # for x:
            xmin1 = xi-self.xc
            xmax1 = xi + self.xc + 1
            xmin = max(0, xmin1)
            xmax = min(self._w, xmax1) # self._w = image.shape[1]
            Mxmin = xmin-xmin1
            Mxmax = self._cornerMask.shape[1] - (xmax1-xmax)
            # for y:
            ymin1 = yi - self.yc
            ymax1 = yi + self.yc + 1 
            ymin = max(0, ymin1)
            ymax = min(self._h, ymax1) # self._h = image.shape[0]
            Mymin = ymin - ymin1
            Mymax = self._cornerMask.shape[0] - (ymax1-ymax)
            # debugging:
            # print('xmin1={}, xmin={}, xmax1={}, xmax={}, Mxmin={}, Mxmax={}'.format(xmin1, xmin, xmax1, xmax, Mxmin, Mxmax))
            # print('ymin1={}, ymin={}, ymax1={}, ymax={}, Mymin={}, Mymax={}'.format(ymin1, ymin, ymax1, ymax, Mymin, Mymax))
            # if Mxmin != 0 or Mymin != 0 or Mxmax != 19 or Mymax != 19:
            #     print('Mxmin={}, Mxmax={}, Mymin={}, Mymax={}'.format(Mxmin, Mxmax, Mymin, Mymax))
            gt_cornersImages[ymin:ymax, xmin:xmax, idx] = self._cornerMask[Mymin:Mymax, Mxmin:Mxmax]
        return gt_cornersImages

    def debug_computeGroundTruthCorners(self, image, markersData, gt_cornersImages, showCorners=True):
        raise NotImplementedError('changed the size of gt_cornersImages from (4, h, w) to (h, w, 4)')
        if showCorners:
            for cornerId, cornerImage in enumerate(gt_cornersImages):
                cv2.imshow('corner #{}'.format(cornerId), cornerImage)
        
        debugImage = image.copy()
        debugImage = debugImage//2
        debugImage = np.mean(debugImage, axis=2).astype(np.uint8)
        for cornerId, cornerImage in enumerate(gt_cornersImages):
            debugImage += (cornerImage * 128).astype(np.uint8)

        # ploting the markerIDs
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        fontColor = (0, 255, 0)
        lineType = 2
        for idx, marker in enumerate(markersData):
            marker = marker.astype(np.int)
            location = (marker[0], marker[1]) 
            cv2.putText(debugImage, str(idx), location, font, fontScale, fontColor, lineType)
        cv2.imshow('debugImage', debugImage)
        cv2.waitKey(1000)

    def computeGroundTruthPartialAfinityFields(self, markersData):
        markersData = np.multiply(markersData, self.markersDataFactor)
        X = markersData[:, :-1] # remove the z component
        gt_pafs = np.zeros((self._h, self._w, 4*2)) # gt_pafs[h, w, i] where i is even for x, and gt_pafs[h, w, j] where j is odd for y.
        for j1 in range(4):
            j2 = self._cornerToCornerMap[j1]
            vect = X[j2]-X[j1]
            l = lg.norm(vect)
            vect = vect / l # normalize the vector
            vect_orth = np.array([-vect[1], vect[0]]) # the orthogonal vector of vect, i` = -j, j` = i

            # compute Ps, the edgs of square with area equals to d*norm(Xj2-Xj1) (refer to the CMU paper).
            P1 = X[j1] + (vect_orth * self._d)
            P1_ = X[j1] + (vect_orth * -self._d)
            P2 = X[j2] + (vect_orth * self._d)
            P2_ = X[j2] + (vect_orth * -self._d)
            Ps = np.round([P1, P1_, P2, P2_]).astype(int)
            Pxmin, Pxmax, Pymin, Pymax = np.min(Ps[:, 0]), np.max(Ps[:, 0]), np.min(Ps[:, 1]), np.max(Ps[:, 1])

            # loop on all the points that are in the square and might saticfy the condition 
            for r in range(Pxmin, Pxmax+1):
                for c in range(Pymin, Pymax+1):
                    if r >= self._w or c >= self._h:
                        continue
                    val1 = np.dot(vect, ([r, c]-X[j1]) )
                    val2 = np.dot(vect_orth, ([r, c] - X[j1]) )
                    if val1>=0 and val1<=l and val2>=0 and val2<=self._d:
                        gt_pafs[c, r, 2*j1] = vect[0]
                        gt_pafs[c, r, 2*j1+1] = vect[1]
        return gt_pafs

    def debug_computeGroundTruthPartialAfinityFields(self, image, gt_pafs):
        zeros = np.zeros((self._h, self._w))
        pafs_image = np.zeros_like(image)
        # pafs_image[:, :, 1:2] = 255
        for j in range(4):
            paf = gt_pafs[:, :, 2*j:2*j+2]
            # color = np.arctan2(paf[:, :, 1], paf[:, :, 0])/np.pi
            # posColor = (np.maximum(zeros, color) * 255).astype(np.uint8)
            # negColor = (np.maximum(zeros, -1*color) * 250).astype(np.uint8)
            # pafs_image[np.logical_and(posColor > 0, posColor < 20), 1] *= 2
            # pafs_image[np.logical_and(negColor > 0, negColor < 20), 2] *= 2

            # pafs_image[:, :, 0] += posColor
            # pafs_image[:, :, 0] += negColor

            pafs_image[:, :, 0] += (np.maximum(zeros, paf[:, :, 0]) * 128).astype(np.uint8)
            pafs_image[:, :, 1] += (np.maximum(zeros, -paf[:, :, 0]) * 128).astype(np.uint8)
            pafs_image[:, :, 2] += (np.maximum(zeros, paf[:, :, 1]) * 128).astype(np.uint8)
            pafs_image[:, :, 2] += (np.maximum(zeros, -paf[:, :, 1]) * 128).astype(np.uint8)
        
        # pafs_image[pafs_image[:, :, 0]==0, 2] = 0
        im2 = image.copy()
        im2 = im2 // 2 + pafs_image
        cv2.imshow('pafs_image', pafs_image)
        cv2.imshow('image', image)
        cv2.imshow('im2', im2)
        cv2.waitKey(1000)



def main():
    imageMarkerDataRootDir = '/home/majd/catkin_ws/src/basic_rl_agent/data/imageMarkersDataWithID'
    imageMarkerDatasets = os.listdir(imageMarkerDataRootDir)

    for k, dataset in enumerate(imageMarkerDatasets[:1]):
        imageMarkersLoader = ImageMarkersDataLoader(os.path.join(imageMarkerDataRootDir, dataset))
        imageNameList, markersDataList, poseDataList = imageMarkersLoader.loadData()            
        imageTargetSize = (768, 1024, 3)
        # process one dataset
        markers_gt_preprocessing = ImageMarkersGroundTruthPreprocessing(imageShape=imageTargetSize, cornerSegma=9)

        for i, imageName in enumerate(imageNameList[:]):
            print('processing dataset#{}, image #{}'.format(k, i))
            image = imageMarkersLoader.loadImage(imageName)
            
            markersData = markersDataList[i]
            if (markersData[:, -1]==0).any():
                continue

            # gt_cornerImages = markers_gt_preprocessing.computeGroundTruthCorners(markersData)
            # markers_gt_preprocessing.debug_computeGroundTruthCorners(image, markersData, gt_cornerImages, showCorners=True)
            gt_pafs = markers_gt_preprocessing.computeGroundTruthPartialAfinityFields(markersData)
            markers_gt_preprocessing.debug_computeGroundTruthPartialAfinityFields(image, gt_pafs)

if __name__ == '__main__':
    main()