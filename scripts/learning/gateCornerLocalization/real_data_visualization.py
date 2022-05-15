import os
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
import numpy as np
import yaml
from imageMarkers_dataPreprocessing import ImageMarkersGroundTruthPreprocessing 



def getImageMarkersDataLists(root_dir, images_dir, label_file):
    with open(os.path.join(root_dir, label_file), 'r') as stream:
        data_dict = yaml.safe_load(stream)
    
    # preprocessing object
    imageTargetSize = (480, 640, 3)
    d = 5
    markersDataFactor = np.ones((3, ))
    conrerToCornerMap = np.array([1, 2, 3, 0], dtype=np.int)
    markers_gt_preprocessing = ImageMarkersGroundTruthPreprocessing(imageShape=imageTargetSize, cornerSegma=7, d=d, markersDataFactor=markersDataFactor, conrerToCornerMap=conrerToCornerMap)

    for img_path, corners_list in data_dict.items():
        img_path = os.path.join(root_dir, img_path[img_path.find('/')+1:])
        img = cv2.imread(img_path)
        markersData = np.zeros((4, 3))
        for i, corner in enumerate(corners_list):
            y = corner[0] * img.shape[0]
            x = corner[1] * img.shape[1]
            markersData[i, 0] = x 
            markersData[i, 1] = y
            markersData[i, 2] = 1 # any value that is not equal to zero!

        gt_cornerImages = markers_gt_preprocessing.computeGroundTruthCorners(markersData)
        gt_h, gt_w, gt_ch = gt_cornerImages.shape
        gt_cornerImages_reshaped = np.zeros((gt_ch, gt_h, gt_w), dtype=gt_cornerImages.dtype)
        for i in range(gt_ch):
            gt_cornerImages_reshaped[i, :, :] = gt_cornerImages[:, :, i]


        markers_gt_preprocessing.debug_computeGroundTruthCorners(img, markersData, gt_cornerImages_reshaped, showCorners=True)
        


def main():
    root_dir = '/home/majd/papers/imagesLabeler'
    images_dir = 'images_resized_croped'
    label_file = 'output/images_resized_croped_labels.yaml'
    getImageMarkersDataLists(root_dir, images_dir, label_file)


if __name__ == '__main__':
    main()