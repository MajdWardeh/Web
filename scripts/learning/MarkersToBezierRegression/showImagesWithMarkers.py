import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import numpy as np
import cv2
import pandas as pd

allDataFileWithMarkers = '/home/majd/catkin_ws/src/basic_rl_agent/data/debugging_data2/dataset_202108052105_53/allDataPklFile/allDataWithMarkers.pkl'

def main():
    df = pd.read_pickle(allDataFileWithMarkers)
    imageList = df['images'].tolist()
    markers = df['markersData'].tolist()
    for i, imName_np in enumerate(imageList):
        imName = np.array2string(imName_np[0, 0])[1:-1]
        print(imName)
        image = cv2.imread(imName)
        for marker in markers[i]:
            marker = marker.astype(np.int)
            image = cv2.circle(image, (marker[0], marker[1]), radius=3, color=(255, 0, 0), thickness=-1)
        cv2.imshow('image', image)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            sys.exit()


if __name__ == '__main__':
    main()