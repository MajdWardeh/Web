import os
import numpy as np
import cv2
import pandas as pd
from markerDataCollector import ImageMarkersDataLoader

def processOneInstace(image, markersData, debugging=False):
    # check if all the markers are spotted
    for marker in markersData:
        if (marker == [0, 0, 0]).all():
            return None

    # find out the opposite points M0 <-> M2
    M0 = markersData[0]
    distances = markersData[1:] - M0
    distances = distances[:, :-1]
    distances = distances**2
    ds = np.sum(distances, axis=1)
    M2_index = np.argmax(ds) + 1
    M2 = markersData[M2_index]
    M1, M3 = [markersData[idx] for idx in [1, 2, 3] if idx != M2_index]

    # calculating the lines paramters mi, pi where y = mi*x + pi
    m1 = (M0[1]-M2[1])/(M0[0]-M2[0])
    p1 = M0[1] - m1*M0[0]
    m2 = (M1[1]-M3[1])/(M1[0]-M3[0])
    p2 = M1[1] - m2*M1[0]

    # calculate the intersection point
    x_intersect = (p1 - p2)/(m2 - m1)
    y_intersect = m1 * x_intersect + p1
    M_intersect = np.array([x_intersect, y_intersect])

    # expand the markers linearly:
    K = 1.75
    markersExpanded = []
    for Mi in [M0, M1, M2, M3]:
        v = K*(Mi[:-1] - M_intersect) + M_intersect
        markersExpanded.append(v)
    markersExpanded = np.array(markersExpanded)
    
    # finding the coordinates of the object detection anchor box:
    x_min_box = np.min(markersExpanded[:, 0], axis=0)
    x_max_box = np.max(markersExpanded[:, 0], axis=0)
    y_min_box = np.min(markersExpanded[:, 1], axis=0)
    y_max_box = np.max(markersExpanded[:, 1], axis=0)
    # normailize the anchor coordiantes:
    height, width, channels = image.shape
    box_normalized = np.array([y_min_box/height, x_min_box/width, y_max_box/height, x_max_box/width])

    if debugging:
        x_min_box, x_max_box, y_min_box, y_max_box = map(int, [x_min_box, x_max_box, y_min_box, y_max_box])
        M0 = M0.astype(np.int)
        M1 = M1.astype(np.int)
        M2 = M2.astype(np.int)
        M3 = M3.astype(np.int)
        x_intersect = int(x_intersect)
        y_intersect = int(y_intersect)
        for i, marker in enumerate(markersData):
            c = map(int, marker)
            if i == 0 or i == M2_index:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            image = cv2.circle(image, (c[0], c[1]), 3, color, -1)
        for i, marker in enumerate(markersExpanded):
            c = map(int, marker)
            if i == 0 or i == M2_index:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            image = cv2.circle(image, (c[0], c[1]), 3, color, -1)
        cv2.line(image, (M0[0], M0[1]), (M2[0], M2[1]), (0, 255, 0), thickness=2)
        cv2.line(image, (M1[0], M1[1]), (M3[0], M3[1]), (0, 255, 0), thickness=2)
        image = cv2.circle(image, (x_intersect, y_intersect), 3, (255, 0, 0), -1)
        image = cv2.rectangle(image, (x_min_box, y_min_box), (x_max_box, y_max_box), (255, 0, 0), 2)
        cv2.imshow('image', image)
        cv2.waitKey(1000)
    return box_normalized


def main():
    imageMarkerDataRootDir = '/home/majd/catkin_ws/src/basic_rl_agent/data/imageMarkersData'
    imageMarkerDatasets = os.listdir(imageMarkerDataRootDir)
    for k, dataset in enumerate(imageMarkerDatasets):
        print('processing dataset #{}'.format(k))
        imageMarkersLoader = ImageMarkersDataLoader(os.path.join(imageMarkerDataRootDir, dataset))
        imageNameList, markersDataList, poseDataList = imageMarkersLoader.loadData()            
        # process one dataset
        anchorDataList = []
        imageNameList_anchor = []
        for i, imageName in enumerate(imageNameList):
            print('processing dataset#{}, image #{}'.format(k, i))
            image = imageMarkersLoader.loadImage(imageName)
            makersData = markersDataList[i]
            anchorInstace = processOneInstace(image, markersDataList[i], debugging=False)
            if not anchorInstace is None:
                anchorDataList.append(anchorInstace)
                imageNameList_anchor.append(imageName)
        df_dict = {
            'images': imageNameList_anchor,
            'achorData': anchorDataList
            }
        df = pd.DataFrame(df_dict)
        markersDataPath = imageMarkersLoader.getPathsDict()['MarkersData']
        df.to_pickle(os.path.join(markersDataPath, 'MarkersAnchorData'))

    
    

if __name__ == '__main__':
    main()