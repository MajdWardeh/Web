import numpy as np

def processMarkersMultiGate(markersMsg):
    gatesMarkersDict = {}
    for marker in markersMsg.markers:
        gate = marker.landmarkID.data
        if not gate in gatesMarkersDict.keys():
            gatesMarkersDict[gate] = np.zeros((4, 3))
        markerArray = gatesMarkersDict[gate]    
        markerId = int(marker.markerID.data)-1
        # markersList.append((marker.x, marker.y, marker.z))
        markerArray[markerId] = [marker.x, marker.y, marker.z]
    return gatesMarkersDict 
