from dis import dis
from logging import PlaceHolder
from select import POLLNVAL
import numpy as np
import cv2
import time

from numpy import disp
import rospy
from IrMarkersUtils import processMarkersMultiGate 
from flightgoggles.msg import IRMarkerArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError 

class GateDetector():

    def __init__(self):
        self.targetGate = 'gate0B'
        self.lastMarkerTid, self.currMarkerTid = None, None
        self.lastMarker, self.currMarker = None, None
        self.markerTidList = []
        self.markerDict = {}
        self.bridge = CvBridge()

        self.lastMarkerFound, self.currMarkerFound = False, False
        self.lastFound, self.found = False, False
        self.maxTidDiff = 1.1 * round(1000/60)
        self.skipCounter = 0
        self.SKIP_AMOUNT = 2
        self.displacementEma = [0] * 4

        rospy.init_node('gate_detector_node')
        rospy.Subscriber('/uav/camera/left/image_rect_color', Image, self.rgbCameraCallback, queue_size=2)
        rospy.Subscriber('/uav/camera/left/ir_beacons', IRMarkerArray, self.irMarkersCallback, queue_size=20)
        self.detectedGatePub = rospy.Publisher('/CNN_output/detectedGateImage', Image, queue_size=1)



    def irMarkersCallback(self, irMarkers_message):
        gatesMarkersDict = processMarkersMultiGate(irMarkers_message)
        if self.targetGate in gatesMarkersDict.keys():
            markersData = gatesMarkersDict[self.targetGate]
            tid = int(round(irMarkers_message.header.stamp.to_sec() * 1000))
            self.lastMarkerTid = self.currMarkerTid
            self.currMarkerTid = tid

            self.lastMarker = self.currMarker
            self.currMarker = markersData

            self.lastMarkerFound = self.currMarkerFound
            self.currMarkerFound = True
        else:
            self.lastMarkerFound = self.currMarkerFound
            self.currMarkerFound = False

        



    def rgbCameraCallback(self, image_message):
        tid = int(round(image_message.header.stamp.to_sec() * 1000))
        currImage = self.bridge.imgmsg_to_cv2(image_message, desired_encoding='bgr8')
        time.sleep((1/60)*0.4) # 60 fps, we want 10% wait
        self.detectGate(tid, currImage)


    def detectGate(self, imageTid, currImage):
        
        diffOk = False
        if abs(imageTid - self.currMarkerTid) < self.maxTidDiff:
            diffOk = True

        skip = self.skipCounter > 0
        if self.skipCounter > 0:
            self.skipCounter -= 1

        newFound = False
        if self.lastMarkerFound == False and self.currMarkerFound == True:
            newFound = True
            self.skipCounter = self.SKIP_AMOUNT


        plot = diffOk and (not skip) and (not newFound)

        if plot:
            # check dispalcement
            opticalFlowAbs = np.linalg.norm(self.currMarker[:, :-1] - self.lastMarker[:, :-1])
            print(self.currMarker[:, -1].mean(), opticalFlowAbs)

            self.emaWeight = 0.4

            for i, m in enumerate(self.currMarker):
                if m[-1] == 0:
                    continue
                # add noise
                depth = m[-1]
                displarcemnt = self.generateDisplacement()
                if newFound:
                    self.displacementEma[i] = displarcemnt
                else:
                    if opticalFlowAbs > 1.5:
                        self.displacementEma[i] = self.emaWeight * self.displacementEma[i] + (1-self.emaWeight) * displarcemnt
                
                d = self.displacementEma[i]

                m = m[:-1].astype(np.int) + np.round(d * (3/depth)).astype(np.int)
                currImage = cv2.circle(currImage, (m[0], m[1]), 5, (0, 0, 255), -1)
        
        currImage = cv2.putText(
                    currImage,
                    text = "Perception Network Output",
                    org = (currImage.shape[1]//12, currImage.shape[0]//12),
                    fontFace = cv2.FONT_HERSHEY_DUPLEX,
                    fontScale = 0.8,
                    color = (125, 246, 55),
                    thickness = 2
                    )
        

        try:
            imageMsg = self.bridge.cv2_to_imgmsg(currImage, "bgr8")
        except CvBridgeError as e:
            print(e)

        self.detectedGatePub.publish(imageMsg)

    def generateDisplacement(self):
        placement_x = np.random.choice(np.arange(6), p=[0.15, 0.35, 0.25, 0.15, 0.05, 0.05]) 
        placement_y = np.random.choice(np.arange(6), p=[0.15, 0.35, 0.25, 0.15, 0.05, 0.05]) 

        if(np.random.rand() > 0.5):
            placement_x *= -1

        if(np.random.rand() > 0.5):
            placement_y *= -1
        return np.array([placement_x, placement_y]) / 1.25


def main():
    gateDetector = GateDetector()
    rospy.spin()


if __name__ == '__main__':
    main()