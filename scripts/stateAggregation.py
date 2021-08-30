import os
import numpy as np

class State:
    '''
        a state is defined by:
            1. a sequence of images with their coresponding irMarkersData having the markers data of a target gate.
            2. a sequence of twist data on x, y, z and yaw
            3. [needed for computing the ground-truth data] the current position, twist, acc both the linear and angular.
    '''
    
    def __init__(self, imageSequence, markersDataSequence, twistDataSequence, pose, twist, acc, jerk, snap):
        self.imageSequence = imageSequence
        self.markersDataSequence = markersDataSequence
        self.twistDataSequence = twistDataSequence
        self.pose = pose
        self.twist = twist
        self.acc = acc
        self.jerk = jerk
        self.snap = snap



    


class StateAggregator:
    '''
        this class takes a state and finds the ground_truth bezier control points for the position and yaw (heading).
        It can also compare the predicted (if provided) to the computed gt data.
    '''
    def __init__(self, numOfImageSequence, numOfTwistSequence):
        self.numOfImageSequence = numOfImageSequence
        self.numOfTwistSequence = numOfTwistSequence
    
    def addState(self, imageSequence, markersDataSequence, twistDataSequence, pose, twist, acc, jerk, snap):
        pass

    def computeGroundTruthData():
        pass

def main():
    pass

if __name__ == '__main__':
    main()