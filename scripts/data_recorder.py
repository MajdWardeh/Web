import os
import numpy as np
import pandas as pd
import cv2
import rospy 
from std_msgs.msg import Empty as std_Empty
from sensor_msgs.msg import Image


class DataRecoder:

    def __init__(self, rootDir):
        rospy.init_node('data_recoder', anonymous=True)
        self.rootDir = rootDir

        self.camera_subs = rospy.Subscriber('/uav/camera/left/image_rect_color', Image, self.rgbCameraCallback, queue_size=1)
        self.startRecord_subs = rospy.Subscriber('/record/start', std_Empty, self.startRecordingCallback, queue_size=1 )
        self.stopRecord_subs = rospy.Subscriber('/record/stop', std_Empty, self.stopRecordingCallback, queue_size=1 )

    def rgbCameraCallback(self, image_message):
        print('image')

    def startRecordingCallback(self, image_message):
        print('startRec')


    def stopRecordingCallback(self, image_message):
        print('stopRec')

def main():
    benchmarkPosesRootDir = '/home/majd/catkin_ws/src/basic_rl_agent/data/deep_learning/benchmarks/benchmarkPosesFiles'
    fileName = 'benchmarkerPosesFile_#100_202205081959_38_copy.pkl'

    targetPosesHumandNumbers = [40, 51, 65, 97, 4, 7, 9, 11, 33, 37, 39,
                                42, 48, 50, 54, 55, 57, 58, 61, 63, 67, 76, 79, 84, 87, 91, 92, 98]
    targetPosesIdices = [i - 1 for i in targetPosesHumandNumbers]

    posesDataFrame = pd.read_pickle(
        os.path.join(benchmarkPosesRootDir, fileName))
    poses = posesDataFrame['poses'].tolist()
    targetPoses = [poses[i] for i in targetPosesIdices]

    df = pd.DataFrame({'poses': targetPoses})
    df_name = 'benchmarkerPosesFile_#100_202205081959_38E.pkl'
    df.to_pickle(os.path.join(benchmarkPosesRootDir, df_name))
    print('file: {} saved'.format(os.path.join(benchmarkPosesRootDir, df_name)))


if __name__ == '__main__':
    main()
